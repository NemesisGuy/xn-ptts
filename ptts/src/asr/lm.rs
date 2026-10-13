//! The speech-recognition language model: a causal transformer that consumes one Mimi
//! frame (`n_q` audio codes) plus the previously emitted text token per step and
//! predicts the next text token.
//!
//! This is the single-stream (`batch_size == 1`) subset of the moshi STT LM. There is no
//! depformer (the checkpoints have `dep_q = 0`, so nothing generates audio) and no batching
//! machinery.

use super::vad::VadReadout;
use crate::rope::RotaryEmbedding;
use crate::transformer::KvCache;
use std::collections::HashMap;
use xn::nn::{Embedding, Linear, var_builder::Path};
use xn::{Backend, BackendQ, Result, Tensor, WithDTypeF};

// ---- Config ----

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelId {
    pub sig: String,
    pub epoch: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LutConfig {
    pub n_bins: usize,
    pub dim: usize,
    #[serde(default)]
    pub possible_values: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConditionerConfig {
    Lut {
        lut: LutConfig,
    },
    /// Conditioner kinds this runtime does not implement. Keeping the catch-all means a
    /// sibling checkpoint still parses; such a conditioner is skipped with a warning.
    #[serde(other)]
    Unsupported,
}

/// The subset of a speech-recognition checkpoint's `config.json` this runtime reads.
/// Unknown fields (the depformer block, delays, ...) are ignored.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub card: usize,
    pub n_q: usize,
    pub dim: usize,
    pub text_card: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub hidden_scale: f64,
    pub context: usize,
    pub max_period: f64,
    /// How many frames the text stream lags the audio.
    pub asr_delay_in_tokens: usize,
    #[serde(default)]
    pub conditioners: HashMap<String, ConditionerConfig>,
    #[serde(default)]
    pub model_id: Option<ModelId>,

    // ---- Voice activity readout ----
    /// One horizon in seconds per VAD head, in head order.
    #[serde(default)]
    pub vad_horizons: Vec<f64>,
    /// `layer_mixer_residual_mlp` for the mixer readout; absent for one linear head per
    /// horizon on the final normed output.
    #[serde(default)]
    pub vad_readout: Option<String>,
    /// The class of each head's softmax that means "the speaker has stopped".
    #[serde(default)]
    pub vad_stopped_class: usize,
    #[serde(default)]
    pub extra_heads_num_heads: Option<usize>,
    #[serde(default)]
    pub extra_heads_dim: Option<usize>,
    /// Transformer layers (0-based) whose outputs the mixer readout reads.
    #[serde(default)]
    pub extra_heads_from_layer: Vec<usize>,
    #[serde(default)]
    pub extra_heads_hidden_dim: Option<usize>,
    #[serde(default)]
    pub extra_heads_residual_blocks: usize,
    #[serde(default)]
    pub extra_heads_mixer_affine: bool,
    #[serde(default)]
    pub extra_heads_mixer_dynamic: bool,
}

impl Config {
    fn dim_feedforward(&self) -> usize {
        (self.dim as f64 * self.hidden_scale) as usize
    }

    /// Width of each half of the gated MLP.
    fn gating_hidden(&self) -> usize {
        let ff = self.dim_feedforward();
        if ff == 4 * self.dim { 11 * self.dim / 4 } else { 2 * ff / 3 }
    }

    fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }

    pub fn audio_vocab_size(&self) -> usize {
        self.card + 1
    }

    pub fn text_in_vocab_size(&self) -> usize {
        self.text_card + 1
    }

    /// `(num_heads, classes)` of the VAD heads, when the checkpoint has any.
    pub(crate) fn extra_heads(&self) -> Option<(usize, usize)> {
        match (self.extra_heads_num_heads, self.extra_heads_dim) {
            (Some(n), Some(dim)) if n > 0 => Some((n, dim)),
            _ => None,
        }
    }

    /// `sig@epoch`, naming the checkpoint.
    pub fn model_ext(&self) -> Option<String> {
        self.model_id.as_ref().map(|id| format!("{}@{}", id.sig, id.epoch))
    }
}

// ---- Layers ----

/// RMS norm as moshi stores it: a single `alpha` gain, no mean removal.
struct RmsNorm<T: WithDTypeF, B: Backend> {
    alpha: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> RmsNorm<T, B> {
    const EPS: f32 = 1e-8;

    fn load(vb: &Path<B>, dim: usize) -> Result<Self> {
        let alpha = vb.tensor::<T>("alpha", (1, 1, dim))?.reshape((dim,))?;
        Ok(Self { alpha })
    }

    fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        xs.rms_norm(&self.alpha, Self::EPS)
    }
}

struct Attention<Q: BackendQ> {
    in_proj: Q::LinearQ,
    out_proj: Q::LinearQ,
    num_heads: usize,
    head_dim: usize,
    context: usize,
}

impl<Q: BackendQ> Attention<Q> {
    fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        let vb = vb.pp("self_attn");
        // `in_proj_weight` is renamed to `in_proj.weight` by `super::remap_key`.
        let in_proj = Q::linear_load(vb.pp("in_proj"), cfg.dim, 3 * cfg.dim)?;
        let out_proj = Q::linear_load(vb.pp("out_proj"), cfg.dim, cfg.dim)?;
        Ok(Self {
            in_proj,
            out_proj,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
            context: cfg.context,
        })
    }

    fn forward(
        &self,
        xs: &Tensor<Q::T, Q::B>,
        rope: &RotaryEmbedding<Q::T, Q::B>,
        cache: &mut KvCache<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;
        let (b, t, _) = xs.dims3()?;
        let (h, d) = (self.num_heads, self.head_dim);
        let dm = h * d;

        let qkv = self.in_proj.forward(xs)?;
        let q = qkv.narrow(2, 0..dm)?.contiguous()?.reshape((b, t, h, d))?;
        let k = qkv.narrow(2, dm..2 * dm)?.contiguous()?.reshape((b, t, h, d))?;
        let v = qkv.narrow(2, 2 * dm..3 * dm)?.contiguous()?.reshape((b, t, h, d))?;

        let (q, k) = rope.forward(&q, &k)?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (k, v) = cache.append(&k, &v)?;
        // The window is `context` positions including the current one.
        let seq = k.dim(2)?;
        let (k, v) = if seq > self.context {
            let window = seq - self.context..seq;
            (k.narrow(2, window.clone())?.contiguous()?, v.narrow(2, window)?.contiguous()?)
        } else {
            (k, v)
        };
        // One query per step and every cached key precedes it, so the causal mask is
        // all-zero and can be skipped.
        let scale = Q::T::from_f32(1.0 / (d as f32).sqrt());
        let attn = q.matmul_t(&k)?.scale(scale)?.softmax()?;
        let out = attn.matmul(&v)?;
        cache.trim()?;

        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, dm))?;
        self.out_proj.forward(&out)
    }
}

/// SwiGLU feed-forward: `linear_in` emits both halves, gate first.
struct Mlp<Q: BackendQ> {
    linear_in: Q::LinearQ,
    linear_out: Q::LinearQ,
    hidden: usize,
}

impl<Q: BackendQ> Mlp<Q> {
    fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        let hidden = cfg.gating_hidden();
        let vb = vb.pp("gating");
        let linear_in = Q::linear_load(vb.pp("linear_in"), cfg.dim, 2 * hidden)?;
        let linear_out = Q::linear_load(vb.pp("linear_out"), hidden, cfg.dim)?;
        Ok(Self { linear_in, linear_out, hidden })
    }

    fn forward(&self, xs: &Tensor<Q::T, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;
        let xs = self.linear_in.forward(xs)?;
        let gate = xs.narrow(2, 0..self.hidden)?.contiguous()?;
        let up = xs.narrow(2, self.hidden..2 * self.hidden)?.contiguous()?;
        self.linear_out.forward(&gate.silu()?.mul(&up)?)
    }
}

struct Layer<Q: BackendQ> {
    norm1: RmsNorm<Q::T, Q::B>,
    self_attn: Attention<Q>,
    norm2: RmsNorm<Q::T, Q::B>,
    mlp: Mlp<Q>,
}

impl<Q: BackendQ> Layer<Q> {
    fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        Ok(Self {
            norm1: RmsNorm::load(&vb.pp("norm1"), cfg.dim)?,
            self_attn: Attention::load(vb, cfg)?,
            norm2: RmsNorm::load(&vb.pp("norm2"), cfg.dim)?,
            mlp: Mlp::load(vb, cfg)?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor<Q::T, Q::B>,
        rope: &RotaryEmbedding<Q::T, Q::B>,
        cache: &mut KvCache<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let residual = self.self_attn.forward(&self.norm1.forward(xs)?, rope, cache)?;
        let xs = xs.add(&residual)?;
        let residual = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        xs.add(&residual)
    }
}

// ---- Conditioning ----

/// Lookup-table conditioner. The learnt padding embedding is appended to the table so "no
/// value given" is just one more index: that is what the model gets when no language is
/// pinned.
pub(crate) struct LutConditioner<T: WithDTypeF, B: Backend> {
    embed: Tensor<T, B>,
    possible_values: Vec<String>,
    padding_id: usize,
    output_dim: usize,
}

impl<T: WithDTypeF, B: Backend> LutConditioner<T, B> {
    fn load(vb: &Path<B>, cfg: &LutConfig, output_dim: usize) -> Result<Self> {
        let embed = vb.tensor::<T>("embed.weight", (cfg.n_bins + 1, cfg.dim))?;
        let proj = Linear::load(vb.pp("output_proj"), cfg.dim, output_dim)?;
        let embed = proj.forward(&embed)?;
        let learnt_padding = vb.tensor::<T>("learnt_padding", (1, 1, output_dim))?.squeeze(0)?;
        let embed = Tensor::cat(&[&embed, &learnt_padding], 0)?;
        Ok(Self {
            embed,
            possible_values: cfg.possible_values.clone(),
            padding_id: cfg.n_bins + 1,
            output_dim,
        })
    }

    /// `(1, 1, output_dim)` embedding for `value`, or the learnt padding when no value is
    /// given.
    pub(crate) fn condition(&self, value: Option<&str>) -> Result<Tensor<T, B>> {
        let index = match value {
            None => self.padding_id,
            Some(value) => {
                self.possible_values.iter().position(|v| v == value).ok_or_else(|| {
                    xn::Error::Msg(format!(
                        "unknown conditioner value {value:?}, expected one of {:?}",
                        self.possible_values
                    ))
                })?
            }
        };
        let index = Tensor::from_vec(vec![index as i64], (1,), self.embed.device())?;
        self.embed.index_select(&index, 0)?.reshape((1, 1, self.output_dim))
    }

    pub(crate) fn possible_values(&self) -> &[String] {
        &self.possible_values
    }
}

// ---- Model ----

pub(crate) struct LmState<T: WithDTypeF, B: Backend> {
    caches: Vec<KvCache<T, B>>,
    offset: usize,
}

/// What one LM step produces.
pub(crate) struct LmStep<T: WithDTypeF, B: Backend> {
    /// Text logits, `(1, 1, text_card)`.
    pub logits: Tensor<T, B>,
    /// The final normed transformer output, `(1, 1, dim)`.
    pub out: Tensor<T, B>,
    /// The residual stream after each layer the VAD readout taps, in tap order; each
    /// `(1, 1, dim)`. Empty when the readout reads `out` instead.
    pub taps: Vec<Tensor<T, B>>,
}

pub(crate) struct LmModel<Q: BackendQ> {
    layers: Vec<Layer<Q>>,
    text_emb: Embedding<Q::T, Q::B>,
    audio_embs: Vec<Embedding<Q::T, Q::B>>,
    out_norm: RmsNorm<Q::T, Q::B>,
    text_linear: Q::LinearQ,
    pub(crate) vad: Option<VadReadout<Q::T, Q::B>>,
    /// Keyed by conditioner name, e.g. `languages_in_segment`.
    pub(crate) conditioners: HashMap<String, LutConditioner<Q::T, Q::B>>,
    context: usize,
    head_dim: usize,
    max_period: f32,
    audio_vocab_size: usize,
    text_in_vocab_size: usize,
}

impl<Q: BackendQ> LmModel<Q> {
    pub(crate) fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        let vb_l = vb.pp("transformer").pp("layers");
        let layers = (0..cfg.num_layers)
            .map(|i| Layer::load(&vb_l.pp(i), cfg))
            .collect::<Result<Vec<_>>>()?;

        let text_emb = Embedding::load(vb.pp("text_emb"), cfg.text_in_vocab_size(), cfg.dim)?;
        let vb_e = vb.pp("emb");
        let audio_embs = (0..cfg.n_q)
            .map(|i| Embedding::load(vb_e.pp(i), cfg.audio_vocab_size(), cfg.dim))
            .collect::<Result<Vec<_>>>()?;

        let out_norm = RmsNorm::load(&vb.pp("out_norm"), cfg.dim)?;
        let text_linear = Q::linear_load(vb.pp("text_linear"), cfg.dim, cfg.text_card)?;
        let vad = VadReadout::load(vb, cfg)?;

        let vb_c = vb.pp("condition_provider").pp("conditioners");
        let mut conditioners = HashMap::new();
        for (name, cond) in cfg.conditioners.iter() {
            match cond {
                ConditionerConfig::Lut { lut } => {
                    conditioners
                        .insert(name.clone(), LutConditioner::load(&vb_c.pp(name), lut, cfg.dim)?);
                }
                ConditionerConfig::Unsupported => {
                    tracing::warn!(name, "ignoring unsupported conditioner")
                }
            }
        }

        Ok(Self {
            layers,
            text_emb,
            audio_embs,
            out_norm,
            text_linear,
            vad,
            conditioners,
            context: cfg.context,
            head_dim: cfg.head_dim(),
            max_period: cfg.max_period as f32,
            audio_vocab_size: cfg.audio_vocab_size(),
            text_in_vocab_size: cfg.text_in_vocab_size(),
        })
    }

    pub(crate) fn init_state(&self) -> LmState<Q::T, Q::B> {
        LmState {
            caches: self.layers.iter().map(|_| KvCache::new(self.context)).collect(),
            offset: 0,
        }
    }

    /// Codebook entry that stands in for "no audio yet".
    pub(crate) fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    /// Text token fed at the very first step.
    pub(crate) fn text_start_token(&self) -> u32 {
        self.text_in_vocab_size as u32 - 1
    }

    pub(crate) fn device(&self) -> &Q::B {
        self.text_emb.device()
    }

    /// One decoding step.
    pub(crate) fn step(
        &self,
        state: &mut LmState<Q::T, Q::B>,
        text_token: u32,
        audio_tokens: &[u32],
        condition: Option<&Tensor<Q::T, Q::B>>,
    ) -> Result<LmStep<Q::T, Q::B>> {
        use xn::ModuleT;
        if audio_tokens.len() != self.audio_embs.len() {
            xn::bail!("expected {} audio tokens, got {}", self.audio_embs.len(), audio_tokens.len())
        }
        let dev = self.device();
        let ids = Tensor::from_vec(vec![text_token as i64], 1, dev)?;
        let mut xs = self.text_emb.forward(&ids)?.unsqueeze(1)?;
        for (emb, &token) in self.audio_embs.iter().zip(audio_tokens.iter()) {
            let ids = Tensor::from_vec(vec![token as i64], 1, dev)?;
            xs = xs.add(&emb.forward(&ids)?.unsqueeze(1)?)?;
        }
        if let Some(condition) = condition {
            xs = xs.add(condition)?;
        }

        let tapped = self.vad.as_ref().map(|v| v.tapped_layers()).unwrap_or(&[]);
        let mut taps = Vec::with_capacity(tapped.len());
        let rope = RotaryEmbedding::new(self.head_dim, state.offset, 1, self.max_period, dev)?;
        for (idx, (layer, cache)) in self.layers.iter().zip(state.caches.iter_mut()).enumerate() {
            xs = layer.forward(&xs, &rope, cache)?;
            if tapped.contains(&idx) {
                taps.push((idx, xs.clone()));
            }
        }
        state.offset += 1;
        // Back into the order the readout was trained with, which need not be layer order.
        let taps = tapped
            .iter()
            .map(|l| taps.iter().find(|(idx, _)| idx == l).map(|(_, t)| t.clone()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| xn::Error::Msg("a VAD tap layer is out of range".into()))?;

        let out = self.out_norm.forward(&xs)?;
        let logits = self.text_linear.forward(&out)?;
        Ok(LmStep { logits, out, taps })
    }
}
