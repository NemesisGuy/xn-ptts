//! Voice activity (end-of-turn) readouts on the speech-recognition LM.
//!
//! Each readout head predicts, per step, whether the speaker has stopped for its horizon
//! (`vad_horizons`). Two layouts exist:
//!
//! - [`VadReadout::Linear`]: one `Linear(dim, classes)` per head on the final normed output
//!   (`extra_heads.{i}.weight`), the layout of the first checkpoints.
//! - [`VadReadout::Mixer`] (`vad_readout = "layer_mixer_residual_mlp"`): the residual stream
//!   after layers `extra_heads_from_layer` is layer-normed (no affine), mixed with per-step
//!   softmax weights from a small shared scorer plus a static per-layer bias, then each head
//!   runs pre-norm residual MLP blocks (`x += W_out gelu(W_in LayerNorm(x))`, exact erf GELU)
//!   and a bias-free classifier.
//!
//! Both are stateless: everything is a function of the current step's hidden states.
//! The probability reported is `softmax(logits)[vad_stopped_class]`.

use super::lm::{Config, LmStep};
use xn::nn::{Linear, var_builder::Path};
use xn::{Backend, Result, Tensor, WithDTypeF};

/// PyTorch's `LayerNorm` default, which the mixer readout was trained with. Not the trunk's
/// RMS norm epsilon.
const LAYER_NORM_EPS: f32 = 1e-5;

pub(crate) const MIXER_READOUT: &str = "layer_mixer_residual_mlp";

struct ResidualBlock<T: WithDTypeF, B: Backend> {
    norm_weight: Tensor<T, B>,
    norm_bias: Tensor<T, B>,
    linear_in: Linear<T, B>,
    linear_out: Linear<T, B>,
}

impl<T: WithDTypeF, B: Backend> ResidualBlock<T, B> {
    fn load(vb: &Path<B>, dim: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            norm_weight: vb.tensor("norm.weight", (dim,))?,
            norm_bias: vb.tensor("norm.bias", (dim,))?,
            linear_in: Linear::load(vb.pp("mlp").pp(0), dim, hidden)?,
            linear_out: Linear::load(vb.pp("mlp").pp(2), hidden, dim)?,
        })
    }

    fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let ys = xs.layer_norm(&self.norm_weight, &self.norm_bias, LAYER_NORM_EPS)?;
        let ys = self.linear_out.forward(&self.linear_in.forward(&ys)?.gelu_erf()?)?;
        xs.add(&ys)
    }
}

struct MixerHead<T: WithDTypeF, B: Backend> {
    blocks: Vec<ResidualBlock<T, B>>,
    classifier: Linear<T, B>,
}

pub(crate) struct Mixer<T: WithDTypeF, B: Backend> {
    layers: Vec<usize>,
    /// Ones and zeros: the per-tap norm has no affine parameters.
    ones: Tensor<T, B>,
    zeros: Tensor<T, B>,
    /// `(1, num_taps)` static logit bias.
    weights: Tensor<T, B>,
    /// `Linear(dim, s) + bias -> tanh -> Linear(s, 1)`, shared across taps. `None` for a
    /// static mixer.
    scorer: Option<(Linear<T, B>, Linear<T, B>)>,
    heads: Vec<MixerHead<T, B>>,
    dim: usize,
}

pub(crate) enum VadReadout<T: WithDTypeF, B: Backend> {
    Linear(Vec<Linear<T, B>>),
    Mixer(Box<Mixer<T, B>>),
}

impl<T: WithDTypeF, B: Backend> VadReadout<T, B> {
    /// The checkpoint's readout, or `None` when it has no VAD heads.
    pub(crate) fn load(vb: &Path<B>, cfg: &Config) -> Result<Option<Self>> {
        let Some((num_heads, classes)) = cfg.extra_heads() else { return Ok(None) };
        if cfg.vad_stopped_class >= classes {
            xn::bail!(
                "vad_stopped_class {} but heads have {classes} classes",
                cfg.vad_stopped_class
            )
        }
        let vb_h = vb.pp("extra_heads");
        match cfg.vad_readout.as_deref() {
            None => {
                if !cfg.extra_heads_from_layer.is_empty() {
                    xn::bail!(
                        "the config taps layers {:?} for its VAD heads but names no `vad_readout`",
                        cfg.extra_heads_from_layer
                    )
                }
                let heads = (0..num_heads)
                    .map(|i| Linear::load(vb_h.pp(i), cfg.dim, classes))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Some(Self::Linear(heads)))
            }
            Some(MIXER_READOUT) => {
                if cfg.extra_heads_mixer_affine {
                    xn::bail!("an affine VAD layer mixer is not supported")
                }
                let layers = cfg.extra_heads_from_layer.clone();
                if layers.is_empty() {
                    xn::bail!("`{MIXER_READOUT}` needs `extra_heads_from_layer`")
                }
                if let Some(l) = layers.iter().find(|&&l| l >= cfg.num_layers) {
                    xn::bail!("VAD tap layer {l} but the model has {} layers", cfg.num_layers)
                }
                let dim = cfg.dim;
                let hidden = cfg.extra_heads_hidden_dim.unwrap_or(dim);
                let dev = vb.device();
                let vb_m = vb.pp("extra_heads_mixer");
                let weights =
                    vb_m.tensor("weights", (layers.len(),))?.reshape((1, layers.len()))?;
                let scorer = if cfg.extra_heads_mixer_dynamic {
                    let vb_s = vb_m.pp("scorer");
                    // The scorer's width is not in the config; the checkpoint says.
                    let width = match vb_s.pp(0).shape("bias").map(|s| s.dims().to_vec()) {
                        Some(dims) if dims.len() == 1 => dims[0],
                        _ => xn::bail!("cannot read the VAD mixer scorer's width"),
                    };
                    let s0 = Linear::load_b(vb_s.pp(0), dim, width)?;
                    Some((s0, Linear::load(vb_s.pp(2), width, 1)?))
                } else {
                    None
                };
                let blocks = cfg.extra_heads_residual_blocks;
                let heads = (0..num_heads)
                    .map(|i| {
                        let vb_i = vb_h.pp(i);
                        Ok(MixerHead {
                            blocks: (0..blocks)
                                .map(|b| ResidualBlock::load(&vb_i.pp(b), dim, hidden))
                                .collect::<Result<Vec<_>>>()?,
                            classifier: Linear::load(vb_i.pp(blocks), dim, classes)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Some(Self::Mixer(Box::new(Mixer {
                    layers,
                    ones: Tensor::full(T::from_f32(1.0), (dim,), dev)?,
                    zeros: Tensor::zeros((dim,), dev)?,
                    weights,
                    scorer,
                    heads,
                    dim,
                }))))
            }
            Some(other) => xn::bail!("unsupported vad_readout {other:?}"),
        }
    }

    pub(crate) fn num_heads(&self) -> usize {
        match self {
            Self::Linear(heads) => heads.len(),
            Self::Mixer(m) => m.heads.len(),
        }
    }

    /// The layers whose outputs [`super::lm::LmModel::step`] has to hand over.
    pub(crate) fn tapped_layers(&self) -> &[usize] {
        match self {
            Self::Linear(_) => &[],
            Self::Mixer(m) => &m.layers,
        }
    }

    /// Per-head class probabilities, `(num_heads, classes)`, left on the device so the caller
    /// reads every head back at once.
    pub(crate) fn probs(&self, step: &LmStep<T, B>) -> Result<Tensor<T, B>> {
        let logits = match self {
            Self::Linear(heads) => {
                heads.iter().map(|h| h.forward(&step.out)).collect::<Result<Vec<_>>>()?
            }
            Self::Mixer(m) => m.logits(&step.taps)?,
        };
        let logits =
            logits.iter().map(|l| l.reshape((1, l.elem_count()))).collect::<Result<Vec<_>>>()?;
        Tensor::cat(&logits.iter().collect::<Vec<_>>(), 0)?.softmax()
    }
}

impl<T: WithDTypeF, B: Backend> Mixer<T, B> {
    /// The mixed representation `m`, `(1, dim)`.
    fn mix(&self, taps: &[Tensor<T, B>]) -> Result<Tensor<T, B>> {
        let n = self.layers.len();
        if taps.len() != n {
            xn::bail!("the VAD mixer expects {n} taps, got {}", taps.len())
        }
        let taps = taps.iter().map(|t| t.reshape((1, self.dim))).collect::<Result<Vec<_>>>()?;
        let taps = Tensor::cat(&taps.iter().collect::<Vec<_>>(), 0)?;
        let normed = taps.layer_norm(&self.ones, &self.zeros, LAYER_NORM_EPS)?;
        let scores = match &self.scorer {
            Some((s0, s2)) => s2.forward(&s0.forward(&normed)?.tanh()?)?.reshape((1, n))?,
            None => Tensor::zeros((1, n), normed.device())?,
        };
        let weights = scores.add(&self.weights)?.softmax()?;
        weights.matmul(&normed)
    }

    fn logits(&self, taps: &[Tensor<T, B>]) -> Result<Vec<Tensor<T, B>>> {
        let mixed = self.mix(taps)?;
        self.heads
            .iter()
            .map(|head| {
                let mut xs = mixed.clone();
                for block in head.blocks.iter() {
                    xs = block.forward(&xs)?;
                }
                head.classifier.forward(&xs)
            })
            .collect()
    }
}
