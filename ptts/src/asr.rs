//! Streaming speech recognition with per-step voice activity, behind the `asr` feature.
//!
//! The model is the moshi-STT layout: audio goes through the Mimi encoder 80 ms at a time,
//! each frame becomes `n_q` codebook indices, and a causal transformer consumes them
//! alongside the text token it emitted on the previous step. The text stream lags the audio
//! by `asr_delay_in_tokens` frames; words close when the model emits one of the
//! word-closing tokens.
//!
//! Checkpoints with VAD heads also report, every step, the probability that the speaker has
//! stopped for each horizon in `vad_horizons` (see [`VadPrediction`]); that is what a voice
//! agent's turn-taking reads.
//!
//! ```no_run
//! # fn main() -> ptts::Result<()> {
//! use ptts::asr::{Asr, FRAME_SIZE, read_config};
//!
//! let config = read_config("asr/config.json")?;
//! let asr = Asr::builder(config, "asr/model.safetensors", "asr/mimi.safetensors").build()?;
//! let mut stream = asr.stream(Some("en"))?;
//! let pcm = vec![0f32; FRAME_SIZE]; // 24 kHz mono, one frame at a time
//! let step = stream.step(&pcm)?;
//! for vad in step.vad {
//!     println!("{} s: {}", vad.horizon_s, vad.inactivity_prob);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Text comes back as token ids ([`Event::Word`]); decode them with the checkpoint's
//! tokenizer (`ptts::tok::Tok` on a `tokenizer.json` converted from its SentencePiece
//! `tokenizer.model` by `scripts/convert-tokenizer.py`).

mod lm;
mod quantizer;
mod vad;

pub use lm::{ConditionerConfig, Config, LutConfig, ModelId};

use crate::deep_copy::DeepCopy;
use crate::mimi::{MimiConfig, MimiEncoder, MimiEncoderState};
use crate::synth::{DeviceKind, Quant};
use crate::{Error, Result};
use lm::{LmModel, LmState};
use quantizer::MimiQuantizer;
use std::path::PathBuf;
use std::sync::Arc;
use xn::nn::VB;
use xn::nn::var_builder::Path;
use xn::{BackendQ, Tensor, Unquantized, WithDTypeF};

pub const SAMPLE_RATE: usize = 24_000;
pub const FRAME_RATE: f64 = 12.5;
/// Samples per model step: 80 ms at 24 kHz.
pub const FRAME_SIZE: usize = 1920;
/// Frames of silence a stream runs before the caller's first frame, as the serving worker
/// does. The first of them is replaced by the audio pad token.
pub const INITIAL_SILENCE_FRAMES: usize = 2;
/// LM steps, counted from the first silent frame, whose VAD output is meaningless: the
/// initial-token step, the leading silence and the first audio frame. [`AsrStream::step`]
/// reports [`WARMUP_INACTIVITY`] for them instead.
pub const VAD_WARMUP_STEPS: usize = 4;
/// What the VAD reports during [`VAD_WARMUP_STEPS`]: nobody has been heard speaking yet.
pub const WARMUP_INACTIVITY: f32 = 1.0;

/// Text tokens that close the current word.
pub const TOKEN_EOP: u32 = 0;
pub const TOKEN_EOS: u32 = 2;
pub const TOKEN_PAD: u32 = 3;
pub const TOKEN_SILENCE_PAD: u32 = 4;

/// What one step closed. Times are seconds of the caller's audio.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// A complete word, as tokenizer ids, starting at `start_time`.
    Word { tokens: Vec<u32>, start_time: f64 },
    /// The previous word ended at `stop_time`.
    EndWord { stop_time: f64 },
    /// The model emitted its end-of-stream token.
    EndOfStream,
}

/// The probability, at one step, that the speaker has stopped for `horizon_s`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadPrediction {
    /// Seconds: the head predicts "no further word within this horizon".
    pub horizon_s: f64,
    /// Near 0 while someone speaks, near 1 once they have stopped for the horizon.
    pub inactivity_prob: f32,
}

/// What [`AsrStream::step`] returns for one frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Step {
    pub events: Vec<Event>,
    /// One entry per VAD head in ascending horizon order; empty when the checkpoint has no
    /// heads.
    pub vad: Vec<VadPrediction>,
}

/// Turns the token stream into words, closing each one when the model says so.
#[derive(Debug, Clone)]
pub struct WordDecoder {
    delay: usize,
    step_idx: usize,
    word_tokens: Vec<u32>,
    unended_word: bool,
    last_stop_time: f64,
}

impl WordDecoder {
    pub fn new(delay: usize) -> Self {
        Self { delay, step_idx: 0, word_tokens: vec![], unended_word: false, last_stop_time: 0.0 }
    }

    /// Times are seconds since the first step.
    pub fn push(&mut self, token: u32) -> Vec<Event> {
        self.step_idx += 1;
        let mut events = vec![];
        if self.step_idx < self.delay {
            return events;
        }
        let closes_word = matches!(token, TOKEN_EOP | TOKEN_EOS | TOKEN_PAD | TOKEN_SILENCE_PAD);
        if closes_word {
            if !self.word_tokens.is_empty() {
                let tokens = std::mem::take(&mut self.word_tokens);
                self.unended_word = true;
                events.push(Event::Word { tokens, start_time: self.last_stop_time });
            }
        } else {
            self.word_tokens.push(token);
        }
        if token == TOKEN_EOP || token == TOKEN_EOS {
            let stop_time = (self.step_idx - self.delay) as f64 / FRAME_RATE;
            if self.unended_word {
                self.unended_word = false;
                events.push(Event::EndWord { stop_time });
            }
            self.last_stop_time = stop_time;
        }
        if token == TOKEN_EOS {
            events.push(Event::EndOfStream);
        }
        events
    }
}

/// Reads a speech-recognition checkpoint's `config.json`.
pub fn read_config(path: impl AsRef<std::path::Path>) -> Result<Config> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::not_found(format!("cannot read {}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| Error::invalid_data(format!("invalid ASR config {}: {e}", path.display())))
}

/// The Mimi codec configuration the speech-recognition checkpoints are trained against.
pub fn mimi_config() -> MimiConfig {
    MimiConfig {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        frame_rate: FRAME_RATE,
        dimension: 512,
        quantizer_dimension: 256,
        quantizer_output_dimension: 512,
        n_filters: 64,
        n_residual_layers: 1,
        ratios: vec![8, 6, 5, 4],
        kernel_size: 7,
        last_kernel_size: 3,
        residual_kernel_size: 3,
        dilation_base: 2,
        compress: 2,
        transformer_d_model: 512,
        transformer_num_heads: 8,
        transformer_num_layers: 8,
        transformer_layer_scale: 0.01,
        transformer_context: 250,
        transformer_max_period: 10000.0,
        transformer_dim_feedforward: 2048,
        downsample_channel_wise: false,
    }
}

/// Weight-name rewrites for the speech-recognition checkpoints (both `model.safetensors`
/// and `mimi.safetensors`).
pub fn remap_key(name: &str) -> Option<String> {
    let name = name.replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight");
    let name = name.replace(".conv.conv.", ".conv.");
    Some(name)
}

// ---- The model, generic over the weight format ----

struct AsrModel<Q: BackendQ> {
    mimi: MimiEncoder<Unquantized<f32, Q::B>>,
    quantizer: MimiQuantizer<f32, Q::B>,
    lm: LmModel<Q>,
    cfg: Config,
    /// One `(1, 1)` pad index per codebook, for the first step.
    audio_pad: Vec<Tensor<i64, Q::B>>,
    /// `vad_horizons` sorted ascending, with the head each reads.
    horizons: Vec<(f64, usize)>,
}

struct AsrState<Q: BackendQ> {
    enc: MimiEncoderState<f32, Q::B>,
    lm: LmState<Q::T, Q::B>,
    decoder: WordDecoder,
    condition: Option<Tensor<Q::T, Q::B>>,
    text_token: u32,
    /// LM steps run so far, the leading silence included.
    steps: usize,
}

impl<Q: BackendQ> AsrState<Q> {
    /// A copy that shares no tensor storage with `self`; see [`crate::deep_copy`].
    fn deep_copy(&self) -> Result<Self> {
        Ok(Self {
            enc: self.enc.deep_copy()?,
            lm: self.lm.deep_copy()?,
            decoder: self.decoder.clone(),
            condition: self.condition.deep_copy()?,
            text_token: self.text_token,
            steps: self.steps,
        })
    }
}

impl<Q: BackendQ> AsrModel<Q> {
    fn load(mimi_vb: &Path<Q::B>, lm_vb: &Path<Q::B>, cfg: Config) -> Result<Self> {
        let mimi_cfg = mimi_config();
        let mimi = MimiEncoder::load(mimi_vb, &mimi_cfg)?;
        let quantizer = MimiQuantizer::load(
            &mimi_vb.pp("quantizer"),
            mimi_cfg.dimension,
            mimi_cfg.quantizer_dimension,
            cfg.n_q,
            cfg.card,
        )?;
        let lm: LmModel<Q> = LmModel::load(lm_vb, &cfg)?;
        let num_heads = lm.vad.as_ref().map_or(0, |v| v.num_heads());
        let horizons = if num_heads == 0 {
            vec![]
        } else if cfg.vad_horizons.len() == num_heads {
            let mut h: Vec<_> = cfg.vad_horizons.iter().copied().zip(0..).collect();
            h.sort_by(|a, b| a.0.total_cmp(&b.0));
            h
        } else {
            return Err(Error::invalid_data(format!(
                "the checkpoint has {num_heads} VAD heads but `vad_horizons` lists {}",
                cfg.vad_horizons.len()
            )));
        };
        let pad = Tensor::from_vec(vec![lm.audio_pad_token() as i64], (1, 1), lm.device())?;
        let audio_pad = vec![pad; cfg.n_q];
        Ok(Self { mimi, quantizer, lm, cfg, audio_pad, horizons })
    }

    fn init_state(&self, language: Option<&str>) -> Result<AsrState<Q>> {
        // The model is conditioned on the spoken language. With none given it gets the learnt
        // padding and picks for itself.
        let condition = match self.lm.conditioners.get("languages_in_segment") {
            None => None,
            Some(conditioner) => {
                if let Some(language) = language
                    && !conditioner.possible_values().iter().any(|v| v == language)
                {
                    return Err(Error::invalid_argument(format!(
                        "unknown language {language:?}, expected one of {:?}",
                        conditioner.possible_values()
                    )));
                }
                Some(conditioner.condition(language)?)
            }
        };
        Ok(AsrState {
            enc: self.mimi.init_state(1)?,
            lm: self.lm.init_state(),
            decoder: WordDecoder::new(self.cfg.asr_delay_in_tokens),
            condition,
            text_token: self.lm.text_start_token(),
            steps: 0,
        })
    }

    /// One LM step over one frame of audio, blocking on both readbacks.
    fn step(&self, state: &mut AsrState<Q>, pcm: &[f32], temperature: f32) -> Result<Step> {
        let dev = self.lm.device();
        let audio = Tensor::from_vec(pcm.to_vec(), (1, 1, FRAME_SIZE), dev)?;
        let latent = self.mimi.encode_to_latent_step(&audio, &mut state.enc)?;
        // `(1, 1)` each, and left on the device: the LM only looks them up in its embeddings.
        let codes = self.quantizer.encode(&latent)?;
        // The first slice is dropped: the encoder has no history yet, so the model is handed
        // pad tokens instead.
        let audio_tokens = if state.steps == 0 { &self.audio_pad } else { &codes };
        let out = self.lm.step(
            &mut state.lm,
            state.text_token,
            audio_tokens,
            state.condition.as_ref(),
        )?;
        let logits = out.logits.reshape((1, self.cfg.text_card))?;
        let token = xn::nn::sampling::gumbel_max(&logits, temperature, 1)?;
        state.steps += 1;
        // Queued before either readback, so the device runs it while the host waits for the
        // token rather than after.
        let vad_probs = match &self.lm.vad {
            Some(readout) if state.steps > VAD_WARMUP_STEPS => Some(readout.probs(&out)?),
            _ => None,
        };
        let token = token.to_vec()?[0] as u32;
        state.text_token = token;

        let vad = match (&self.lm.vad, vad_probs) {
            (Some(_), Some(probs)) => {
                let classes = probs.dim(1usize)?;
                let probs = probs.to_vec()?;
                let stopped = self.cfg.vad_stopped_class;
                self.horizons
                    .iter()
                    .map(|&(horizon_s, head)| VadPrediction {
                        horizon_s,
                        inactivity_prob: probs[head * classes + stopped].to_f32(),
                    })
                    .collect()
            }
            (Some(_), None) => self
                .horizons
                .iter()
                .map(|&(horizon_s, _)| VadPrediction {
                    horizon_s,
                    inactivity_prob: WARMUP_INACTIVITY,
                })
                .collect(),
            (None, _) => vec![],
        };
        // The decoder counts from the first silent frame; report the caller's audio time.
        let lead = INITIAL_SILENCE_FRAMES as f64 / FRAME_RATE;
        let events = state
            .decoder
            .push(token)
            .into_iter()
            .map(|e| match e {
                Event::Word { tokens, start_time } => {
                    Event::Word { tokens, start_time: (start_time - lead).max(0.0) }
                }
                Event::EndWord { stop_time } => {
                    Event::EndWord { stop_time: (stop_time - lead).max(0.0) }
                }
                Event::EndOfStream => Event::EndOfStream,
            })
            .collect();
        Ok(Step { events, vad })
    }
}

// ---- Type erasure, so a flag can pick the weight format ----

trait AsrApi: Send + Sync {
    fn stream(&self, language: Option<&str>) -> Result<Box<dyn StreamApi>>;
    fn delay_frames(&self) -> usize;
    fn vad_horizons(&self) -> Vec<f64>;
    fn languages(&self) -> Vec<String>;
    fn model_ext(&self) -> Option<String>;
    fn device_name(&self) -> String;
}

trait StreamApi: Send {
    fn step(&mut self, pcm: &[f32]) -> Result<Step>;
    fn fork(&self) -> Result<Box<dyn StreamApi>>;
}

struct AsrOf<Q: BackendQ> {
    model: Arc<AsrModel<Q>>,
    temperature: f32,
}

struct StreamOf<Q: BackendQ> {
    model: Arc<AsrModel<Q>>,
    state: AsrState<Q>,
    temperature: f32,
}

impl<Q: BackendQ> AsrApi for AsrOf<Q>
where
    AsrModel<Q>: Send + Sync,
    AsrState<Q>: Send,
{
    fn stream(&self, language: Option<&str>) -> Result<Box<dyn StreamApi>> {
        let mut stream = StreamOf {
            model: self.model.clone(),
            state: self.model.init_state(language)?,
            temperature: self.temperature,
        };
        let silence = vec![0f32; FRAME_SIZE];
        for _ in 0..INITIAL_SILENCE_FRAMES {
            stream.model.step(&mut stream.state, &silence, stream.temperature)?;
        }
        Ok(Box::new(stream))
    }

    fn delay_frames(&self) -> usize {
        self.model.cfg.asr_delay_in_tokens
    }

    fn vad_horizons(&self) -> Vec<f64> {
        self.model.horizons.iter().map(|h| h.0).collect()
    }

    fn languages(&self) -> Vec<String> {
        match self.model.lm.conditioners.get("languages_in_segment") {
            Some(c) => c.possible_values().to_vec(),
            None => vec![],
        }
    }

    fn model_ext(&self) -> Option<String> {
        self.model.cfg.model_ext()
    }

    fn device_name(&self) -> String {
        use xn::Backend as _;
        self.model.lm.device().name()
    }
}

impl<Q: BackendQ> StreamApi for StreamOf<Q>
where
    AsrModel<Q>: Send + Sync,
    AsrState<Q>: Send,
{
    fn step(&mut self, pcm: &[f32]) -> Result<Step> {
        if pcm.len() != FRAME_SIZE {
            return Err(Error::invalid_argument(format!(
                "an ASR frame is {FRAME_SIZE} samples, got {}",
                pcm.len()
            )));
        }
        self.model.step(&mut self.state, pcm, self.temperature)
    }

    fn fork(&self) -> Result<Box<dyn StreamApi>> {
        Ok(Box::new(StreamOf {
            model: self.model.clone(),
            state: self.state.deep_copy()?,
            temperature: self.temperature,
        }))
    }
}

/// A loaded speech recognizer, shared by any number of [`AsrStream`]s.
#[derive(Clone)]
pub struct Asr(Arc<dyn AsrApi>);

/// One audio stream being transcribed. Holds the encoder and LM caches.
pub struct AsrStream(Box<dyn StreamApi>);

impl Asr {
    /// A builder over a checkpoint's config, LM weights and Mimi weights.
    pub fn builder(
        config: Config,
        model: impl Into<PathBuf>,
        mimi: impl Into<PathBuf>,
    ) -> AsrBuilder {
        AsrBuilder {
            config,
            model: model.into(),
            mimi: mimi.into(),
            device: DeviceKind::Cpu,
            quant: Quant::F32,
            dtype: None,
            temperature: 0.0,
        }
    }

    /// Start transcribing a stream. `language` pins the spoken language when the checkpoint
    /// is conditioned on one (see [`Self::languages`]); `None` lets the model pick.
    ///
    /// Runs the [`INITIAL_SILENCE_FRAMES`] before returning.
    pub fn stream(&self, language: Option<&str>) -> Result<AsrStream> {
        Ok(AsrStream(self.0.stream(language)?))
    }

    /// How many frames the text lags the audio. Feed that many frames of silence after the
    /// last audio to get its words out, or feed them to an [`AsrStream::fork`] to see those
    /// words without ending the stream.
    pub fn delay_frames(&self) -> usize {
        self.0.delay_frames()
    }

    /// The VAD horizons in seconds, ascending, in the order [`Step::vad`] reports them.
    pub fn vad_horizons(&self) -> Vec<f64> {
        self.0.vad_horizons()
    }

    /// Languages [`Self::stream`] accepts; empty when the checkpoint is not conditioned on
    /// one.
    pub fn languages(&self) -> Vec<String> {
        self.0.languages()
    }

    /// `sig@epoch` from the checkpoint's `model_id`, if it has one.
    pub fn model_ext(&self) -> Option<String> {
        self.0.model_ext()
    }

    /// Name of the device the model runs on, e.g. `"cpu"` or `"cuda:0"`.
    pub fn device_name(&self) -> String {
        self.0.device_name()
    }
}

impl AsrStream {
    /// Consume one frame of exactly [`FRAME_SIZE`] samples of 24 kHz mono audio.
    pub fn step(&mut self, pcm: &[f32]) -> Result<Step> {
        self.0.step(pcm)
    }

    /// An independent copy of the stream as it stands: the Mimi encoder's and the LM's caches,
    /// the word in progress and the step count. Stepping either one leaves the other exactly
    /// as it was, so a caller can run the text delay out on silence on a fork to see the words
    /// still in flight, and then drop it, without the live stream hearing that silence.
    ///
    /// Copies every cache (about 57 MB for `asr-67f06783.100`), which takes a few
    /// milliseconds on the CPU and about 0.2 ms on a desktop GPU; the model weights are shared.
    pub fn fork(&self) -> Result<AsrStream> {
        Ok(AsrStream(self.0.fork()?))
    }
}

/// What the LM and its VAD readout compute in on a GPU. The Mimi encoder and the quantizer
/// always run in f32, whatever the device.
///
/// The CPU computes in f32 only (its weight formats are [`AsrBuilder::quant`]'s business), so
/// asking for anything else there is an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    Bf16,
    F16,
}

impl Dtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
        }
    }
}

impl std::str::FromStr for Dtype {
    type Err = crate::Error;

    fn from_str(name: &str) -> crate::Result<Self> {
        match name.trim().to_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "bf16" => Ok(Self::Bf16),
            "f16" => Ok(Self::F16),
            _ => Err(Error::invalid_argument(format!(
                "unsupported ASR dtype '{name}'; expected f32, bf16 or f16"
            ))),
        }
    }
}

/// Loads an [`Asr`]. See [`Asr::builder`].
pub struct AsrBuilder {
    config: Config,
    model: PathBuf,
    mimi: PathBuf,
    device: DeviceKind,
    quant: Quant,
    dtype: Option<Dtype>,
    temperature: f32,
}

impl AsrBuilder {
    /// The CPU (the default) or CUDA; [`DeviceKind::Auto`] takes CUDA when this build has it.
    /// Vulkan and Metal are not supported.
    pub fn device(mut self, device: DeviceKind) -> Self {
        self.device = device;
        self
    }

    /// Weight format for the LM transformer linears, quantized at load time. Mimi and the
    /// VAD readout stay f32. Quantization is CPU-only.
    pub fn quant(mut self, quant: Quant) -> Self {
        self.quant = quant;
        self
    }

    /// What the LM computes in on a GPU; see [`Dtype`]. Unset, the device's default: f32
    /// everywhere.
    pub fn dtype(mut self, dtype: Dtype) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Text sampling temperature; 0 (the default) is greedy.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn build(self) -> Result<Asr> {
        let device = self.device.resolve();
        self.quant.check_device(device)?;
        match device {
            DeviceKind::Cpu => self.build_cpu(),
            DeviceKind::Cuda => self.build_cuda(),
            other => Err(Error::unsupported(format!(
                "speech recognition runs on the CPU or CUDA, not {other:?}"
            ))),
        }
    }

    fn build_cpu(self) -> Result<Asr> {
        if let Some(dtype) = self.dtype
            && dtype != Dtype::F32
        {
            return Err(Error::unsupported(format!(
                "speech recognition computes in f32 on the CPU, not {}; quantize the weights \
                 instead",
                dtype.as_str()
            )));
        }
        macro_rules! cpu {
            ($q:ty) => {
                self.load::<$q>(xn::CPU)
            };
        }
        match self.quant {
            Quant::F32 => cpu!(Unquantized<f32, xn::CpuDevice>),
            Quant::Q80 => cpu!(xn::quantized::Q80F32),
            Quant::Q81 => cpu!(xn::quantized::Q81F32),
            Quant::Q8k => cpu!(xn::quantized::Q8kF32),
            Quant::Q6k => cpu!(xn::quantized::Q6kF32),
            Quant::Q50 => cpu!(xn::quantized::Q50F32),
            Quant::Q51 => cpu!(xn::quantized::Q51F32),
            Quant::Q5k => cpu!(xn::quantized::Q5kF32),
            Quant::Q40 => cpu!(xn::quantized::Q40F32),
            Quant::Q41 => cpu!(xn::quantized::Q41F32),
            Quant::Q4k => cpu!(xn::quantized::Q4kF32),
        }
    }

    #[cfg(feature = "cuda")]
    fn build_cuda(self) -> Result<Asr> {
        // `check_device` keeps the weights unquantized off the CPU.
        debug_assert_eq!(self.quant, Quant::F32);
        let dev = xn::cuda_backend::Device::new(0)?;
        // The codes are an argmin over codebook distances, and the tf32 convolutions xn uses
        // for f32 by default move the Mimi latents enough to flip some of them: transcripts
        // still matched the CPU's on our clips, but VAD probabilities drifted by up to 0.06.
        // Full precision matches the CPU to 1e-4 and costs nothing measurable at one frame per
        // step. The switch is process-wide, so a TTS in the same process decodes with
        // full-precision convolutions too.
        xn::cuda_backend::set_conv_reduced_precision_f32(false);
        match self.dtype.unwrap_or(Dtype::F32) {
            Dtype::F32 => self.load::<Unquantized<f32, _>>(dev),
            Dtype::Bf16 => self.load::<Unquantized<half::bf16, _>>(dev),
            Dtype::F16 => self.load::<Unquantized<half::f16, _>>(dev),
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn build_cuda(self) -> Result<Asr> {
        Err(Error::unsupported("this build has no CUDA support; rebuild with the `cuda` feature"))
    }

    fn load<Q: BackendQ>(self, dev: Q::B) -> Result<Asr>
    where
        Q::T: WithDTypeF,
        AsrModel<Q>: Send + Sync,
        AsrState<Q>: Send,
    {
        for path in [&self.model, &self.mimi] {
            if !path.is_file() {
                return Err(Error::not_found(format!("no such file: {}", path.display())));
            }
        }
        let mimi_vb = VB::load_with_key_map(&[&self.mimi], dev.clone(), remap_key)?.root();
        let lm_vb = VB::load_with_key_map(&[&self.model], dev, remap_key)?.root();
        let model = AsrModel::<Q>::load(&mimi_vb, &lm_vb, self.config)?;
        Ok(Asr(Arc::new(AsrOf { model: Arc::new(model), temperature: self.temperature })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_close_on_pads_and_end_on_eop() {
        let mut d = WordDecoder::new(2);
        // Inside the delay nothing is emitted, whatever the token.
        assert!(d.push(100).is_empty());
        let mut d = WordDecoder::new(1);
        assert!(d.push(100).is_empty());
        assert!(d.push(101).is_empty());
        assert_eq!(
            d.push(TOKEN_PAD),
            vec![Event::Word { tokens: vec![100, 101], start_time: 0.0 }]
        );
        assert_eq!(d.push(TOKEN_EOP), vec![Event::EndWord { stop_time: 3.0 / FRAME_RATE }]);
        assert!(d.push(TOKEN_PAD).is_empty());
        assert_eq!(d.push(7), vec![]);
        assert_eq!(
            d.push(TOKEN_EOS),
            vec![
                Event::Word { tokens: vec![7], start_time: 3.0 / FRAME_RATE },
                Event::EndWord { stop_time: 6.0 / FRAME_RATE },
                Event::EndOfStream,
            ]
        );
    }

    /// The samples of a 16-bit PCM WAV file's first channel, as f32.
    fn read_wav_s16(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[..4], b"RIFF", "{} is not a WAV file", path.display());
        let (mut pos, mut channels) = (12, 1);
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
            match id {
                b"fmt " => {
                    assert_eq!(u16::from_le_bytes([body[0], body[1]]), 1, "not PCM");
                    channels = u16::from_le_bytes([body[2], body[3]]) as usize;
                    let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                    assert_eq!(rate as usize, SAMPLE_RATE, "resample to 24 kHz first");
                    assert_eq!(u16::from_le_bytes([body[14], body[15]]), 16, "not 16-bit");
                }
                b"data" => {
                    return body
                        .chunks_exact(2 * channels)
                        .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0)
                        .collect();
                }
                _ => {}
            }
            pos += 8 + len + (len & 1);
        }
        panic!("{} has no data chunk", path.display())
    }

    /// Run with `cargo test --release -p ptts --features asr -- --ignored fork`. Set
    /// `PTTS_TEST_ASR_DEVICE=cuda` (with the `cuda` feature) to run it on the GPU, and
    /// optionally `PTTS_TEST_ASR_DTYPE`; on the CPU the weights are `q8_0`.
    #[test]
    #[ignore = "requires PTTS_TEST_ASR_MODEL (a speech-recognition checkpoint folder) and \
                PTTS_TEST_ASR_AUDIO (24 kHz mono 16-bit WAV of speech)"]
    fn a_forked_stream_runs_independently_of_its_original() {
        let dir = std::path::PathBuf::from(std::env::var("PTTS_TEST_ASR_MODEL").unwrap());
        let audio = std::env::var("PTTS_TEST_ASR_AUDIO").unwrap();
        let device: DeviceKind = match std::env::var("PTTS_TEST_ASR_DEVICE") {
            Ok(name) => name.parse().unwrap(),
            Err(_) => DeviceKind::Cpu,
        };
        let mut builder = Asr::builder(
            read_config(dir.join("config.json")).unwrap(),
            dir.join("model.safetensors"),
            dir.join("mimi.safetensors"),
        )
        .device(device);
        if device.resolve() == DeviceKind::Cpu {
            builder = builder.quant(Quant::Q80);
        }
        if let Ok(dtype) = std::env::var("PTTS_TEST_ASR_DTYPE") {
            builder = builder.dtype(dtype.parse().unwrap());
        }
        let asr = builder.build().unwrap();
        let mut pcm = read_wav_s16(std::path::Path::new(&audio));
        pcm.resize(pcm.len().div_ceil(FRAME_SIZE) * FRAME_SIZE, 0.0);
        let frames: Vec<&[f32]> = pcm.chunks(FRAME_SIZE).collect();
        let silence = vec![0f32; FRAME_SIZE];

        let mut reference = asr.stream(Some("en")).unwrap();
        let expected: Vec<Step> = frames.iter().map(|f| reference.step(f).unwrap()).collect();
        let words = expected.iter().flat_map(|s| &s.events);
        assert!(words.filter(|e| matches!(e, Event::Word { .. })).count() > 3, "no speech?");

        // Fork every few frames and run the fork on silence: the original must not notice.
        // One fork instead gets the rest of the real audio, and must step exactly as the
        // original did, which it can only if the copy left no state behind.
        let mut live = asr.stream(Some("en")).unwrap();
        let mut forks_with_words = 0;
        for (i, frame) in frames.iter().enumerate() {
            if i % 5 == 2 {
                let mut fork = live.fork().unwrap();
                let mut events = vec![];
                for _ in 0..=asr.delay_frames() {
                    events.extend(fork.step(&silence).unwrap().events);
                }
                forks_with_words += events.iter().any(|e| matches!(e, Event::Word { .. })) as usize;
            }
            if i == frames.len() / 2 {
                let mut fork = live.fork().unwrap();
                for (j, frame) in frames.iter().enumerate().skip(i) {
                    assert_eq!(fork.step(frame).unwrap(), expected[j], "fork, frame {j}");
                }
            }
            assert_eq!(live.step(frame).unwrap(), expected[i], "live, frame {i}");
        }
        assert!(forks_with_words > 0, "the forks never flushed a word out");
    }

    #[test]
    fn a_gpu_dtype_or_backend_the_asr_cannot_use_is_refused_before_loading() {
        for name in ["f32", "bf16", "F16"] {
            let dtype: Dtype = name.parse().unwrap();
            assert_eq!(dtype.as_str(), name.to_lowercase());
        }
        assert!("q8".parse::<Dtype>().is_err());
        let cfg: Config = serde_json::from_str(
            r#"{"card":4,"n_q":2,"dim":8,"text_card":16,"num_heads":2,"num_layers":1,
                "hidden_scale":4.0,"context":8,"max_period":10000,"asr_delay_in_tokens":2}"#,
        )
        .unwrap();
        // None of these get as far as the (missing) weights.
        let builder = || Asr::builder(cfg.clone(), "no/model.safetensors", "no/mimi.safetensors");
        let refused = |r: Result<Asr>| matches!(r, Err(Error::Unsupported(_)));
        assert!(refused(builder().dtype(Dtype::Bf16).build()));
        assert!(refused(builder().device(DeviceKind::Metal).build()));
        if cfg!(not(feature = "cuda")) {
            assert!(refused(builder().device(DeviceKind::Cuda).build()));
        }
        assert!(refused(builder().device(DeviceKind::Cuda).quant(Quant::Q80).build()));
        assert!(matches!(builder().build(), Err(Error::NotFound(_))));
    }

    #[test]
    fn the_mixer_readout_config_parses() {
        let cfg: Config = serde_json::from_str(
            r#"{"card":2048,"n_q":32,"dim":1024,"text_card":20000,"num_heads":8,
                "num_layers":16,"hidden_scale":4.125,"context":375,"max_period":10000,
                "asr_delay_in_tokens":10,"extra_heads_num_heads":4,"extra_heads_dim":2,
                "vad_horizons":[0.5,1.0,2.0,3.0],"vad_readout":"layer_mixer_residual_mlp",
                "extra_heads_from_layer":[6,7,8,9,10,11,12,13],"extra_heads_hidden_dim":1024,
                "extra_heads_residual_blocks":3,"extra_heads_mixer_affine":false,
                "extra_heads_mixer_dynamic":true,"vad_stopped_class":0,
                "model_id":{"sig":"67f06783","epoch":100},"depformer_dim":1024}"#,
        )
        .unwrap();
        assert_eq!(cfg.extra_heads(), Some((4, 2)));
        assert_eq!(cfg.vad_readout.as_deref(), Some(vad::MIXER_READOUT));
        assert_eq!(cfg.model_ext().as_deref(), Some("67f06783@100"));
    }
}
