//! Locating and loading the speech-recognition checkpoint for `/speech/asr`.
//!
//! The handler sees the model only through [`Recognizer`], so tests can drive it with a fake.

use anyhow::{Context as _, Result};
use ptts::Tokenizer as _;
use ptts::asr::Dtype;
use ptts::asr::{Asr, Step};
use ptts::checkpoint::is_local_source;
use ptts::synth::{DeviceKind, Quant};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A loaded recognizer, shared by every `/speech/asr` connection.
pub trait Recognizer: Send + Sync {
    /// Start a stream. `language` pins the spoken language when the model takes one.
    fn stream(&self, language: Option<&str>) -> Result<Box<dyn RecognizerStream>>;
    /// Frames the text lags the audio: how much silence flushes the last words out.
    fn delay_frames(&self) -> usize;
    /// Languages [`Self::stream`] accepts; empty when the model takes none.
    fn languages(&self) -> Vec<String>;
    /// Text for the tokens of one word.
    fn decode(&self, tokens: &[u32]) -> Result<String>;
}

/// One stream being transcribed: [`ptts::asr::FRAME_SIZE`] samples in, a [`Step`] out.
pub trait RecognizerStream: Send {
    fn step(&mut self, pcm: &[f32]) -> Result<Step>;
    /// An independent copy of the stream: stepping it leaves `self` exactly as it was.
    fn fork(&self) -> Result<Box<dyn RecognizerStream>>;
}

struct PttsRecognizer {
    asr: Asr,
    tokenizer: ptts::tok::Tok,
}

struct PttsStream(ptts::asr::AsrStream);

impl Recognizer for PttsRecognizer {
    fn stream(&self, language: Option<&str>) -> Result<Box<dyn RecognizerStream>> {
        Ok(Box::new(PttsStream(self.asr.stream(language)?)))
    }

    fn delay_frames(&self) -> usize {
        self.asr.delay_frames()
    }

    fn languages(&self) -> Vec<String> {
        self.asr.languages()
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(self.tokenizer.decode(tokens)?)
    }
}

impl RecognizerStream for PttsStream {
    fn step(&mut self, pcm: &[f32]) -> Result<Step> {
        Ok(self.0.step(pcm)?)
    }

    fn fork(&self) -> Result<Box<dyn RecognizerStream>> {
        Ok(Box::new(PttsStream(self.0.fork()?)))
    }
}

/// What `/speech/asr` serves, or `None` when the server was started without `--asr-config`.
#[derive(Clone)]
pub struct AsrApp(pub Option<Arc<AsrInner>>);

pub struct AsrInner {
    pub recognizer: Arc<dyn Recognizer>,
    pub requests: Arc<tokio::sync::Semaphore>,
    /// The checkpoint that loaded: its repo id, or for a local folder its name.
    pub model_name: String,
}

impl AsrApp {
    pub fn new(
        recognizer: Arc<dyn Recognizer>,
        model_name: String,
        max_concurrent_requests: std::num::NonZeroUsize,
    ) -> Self {
        Self(Some(Arc::new(AsrInner {
            recognizer,
            requests: Arc::new(tokio::sync::Semaphore::new(max_concurrent_requests.get())),
            model_name,
        })))
    }
}

const FILES: [&str; 3] = ["config.json", "model.safetensors", "mimi.safetensors"];
const TOKENIZER: &str = "tokenizer.json";

fn missing_tokenizer(dir_or_repo: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "the ASR checkpoint {dir_or_repo} has no {TOKENIZER}; convert its SentencePiece \
         tokenizer.model with `uv run scripts/convert-tokenizer.py tokenizer.model` and pass \
         the result with --asr-tokenizer"
    )
}

/// Load the speech-recognition checkpoint in a local folder (or beside a local
/// `config.json`), or in a Hugging Face repo.
pub async fn load_asr(
    source: &Path,
    revision: Option<&str>,
    tokenizer: Option<&Path>,
    device: DeviceKind,
    quant: Quant,
    dtype: Option<Dtype>,
    max_concurrent_requests: std::num::NonZeroUsize,
) -> Result<AsrApp> {
    let (files, model_name): (Vec<PathBuf>, String) = if is_local_source(source) {
        anyhow::ensure!(revision.is_none(), "--asr-revision requires a Hugging Face repo ID");
        let dir = if source.is_dir() {
            source.to_path_buf()
        } else {
            source.parent().context("the ASR config has a parent directory")?.to_path_buf()
        };
        let mut files: Vec<PathBuf> = FILES.iter().map(|f| dir.join(f)).collect();
        for f in &files {
            anyhow::ensure!(f.is_file(), "the ASR checkpoint has no {}", f.display());
        }
        match tokenizer {
            Some(t) => files.push(t.to_path_buf()),
            None if dir.join(TOKENIZER).is_file() => files.push(dir.join(TOKENIZER)),
            None => return Err(missing_tokenizer(&dir.display().to_string())),
        }
        let absolute = std::fs::canonicalize(&dir)?;
        let name = absolute.file_name().unwrap_or_default().to_string_lossy().into_owned();
        (files, name)
    } else {
        let repo_id = source.to_str().context("invalid ASR repo ID")?;
        tracing::info!(repo_id, revision = revision.unwrap_or("main"), "downloading ASR model");
        let repo = crate::utils::HfRepo::model(repo_id, revision)?;
        let mut files = Vec::new();
        for f in FILES {
            files.push(repo.get(f).await?);
        }
        match tokenizer {
            Some(t) => files.push(t.to_path_buf()),
            None => files.push(repo.get(TOKENIZER).await.map_err(|e| {
                tracing::warn!(error = %e, "no {TOKENIZER} in the ASR repo");
                missing_tokenizer(repo_id)
            })?),
        }
        (files, repo_id.to_string())
    };
    let [config, model, mimi, tokenizer] = &files[..] else { unreachable!("four files") };
    let config = ptts::asr::read_config(config)?;
    let model_ext = config.model_ext();
    let mut builder = Asr::builder(config, model, mimi).device(device).quant(quant);
    if let Some(dtype) = dtype {
        builder = builder.dtype(dtype);
    }
    let asr = builder.build()?;
    let tokenizer = ptts::tok::Tok::open(tokenizer)?;
    tracing::info!(
        model = %model_name,
        checkpoint = model_ext.as_deref().unwrap_or("?"),
        device = %asr.device_name(),
        weights = %quant.as_str(),
        dtype = dtype.map_or("default", Dtype::as_str),
        vad_horizons = ?asr.vad_horizons(),
        delay_frames = asr.delay_frames(),
        "ASR model loaded"
    );
    let recognizer = Arc::new(PttsRecognizer { asr, tokenizer });
    Ok(AsrApp::new(recognizer, model_name, max_concurrent_requests))
}
