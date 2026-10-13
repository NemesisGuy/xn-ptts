mod api;
mod encoder;
mod model;
mod mp3;
mod wyoming;

use anyhow::{Context, Result};
use axum::Router;
use axum::routing::{get, post};
use clap::Parser;
use ptts::preprocess::{Normalize, Rules};
use ptts::synth::{DeviceKind, Quant};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "ptts-openai-server")]
#[command(about = "OpenAI-compatible speech server for Phonon")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8880", env = "PTTS_ADDR")]
    addr: String,

    /// Also serve the Wyoming protocol, which Home Assistant uses for local voice services, on
    /// this address, such as `0.0.0.0:10200`. Off when not set.
    #[arg(long, env = "PTTS_WYOMING_ADDR")]
    wyoming_addr: Option<String>,

    /// Maximum simultaneous speech generations. Excess requests receive HTTP 429.
    #[arg(long, default_value = "1", env = "PTTS_MAX_CONCURRENT_REQUESTS", value_parser = parse_request_limit)]
    max_concurrent_requests: std::num::NonZeroUsize,

    /// Required. Model to load: a local checkpoint folder or a `config.json` in one, or a Hugging
    /// Face repo as `hf://owner/name` or `hf://owner/name@revision`. Use a commit as the revision
    /// to reproduce a release.
    #[arg(long, env = "PTTS_MODEL", required_unless_present = "config")]
    model: Option<String>,

    /// Deprecated: a local model or a bare Hugging Face repo ID. Use --model.
    #[arg(long, env = "PTTS_CONFIG", hide = true)]
    config: Option<String>,

    /// Deprecated: the revision of --config. Use --model hf://owner/name@revision.
    #[arg(long, env = "PTTS_REVISION", hide = true)]
    revision: Option<String>,

    /// Optional directory of additional voice safetensors to load. Each
    /// `*.safetensors` file is loaded as a voice keyed by its file stem; load
    /// errors are logged and skipped rather than fatal.
    #[arg(long, env = "PTTS_VOICE_DIR")]
    voice_dir: Option<std::path::PathBuf>,

    /// Optional JSONL voice catalog to load, one `{"name", "filename"}` object per line with
    /// `filename` relative to the catalog. Voices that fail to load are logged and skipped.
    #[arg(long, env = "PTTS_VOICE_CATALOG")]
    voice_catalog: Option<std::path::PathBuf>,

    #[arg(long, default_value_t = 0.3, env = "PTTS_TEMPERATURE")]
    temperature: f32,

    #[arg(long, default_value_t = 4242424242424242, env = "PTTS_SEED")]
    seed: u64,

    /// Device to run on: auto, cpu, cuda, vulkan or metal. `auto` picks the GPU backend this
    /// build was compiled with, if any, and the CPU otherwise.
    #[arg(long, default_value = "auto", env = "PTTS_DEVICE")]
    device: String,

    /// Quantization for the flow_lm transformer linear weights.
    /// One of: q8|q8_0, q8_1, q8k, q6k, q5|q5_0, q5_1, q5k, q4|q4_0, q4_1, q4k.
    /// CPU only.
    #[arg(long, env = "PTTS_QUANT")]
    quant: Option<String>,

    /// Language incoming text is normalized as before tokenizing: `en`, `fr`, `de`, `es` or
    /// `pt`. Required: the spoken forms differ per language, so there is nothing safe to guess.
    /// `none` serves the text as written, which the model reads less well.
    #[arg(long, env = "PTTS_LANG")]
    lang: String,

    /// Which word rewrites run on the normalized text: `default` (numbers, currency,
    /// dashed-digits, emails, urls, abbreviations, elongations), `all` (those and phones, times,
    /// dates), `none`, or a comma-separated list of rule names. Has no effect with `--lang none`.
    #[arg(long, default_value = "default", env = "PTTS_REWRITES")]
    rewrites: String,

    /// Set one of the checkpoint's conditioners, e.g. `padding_bonus=0.5`; repeatable. Those
    /// not set take their defaults. In the environment, a comma-separated list.
    #[arg(
        long = "condition",
        value_name = "NAME=VALUE",
        env = "PTTS_CONDITION",
        value_delimiter = ','
    )]
    conditions: Vec<String>,
}

impl Args {
    /// The deprecated flags are checked here rather than by clap, whose errors would name the
    /// hidden flags, not the variable left over from an older setup.
    fn model_source(&self) -> Result<String> {
        match (&self.model, &self.config) {
            (Some(_), Some(_)) => {
                anyhow::bail!("PTTS_MODEL replaces PTTS_CONFIG: unset PTTS_CONFIG (or --config)")
            }
            (Some(_), None) if self.revision.is_some() => anyhow::bail!(
                "PTTS_MODEL takes the revision itself, as hf://owner/name@revision: unset \
                 PTTS_REVISION (or --revision)"
            ),
            (Some(model), None) => Ok(model.clone()),
            (None, Some(config)) => {
                tracing::warn!("--config and --revision are deprecated: use --model (PTTS_MODEL)");
                Ok(ptts::checkpoint::legacy_source(config, self.revision.as_deref())?)
            }
            (None, None) => anyhow::bail!("--model is required"),
        }
    }
}

fn parse_request_limit(value: &str) -> Result<std::num::NonZeroUsize, String> {
    let limit: std::num::NonZeroUsize = value.parse().map_err(|e| format!("{e}"))?;
    if limit.get() > tokio::sync::Semaphore::MAX_PERMITS {
        return Err(format!("limit must be at most {}", tokio::sync::Semaphore::MAX_PERMITS));
    }
    Ok(limit)
}

fn init_tracing() {
    // `info` for everything but the Hub download stack: `hf_hub` transfers through the Xet
    // backend, which reports every retry policy and range probe at `info`. Keep in sync with
    // `LOG_DIRECTIVES` in `ptts/src/bin/ptts/model_helpers.rs`.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "info,xet=warn,xet_client=warn,xet_data=warn,xet_runtime=warn,xet_core_structures=warn",
        )
    });
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::Layer::new().with_target(false))
        .with(filter)
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();

    // The weight loading after the download is CPU-bound and blocks the runtime thread, which is
    // fine here: nothing is served until it is done.
    let app_state = build_app_state(&args).await?;

    if let Some(addr) = &args.wyoming_addr {
        let server = std::sync::Arc::new(wyoming::Server::new(&app_state)?);
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("cannot listen for Wyoming on {addr}"))?;
        tracing::info!(%addr, "listening for Wyoming");
        tokio::spawn(wyoming::serve(listener, server));
    }

    let app = Router::new()
        .route("/v1/audio/speech", post(api::speech))
        .route("/v1/audio/voices", get(api::voices))
        .route("/v1/models", get(api::models))
        .route("/health", get(api::health))
        .with_state(app_state)
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    tracing::info!(addr = %args.addr, "listening on /v1/audio/speech");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutdown requested");
        })
        .await?;
    Ok(())
}

async fn build_app_state(args: &Args) -> Result<model::AppState> {
    let device = args.device.parse::<DeviceKind>()?;
    let quant = match args.quant.as_deref() {
        None => Quant::F32,
        Some(name) => name.parse::<Quant>()?,
    };
    // Device and weight-format checks run before downloading: `SynthBuilder`
    // would catch them, but only after the checkpoint is on disk.
    quant.check_device(device)?;
    let normalize = args.lang.parse::<Normalize>()?.with_rules(args.rewrites.parse::<Rules>()?);
    let mut conditions = Vec::new();
    // Empty entries are skipped: compose files often pass `PTTS_CONDITION=` for none, which
    // clap reads as one empty value, and a trailing comma leaves one too.
    for condition in args.conditions.iter().filter(|c| !c.is_empty()) {
        let (name, value) = condition.split_once('=').context("--condition takes NAME=VALUE")?;
        conditions.push((name.to_string(), value.to_string()));
    }
    model::load_ptts(
        &args.model_source()?,
        args.voice_dir.as_ref(),
        args.voice_catalog.as_deref(),
        device,
        quant,
        args.temperature,
        args.seed,
        normalize,
        &conditions,
        args.max_concurrent_requests,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_deprecated_config_flag_still_names_a_model() {
        let args = Args::try_parse_from([
            "server",
            "--config",
            "owner/model",
            "--revision",
            "abc",
            "--lang",
            "none",
        ])
        .unwrap();
        assert_eq!(args.model_source().unwrap(), "hf://owner/model@abc");
        let both = ["server", "--model", "a", "--config", "b", "--lang", "none"];
        let err = Args::try_parse_from(both).unwrap().model_source().unwrap_err();
        assert!(err.to_string().contains("unset PTTS_CONFIG"), "{err}");
        let stale = ["server", "--model", "a", "--revision", "main", "--lang", "none"];
        let err = Args::try_parse_from(stale).unwrap().model_source().unwrap_err();
        assert!(err.to_string().contains("hf://owner/name@revision"), "{err}");
    }

    #[test]
    fn request_limit_is_positive_and_defaults_to_one() {
        assert!(parse_request_limit(&usize::MAX.to_string()).is_err());
        let args = Args::try_parse_from(["server", "--model", "model", "--lang", "none"]).unwrap();
        assert_eq!(args.max_concurrent_requests.get(), 1);
        let args = Args::try_parse_from([
            "server",
            "--model",
            "model",
            "--lang",
            "none",
            "--max-concurrent-requests",
            "3",
        ])
        .unwrap();
        assert_eq!(args.max_concurrent_requests.get(), 3);
        assert!(
            Args::try_parse_from([
                "server",
                "--model",
                "model",
                "--lang",
                "none",
                "--max-concurrent-requests",
                "0",
            ])
            .is_err()
        );
    }
}
