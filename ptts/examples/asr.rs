//! Transcribe an audio file with a speech-recognition checkpoint, printing each word, the
//! voice-activity probabilities and the real-time factor.
//!
//! ```text
//! cargo run --release --example asr --features hf,audio,asr -- /path/to/asr-model audio.wav \
//!     [--quant q8] [--lang en] [--vad]
//! ```
//!
//! The folder holds `config.json`, `model.safetensors`, `mimi.safetensors` and a
//! `tokenizer.json` (convert a SentencePiece `tokenizer.model` with
//! `scripts/convert-tokenizer.py`). `--vad` prints every step's probabilities.

use anyhow::Context as _;
use ptts::Tokenizer as _;
use ptts::asr::{Asr, Event, FRAME_RATE, FRAME_SIZE, SAMPLE_RATE, read_config};
use ptts::synth::Quant;

fn main() -> anyhow::Result<()> {
    let mut positional = vec![];
    let (mut quant, mut lang, mut print_vad) = (Quant::F32, None, false);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--quant" => quant = args.next().context("--quant takes a value")?.parse()?,
            "--lang" => lang = Some(args.next().context("--lang takes a value")?),
            "--vad" => print_vad = true,
            _ => positional.push(arg),
        }
    }
    let [dir, audio] = &positional[..] else {
        anyhow::bail!("usage: asr <model directory> <audio file> [--quant q8] [--lang en] [--vad]")
    };
    let dir = std::path::Path::new(dir);
    let tok = ptts::tok::Tok::open(&dir.join("tokenizer.json"))?;

    let t0 = std::time::Instant::now();
    let asr = Asr::builder(
        read_config(dir.join("config.json"))?,
        dir.join("model.safetensors"),
        dir.join("mimi.safetensors"),
    )
    .quant(quant)
    .build()?;
    println!("loaded in {:.1}s, VAD horizons {:?}", t0.elapsed().as_secs_f64(), asr.vad_horizons());

    let mut pcm = ptts::audio::load_mono_at(std::path::Path::new(audio), SAMPLE_RATE as u32)?;
    let audio_frames = pcm.len().div_ceil(FRAME_SIZE);
    // Flush the text delay with silence so the last words come out.
    pcm.resize((audio_frames + asr.delay_frames()) * FRAME_SIZE, 0.0);

    let mut stream = asr.stream(lang.as_deref())?;
    let mut words = vec![];
    let t0 = std::time::Instant::now();
    for (i, frame) in pcm.chunks(FRAME_SIZE).enumerate() {
        let step = stream.step(frame)?;
        let t = (i + 1) as f64 / FRAME_RATE;
        if print_vad && !step.vad.is_empty() {
            let probs: Vec<String> =
                step.vad.iter().map(|v| format!("{:.2}", v.inactivity_prob)).collect();
            println!("{t:6.2}s vad {}", probs.join(" "));
        }
        for event in step.events {
            match event {
                Event::Word { tokens, start_time } => {
                    let word = tok.decode(&tokens)?;
                    println!("{t:6.2}s word {word:?} start {start_time:.2}");
                    words.push(word);
                }
                Event::EndWord { stop_time } => println!("{t:6.2}s end {stop_time:.2}"),
                Event::EndOfStream => println!("{t:6.2}s end of stream"),
            }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let duration = pcm.len() as f64 / SAMPLE_RATE as f64;
    println!("transcript: {}", words.join(" "));
    println!(
        "{duration:.2}s of audio in {elapsed:.2}s: RTF {:.3} ({:.1} ms per 80 ms frame)",
        elapsed / duration,
        elapsed * 1e3 / (pcm.len() / FRAME_SIZE) as f64
    );
    Ok(())
}
