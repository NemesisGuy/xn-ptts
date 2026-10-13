//! Transcribe an audio file with a speech-recognition checkpoint, printing each word, the
//! voice-activity probabilities and the real-time factor.
//!
//! ```text
//! cargo run --release --example asr --features hf,audio,asr -- /path/to/asr-model audio.wav \
//!     [--quant q8] [--lang en] [--vad] [--peek-every 0.5]
//! ```
//!
//! The folder holds `config.json`, `model.safetensors`, `mimi.safetensors` and a
//! `tokenizer.json` (convert a SentencePiece `tokenizer.model` with
//! `scripts/convert-tokenizer.py`). `--vad` prints every step's probabilities.
//!
//! `--peek-every SECONDS` also peeks that often, as `/speech/asr`'s `peek` does: fork the
//! stream, run the text delay out on silence on the fork, print the words still in flight
//! and drop the fork. It reports what a peek costs and the real-time factor with and
//! without them.

use anyhow::Context as _;
use ptts::Tokenizer as _;
use ptts::asr::{Asr, Event, FRAME_RATE, FRAME_SIZE, SAMPLE_RATE, read_config};
use ptts::synth::Quant;

fn main() -> anyhow::Result<()> {
    let mut positional = vec![];
    let (mut quant, mut lang, mut print_vad, mut peek_every) = (Quant::F32, None, false, None);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--quant" => quant = args.next().context("--quant takes a value")?.parse()?,
            "--lang" => lang = Some(args.next().context("--lang takes a value")?),
            "--vad" => print_vad = true,
            "--peek-every" => {
                let s: f64 = args.next().context("--peek-every takes seconds")?.parse()?;
                anyhow::ensure!(s > 0.0, "--peek-every must be positive");
                peek_every = Some(((s * FRAME_RATE).round() as usize).max(1));
            }
            _ => positional.push(arg),
        }
    }
    let [dir, audio] = &positional[..] else {
        anyhow::bail!(
            "usage: asr <model directory> <audio file> [--quant q8] [--lang en] [--vad] \
             [--peek-every SECONDS]"
        )
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
    let silence = vec![0f32; FRAME_SIZE];
    let (mut fork_s, mut drain_s, mut peeks) = (vec![], vec![], 0);
    let t0 = std::time::Instant::now();
    for (i, frame) in pcm.chunks(FRAME_SIZE).enumerate() {
        if let Some(every) = peek_every
            && i > 0
            && i % every == 0
            && i <= audio_frames
        {
            let t = std::time::Instant::now();
            let mut fork = stream.fork()?;
            fork_s.push(t.elapsed().as_secs_f64());
            let t = std::time::Instant::now();
            let mut pending = vec![];
            for _ in 0..asr.delay_frames() + 1 {
                for event in fork.step(&silence)?.events {
                    if let Event::Word { tokens, .. } = event {
                        pending.push(tok.decode(&tokens)?);
                    }
                }
            }
            drop(fork);
            drain_s.push(t.elapsed().as_secs_f64());
            peeks += 1;
            println!("{:6.2}s peek {:?}", i as f64 / FRAME_RATE, pending.join(" "));
        }
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
    let peek_total: f64 = fork_s.iter().chain(&drain_s).sum();
    let live = elapsed - peek_total;
    println!(
        "{duration:.2}s of audio in {live:.2}s: RTF {:.3} ({:.1} ms per 80 ms frame)",
        live / duration,
        live * 1e3 / (pcm.len() / FRAME_SIZE) as f64
    );
    if peeks > 0 {
        let ms = |v: &mut Vec<f64>, q: f64| {
            v.sort_by(f64::total_cmp);
            v[((v.len() - 1) as f64 * q).round() as usize] * 1e3
        };
        let mut total: Vec<f64> = fork_s.iter().zip(&drain_s).map(|(a, b)| a + b).collect();
        println!(
            "{peeks} peeks: fork median {:.1} ms (max {:.1}), {} silent frames median {:.1} ms \
             (max {:.1}), peek median {:.1} ms (max {:.1}); RTF with peeks {:.3}",
            ms(&mut fork_s, 0.5),
            ms(&mut fork_s, 1.0),
            asr.delay_frames() + 1,
            ms(&mut drain_s, 0.5),
            ms(&mut drain_s, 1.0),
            ms(&mut total, 0.5),
            ms(&mut total, 1.0),
            elapsed / duration,
        );
    }
    Ok(())
}
