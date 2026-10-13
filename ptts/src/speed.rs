//! Speaking faster or slower without changing pitch.
//!
//! [`TimeStretch`] is a post-processing stage on generated PCM. It is a port of the speed path
//! of [Sonic](https://github.com/waywardgeek/sonic) (Copyright 2010 Bill Cox, Apache-2.0), the
//! library Android's TTS and ExoPlayer use for the same job: it finds the pitch period of the
//! voice and drops or repeats whole periods, cross-fading across each splice. Cutting on period
//! boundaries is what keeps the pitch. Working on the waveform rather than on a spectrum, as a
//! phase vocoder does, is what keeps speech from sounding smeared; on Pocket TTS output it beat
//! the phase-vocoder libraries tried by a wide margin, and ran thousands of times faster than
//! real time.
//!
//! Only speed is ported. Sonic's pitch, rate and volume controls, its multi-channel input and
//! its 16-bit sample format are not: this takes mono `f32` and stays in `f32`.
//!
//! ```
//! # fn main() -> ptts::Result<()> {
//! use ptts::speed::TimeStretch;
//!
//! let mut stretch = TimeStretch::new(24_000, 1.5)?;
//! let mut out = Vec::new();
//! # let chunks: Vec<Vec<f32>> = vec![vec![0.0; 1920]; 10];
//! for pcm in chunks {
//!     out.extend(stretch.process(&pcm));
//! }
//! out.extend(stretch.flush());
//! # assert!(out.len().abs_diff(12_800) < 400);
//! # Ok(())
//! # }
//! ```

use crate::{Error, Result};

/// Slowest speed accepted. Sonic goes lower, but every method tried on Pocket TTS output was
/// noticeably worse by 2x either way, and nothing a listener wants lies past 4x.
pub const MIN_SPEED: f32 = 0.25;
/// Fastest speed accepted. See [`MIN_SPEED`].
pub const MAX_SPEED: f32 = 4.0;

/// The range of voice pitches searched, in Hz. Sonic's, which covers adult and child voices.
const MIN_PITCH: usize = 65;
const MAX_PITCH: usize = 400;
/// The coarse pitch search runs on input averaged down to roughly this rate.
const AMDF_FREQ: usize = 4000;

/// Error unless `speed` is in [`MIN_SPEED`]`..=`[`MAX_SPEED`]. NaN is out of range too.
pub fn check_speed(speed: f32) -> Result<()> {
    if !(MIN_SPEED..=MAX_SPEED).contains(&speed) {
        return Err(Error::invalid_argument(format!(
            "speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}"
        )));
    }
    Ok(())
}

/// A streaming, pitch-preserving speed change for mono PCM.
///
/// Feed it audio with [`Self::process`] in chunks of any size, and call [`Self::flush`] once
/// the input ends to get the last of it. At a speed of 1 the input comes straight back out,
/// unchanged and undelayed. Otherwise up to two of the longest pitch periods searched (about
/// 30 ms) stay buffered between calls, since a splice needs that much audio past it.
///
/// The speed can change mid-stream with [`Self::set_speed`]. Audio already buffered keeps the
/// speed it arrived at, so the change takes effect smoothly.
#[derive(Clone, Debug)]
pub struct TimeStretch {
    speed: f32,
    /// Seconds per sample.
    sample_period: f64,
    min_period: usize,
    max_period: usize,
    /// Input a splice needs ahead of it: two of the longest periods.
    max_required: usize,
    /// Decimation factor for the coarse pitch search.
    skip: usize,
    /// Input not yet turned into output.
    input: Vec<f32>,
    /// How long `input` should play for, in seconds, at the speed each sample arrived at.
    input_play_time: f64,
    /// Output so far minus what the speed asks for, in seconds. After a splice it is off by up
    /// to a period, and input is copied through unchanged until it crosses zero again.
    time_error: f64,
    prev_period: usize,
    prev_min_diff: f32,
    /// Scratch for the decimated input of the coarse search.
    down: Vec<f32>,
}

impl TimeStretch {
    /// A stretcher for mono audio at `sample_rate`, playing it at `speed` times its natural rate.
    pub fn new(sample_rate: usize, speed: f32) -> Result<Self> {
        check_speed(speed)?;
        // Below this the shortest period searched is under two samples.
        if sample_rate < 2 * MAX_PITCH {
            return Err(Error::invalid_argument(format!(
                "sample rate must be at least {} Hz, got {sample_rate}",
                2 * MAX_PITCH
            )));
        }
        let max_period = sample_rate / MIN_PITCH;
        Ok(Self {
            speed,
            sample_period: 1.0 / sample_rate as f64,
            min_period: sample_rate / MAX_PITCH,
            max_period,
            max_required: 2 * max_period,
            skip: (sample_rate / AMDF_FREQ).max(1),
            input: Vec::new(),
            input_play_time: 0.0,
            time_error: 0.0,
            prev_period: 0,
            prev_min_diff: 0.0,
            down: Vec::new(),
        })
    }

    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Change the speed for audio passed to [`Self::process`] from now on.
    pub fn set_speed(&mut self, speed: f32) -> Result<()> {
        check_speed(speed)?;
        self.speed = speed;
        Ok(())
    }

    /// Add `pcm` and return whatever output is ready. Can be empty.
    pub fn process(&mut self, pcm: &[f32]) -> Vec<f32> {
        self.input.extend_from_slice(pcm);
        self.input_play_time += pcm.len() as f64 * self.sample_period / self.speed as f64;
        let mut out = Vec::with_capacity((pcm.len() as f32 / self.speed) as usize + 64);
        self.run(&mut out);
        out
    }

    /// Return the rest of the output, once the input has ended. The stretcher is empty
    /// afterwards and can start on a new stream.
    pub fn flush(&mut self) -> Vec<f32> {
        let mut out = Vec::new();
        if !self.input.is_empty() {
            let owed = (self.input_play_time / self.sample_period).round() as usize;
            // A splice needs `max_required` samples past it, so push the tail through with
            // silence and keep only what the real input is owed. Unlike Sonic, the silence is
            // given play time: without it, the average speed of the buffer jumps and the last
            // real samples are stretched at the wrong speed.
            let pad = 2 * self.max_required;
            self.input.resize(self.input.len() + pad, 0.0);
            self.input_play_time += pad as f64 * self.sample_period / self.speed as f64;
            self.run(&mut out);
            out.truncate(owed);
        }
        self.input.clear();
        self.input_play_time = 0.0;
        self.time_error = 0.0;
        out
    }

    fn run(&mut self, out: &mut Vec<f32>) {
        if self.input.is_empty() {
            return;
        }
        // The speed the buffered input asks for on average. It differs from `self.speed` only
        // while audio from before a `set_speed` is still buffered.
        let mut speed = self.input.len() as f64 * self.sample_period / self.input_play_time;
        // Otherwise rounding in the running play time nudges it off the speed that was set,
        // and since splice lengths are truncated to whole samples, the output would depend on
        // how the input happened to be chunked.
        if (speed / self.speed as f64 - 1.0).abs() < 1e-6 {
            speed = self.speed as f64;
        }
        if (0.99999..=1.00001).contains(&speed) {
            out.append(&mut self.input);
            self.input_play_time = 0.0;
        } else {
            self.change_speed(speed, out);
        }
    }

    /// Splice through as much of the input as has `max_required` samples ahead of it.
    fn change_speed(&mut self, speed: f64, out: &mut Vec<f32>) {
        let len = self.input.len();
        if len < self.max_required {
            return;
        }
        let play_time_per_sample = self.input_play_time / len as f64;
        let mut pos = 0;
        loop {
            if (speed > 1.0 && speed < 2.0 && self.time_error < 0.0)
                || (speed < 1.0 && speed > 0.5 && self.time_error > 0.0)
            {
                // Still making up for the last splice: copy input through unchanged.
                let wanted = 1.0 - self.time_error * speed / (self.sample_period * (speed - 1.0));
                let n = (wanted as usize).min(len - pos);
                out.extend_from_slice(&self.input[pos..pos + n]);
                self.time_error += n as f64 * self.sample_period * (speed - 1.0) / speed;
                pos += n;
            } else {
                let period = self.find_pitch_period(pos);
                let s = &self.input[pos..];
                if speed > 1.0 {
                    // Drop a period: fade from this one into the next. From 2x up, part of
                    // every period goes rather than a whole one every so often.
                    let n = if speed >= 2.0 {
                        (period as f64 / (speed - 1.0)) as usize
                    } else {
                        period
                    };
                    overlap_add(out, &s[..n], &s[period..period + n]);
                    pos += period + n;
                    if speed < 2.0 {
                        self.time_error += n as f64 * self.sample_period
                            - (period + n) as f64 * play_time_per_sample;
                    }
                } else {
                    // Repeat a period: play it, then fade from the next one back into it.
                    let n = if speed <= 0.5 {
                        (period as f64 * speed / (1.0 - speed)) as usize
                    } else {
                        period
                    };
                    // The speed is at least MIN_SPEED and the period at least `min_period`, so
                    // this always consumes input.
                    debug_assert!(n > 0);
                    out.extend_from_slice(&s[..period]);
                    overlap_add(out, &s[period..period + n], &s[..n]);
                    pos += n;
                    if speed > 0.5 {
                        self.time_error += (period + n) as f64 * self.sample_period
                            - n as f64 * play_time_per_sample;
                    }
                }
            }
            if pos + self.max_required > len {
                break;
            }
        }
        self.input.drain(..pos);
        self.input_play_time *= (len - pos) as f64 / len as f64;
    }

    /// The pitch period of the input at `pos`, in samples.
    fn find_pitch_period(&mut self, pos: usize) -> usize {
        let samples = &self.input[pos..pos + self.max_required];
        let (period, min_diff, max_diff) = if self.skip == 1 {
            amdf(samples, self.min_period, self.max_period)
        } else {
            // A coarse search on decimated input, then a fine one around what it found.
            let skip = self.skip;
            self.down.clear();
            self.down
                .extend(samples.chunks_exact(skip).map(|c| c.iter().sum::<f32>() / skip as f32));
            let (coarse, _, _) = amdf(&self.down, self.min_period / skip, self.max_period / skip);
            let coarse = coarse * skip;
            let lo = coarse.saturating_sub(4 * skip).max(self.min_period);
            let hi = (coarse + 4 * skip).min(self.max_period);
            amdf(samples, lo, hi)
        };
        // At the abrupt end of a voiced sound the new estimate can be poor where the previous
        // one still fits: a weak match this time, and clearly worse than last time.
        let better_prev = self.prev_period != 0
            && min_diff != 0.0
            && max_diff <= min_diff * 3.0
            && min_diff * 2.0 > self.prev_min_diff * 3.0;
        let ret = if better_prev { self.prev_period } else { period };
        self.prev_min_diff = min_diff;
        self.prev_period = period;
        ret
    }
}

/// Average magnitude difference function: the period in `min..=max` over which `s` best repeats
/// itself, with the smallest and largest average per-sample differences seen.
fn amdf(s: &[f32], min: usize, max: usize) -> (usize, f32, f32) {
    let (mut best, mut min_diff, mut max_diff) = (min, f32::INFINITY, 0.0f32);
    for period in min..=max {
        let diff: f32 =
            s[..period].iter().zip(&s[period..2 * period]).map(|(a, b)| (a - b).abs()).sum();
        let diff = diff / period as f32;
        if diff < min_diff {
            (best, min_diff) = (period, diff);
        }
        max_diff = max_diff.max(diff);
    }
    (best, min_diff, max_diff)
}

/// Append a linear cross-fade from `down` to `up`, which have the same length.
fn overlap_add(out: &mut Vec<f32>, down: &[f32], up: &[f32]) {
    let n = down.len() as f32;
    out.extend(down.iter().zip(up).enumerate().map(|(t, (d, u))| {
        let t = t as f32;
        (d * (n - t) + u * t) / n
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: usize = 24_000;

    /// A crude voiced sound: harmonics of `f0`, swelling and fading at a syllable-ish rate, so
    /// the period search has something to find and the amplitude is not constant.
    fn voice(f0: f32, seconds: f32) -> Vec<f32> {
        let n = (seconds * SR as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                let env = 0.5 + 0.5 * (std::f32::consts::TAU * 3.0 * t).sin();
                let tone: f32 = (1..=5)
                    .map(|h| (std::f32::consts::TAU * f0 * h as f32 * t).sin() / h as f32)
                    .sum();
                0.3 * env * tone
            })
            .collect()
    }

    fn stretch_in_chunks(pcm: &[f32], speed: f32, chunk: usize) -> Vec<f32> {
        let mut s = TimeStretch::new(SR, speed).unwrap();
        let mut out = Vec::new();
        for c in pcm.chunks(chunk) {
            out.extend(s.process(c));
        }
        out.extend(s.flush());
        out
    }

    /// The fundamental of `pcm`, from the strongest autocorrelation lag between 2 ms and 15 ms.
    fn f0(pcm: &[f32]) -> f32 {
        let window = &pcm[pcm.len() / 4..pcm.len() / 4 + 4096];
        let lag = (SR / 500..SR / 65)
            .max_by(|&a, &b| {
                let r = |l: usize| -> f32 {
                    window[..window.len() - l].iter().zip(&window[l..]).map(|(x, y)| x * y).sum()
                };
                r(a).total_cmp(&r(b))
            })
            .unwrap();
        SR as f32 / lag as f32
    }

    #[test]
    fn normal_speed_passes_audio_through_unchanged() {
        let pcm = voice(120.0, 1.0);
        let mut s = TimeStretch::new(SR, 1.0).unwrap();
        let first = s.process(&pcm[..1000]);
        assert_eq!(first, pcm[..1000], "nothing is held back at 1x");
        assert_eq!(stretch_in_chunks(&pcm, 1.0, 1920), pcm);
    }

    #[test]
    fn duration_follows_the_speed() {
        let pcm = voice(120.0, 3.0);
        for speed in [0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 4.0] {
            let out = stretch_in_chunks(&pcm, speed, 1920);
            let want = pcm.len() as f32 / speed;
            let err = (out.len() as f32 - want).abs() / want;
            assert!(err < 0.01, "{speed}x: {} samples, wanted {want}", out.len());
        }
    }

    #[test]
    fn pitch_is_preserved() {
        for f in [90.0, 150.0, 260.0] {
            let pcm = voice(f, 2.0);
            for speed in [0.5, 0.75, 1.5, 2.0, 3.0] {
                let got = f0(&stretch_in_chunks(&pcm, speed, 1920));
                let cents = 1200.0 * (got / f0(&pcm)).log2();
                assert!(cents.abs() < 25.0, "{f} Hz at {speed}x came out at {got} Hz");
            }
        }
    }

    /// Generation hands over 80 ms frames, sometimes several at once, so how the input is cut
    /// up must not change the output by more than rounding.
    #[test]
    fn chunking_does_not_change_the_output() {
        let pcm = voice(140.0, 2.0);
        for speed in [0.6, 1.3, 2.5] {
            let whole = stretch_in_chunks(&pcm, speed, pcm.len());
            for chunk in [1920, 3840, 777] {
                let chunked = stretch_in_chunks(&pcm, speed, chunk);
                assert!(
                    whole.len().abs_diff(chunked.len()) <= 1,
                    "{speed}x in chunks of {chunk}: {} vs {}",
                    chunked.len(),
                    whole.len()
                );
                let worst =
                    whole.iter().zip(&chunked).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                assert!(worst < 1e-4, "{speed}x in chunks of {chunk}: differs by {worst}");
            }
        }
    }

    #[test]
    fn a_speed_change_mid_stream_applies_to_what_follows() {
        let pcm = voice(120.0, 4.0);
        let (a, b) = pcm.split_at(pcm.len() / 2);
        let mut s = TimeStretch::new(SR, 2.0).unwrap();
        let mut out = s.process(a);
        s.set_speed(0.5).unwrap();
        out.extend(s.process(b));
        out.extend(s.flush());
        let want = a.len() as f32 / 2.0 + b.len() as f32 / 0.5;
        let err = (out.len() as f32 - want).abs() / want;
        assert!(err < 0.01, "{} samples, wanted {want}", out.len());
    }

    #[test]
    fn flush_empties_the_stretcher_for_the_next_stream() {
        let mut s = TimeStretch::new(SR, 1.5).unwrap();
        assert!(s.flush().is_empty());
        let pcm = voice(120.0, 1.0);
        let mut first = s.process(&pcm);
        first.extend(s.flush());
        let mut second = s.process(&pcm);
        second.extend(s.flush());
        assert_eq!(first.len().abs_diff(second.len()), 0);
    }

    #[test]
    fn out_of_range_speeds_are_rejected() {
        for bad in [0.0, -1.0, 0.2, 4.5, f32::NAN, f32::INFINITY] {
            assert!(matches!(check_speed(bad), Err(Error::InvalidArgument(_))), "{bad}");
            assert!(TimeStretch::new(SR, bad).is_err(), "{bad}");
        }
        for ok in [MIN_SPEED, 1.0, MAX_SPEED] {
            assert!(check_speed(ok).is_ok(), "{ok}");
        }
        let mut s = TimeStretch::new(SR, 1.0).unwrap();
        assert!(s.set_speed(f32::NAN).is_err());
        assert_eq!(s.speed(), 1.0, "a rejected speed leaves the old one in place");
    }
}
