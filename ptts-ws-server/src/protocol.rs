#[allow(dead_code)]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ErrorMsg {
    Error { message: String },
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TtsRequest {
    Setup {
        #[serde(default)]
        json_config: String,
        #[serde(default)]
        model_name: String,
        output_format: String,
        voice: Option<String>,
        voice_id: Option<String>,
        voice_emb: Option<String>,
    },
    Text {
        text: String,
    },
    Flush {
        flush_id: u64,
    },
    EndOfStream,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TtsReply {
    Text {
        text: String,
        start_s: f64,
        stop_s: f64,
        stream_id: u32,
    },
    Ready {
        model_name: String,
        sample_rate: u32,
        frame_size: u32,
        audio_stream_names: Vec<String>,
        text_stream_names: Vec<String>,
        request_id: String,
    },
    Audio {
        audio: String,
        start_s: f64,
        stop_s: f64,
        stream_id: u32,
    },
    Error {
        message: String,
        code: u32,
    },
    Stats {
        json_stats: String,
    },
    EndOfStream,
    Flushed {
        flush_id: u64,
    },
}

pub mod error_codes {
    pub const BAD_REQUEST: u32 = 400;
    pub const NOT_FOUND: u32 = 404;
    pub const TIMEOUT: u32 = 408;
    pub const INTERNAL: u32 = 500;
    pub const NOT_IMPLEMENTED: u32 = 501;
}

/// Client messages of the speech-recognition route (`/speech/asr`).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SttRequest {
    Setup {
        #[serde(default)]
        model_name: String,
        /// Only `pcm` (s16le mono at the `sample_rate` of `ready`) is accepted.
        #[serde(default = "default_input_format")]
        input_format: String,
        /// A JSON object as a string; `language` pins the spoken language.
        #[serde(default)]
        json_config: Option<String>,
    },
    Audio {
        /// Base64 of the raw audio bytes.
        audio: String,
    },
    /// The words still in flight for the audio sent so far, without disturbing the stream:
    /// answered with [`SttReply::PeekResult`].
    Peek {
        peek_id: u64,
    },
    EndOfStream,
}

fn default_input_format() -> String {
    "pcm".to_string()
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VadPrediction {
    pub horizon_s: f64,
    pub inactivity_prob: f64,
}

/// Server messages of the speech-recognition route (`/speech/asr`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SttReply {
    Ready {
        model_name: String,
        sample_rate: u32,
        frame_size: u32,
        delay_in_frames: f64,
        text_stream_names: Vec<String>,
        request_id: String,
    },
    /// Voice activity, once per 80 ms frame.
    Step {
        step_idx: u64,
        step_duration_s: f64,
        total_duration_s: f64,
        vad: Vec<VadPrediction>,
    },
    Text {
        text: String,
        start_s: f64,
        stream_id: u32,
    },
    EndText {
        stop_s: f64,
        stream_id: u32,
    },
    Error {
        code: u32,
        message: String,
    },
    EndOfStream,
    /// Answer to [`SttRequest::Peek`]: the words the stream has not sent yet but would if the
    /// audio ended at the peek.
    PeekResult {
        peek_id: u64,
        total_duration_s: f64,
        words: Vec<PeekWord>,
    },
}

/// One word of [`SttReply::PeekResult`], as a `text` message would carry it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PeekWord {
    pub text: String,
    pub start_s: f64,
}
