//! `/speech/asr`: streaming speech recognition over a WebSocket, in the JSON message style of
//! `/speech/tts`.
//!
//! The client sends `setup`, then base64 PCM in `audio` messages of any length, then
//! `end_of_stream`. The server answers `ready`, then for every 80 ms frame of audio a `step`
//! message with the voice-activity probabilities, and `text` / `end_text` as words close.
//! On `end_of_stream` it pads the last frame, runs the model's text delay out on silence so
//! the last words arrive, and ends with `end_of_stream`.
//!
//! Inference runs on a blocking thread per session; the async side only parses, buffers and
//! forwards. Both directions are bounded, so a client sending faster than real time is slowed
//! down by the socket rather than buffered without limit.

use crate::asr::{AsrApp, AsrInner, Recognizer, RecognizerStream};
use crate::handler::{IDLE_TIMEOUT, SEND_TIMEOUT, cancelled, forward_replies, read_requests};
use crate::protocol::{SttReply, SttRequest, VadPrediction, error_codes};
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use ptts::asr::{Event, FRAME_RATE, FRAME_SIZE, SAMPLE_RATE};
use std::sync::Arc;
use tokio::sync::{mpsc, watch};

#[cfg(test)]
#[path = "asr_handler_tests.rs"]
mod tests;

const REPLY_QUEUE: usize = 64;
/// Frames queued for the model: 1.28 s of audio.
const FRAME_QUEUE: usize = 16;
const REQUEST_QUEUE: usize = 16;
/// About eight seconds of 24 kHz s16le audio, base64-encoded, per message.
const MAX_MESSAGE_BYTES: usize = 1024 * 1024;
const NOT_CONFIGURED: &str =
    "speech recognition is not configured on this server; start it with --asr-config";

pub async fn ws_handler(
    State(app): State<AsrApp>,
    ws: WebSocketUpgrade,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let ws = ws.max_message_size(MAX_MESSAGE_BYTES);
    let Some(app) = app.0 else {
        // Answer in the protocol rather than with a bare HTTP status, whose body WebSocket
        // clients typically do not show.
        return ws.on_upgrade(not_configured);
    };
    let permit = match app.requests.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "the server is busy; retry after an active WebSocket session closes",
            )
                .into_response();
        }
    };
    async fn handle_socket(
        socket: WebSocket,
        app: Arc<AsrInner>,
        _permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        if let Err(e) = serve(socket, app).await {
            tracing::error!(error = %e, "asr session terminated");
        }
    }
    ws.on_upgrade(move |socket| handle_socket(socket, app, permit))
}

async fn not_configured(mut socket: WebSocket) {
    use futures_util::SinkExt;
    let reply =
        SttReply::Error { code: error_codes::NOT_IMPLEMENTED, message: NOT_CONFIGURED.into() };
    if let Ok(json) = serde_json::to_string(&reply) {
        let _ = tokio::time::timeout(SEND_TIMEOUT, socket.send(Message::Text(json.into()))).await;
    }
    let _ = tokio::time::timeout(SEND_TIMEOUT, socket.close()).await;
}

async fn serve(socket: WebSocket, app: Arc<AsrInner>) -> Result<()> {
    use futures_util::StreamExt;
    let (tx, rx) = socket.split();
    let (reply_tx, reply_rx) = mpsc::channel(REPLY_QUEUE);
    let (request_tx, mut request_rx) = mpsc::channel(REQUEST_QUEUE);
    let (cancel_tx, mut cancel_rx) = watch::channel(false);
    let reader = tokio::spawn(read_requests(rx, request_tx, cancel_tx.clone()));
    let forwarder = tokio::spawn(forward_replies(tx, reply_rx, cancel_rx.clone(), SEND_TIMEOUT));
    let outcome = run_session(app, &mut request_rx, &reply_tx, &mut cancel_rx).await;
    reader.abort();
    let _ = reader.await;
    drop(reply_tx);
    let forwarded = forwarder.await?;
    drop(cancel_tx);
    tracing::info!("asr session ended");
    outcome.and(forwarded)
}

/// What the async side hands the inference thread.
enum Input {
    /// One frame of [`FRAME_SIZE`] samples, holding `real` samples of the client's audio
    /// (fewer only for the zero-padded last frame).
    Frame { pcm: Vec<f32>, real: usize },
    /// Run the text delay out on silence, then send `end_of_stream`.
    Finish,
}

pub(crate) async fn run_session(
    app: Arc<AsrInner>,
    stream: &mut mpsc::Receiver<Message>,
    reply_tx: &mpsc::Sender<SttReply>,
    cancel: &mut watch::Receiver<bool>,
) -> Result<()> {
    // Set up once `setup` arrives: the frame queue into the inference thread, and its handle.
    let mut worker: Option<(mpsc::Sender<Input>, tokio::task::JoinHandle<Result<()>>)> = None;
    let mut pending: Vec<f32> = Vec::with_capacity(FRAME_SIZE);
    let mut odd_byte: Option<u8> = None;
    let outcome = 'session: loop {
        let msg = tokio::select! {
            biased;
            _ = cancelled(cancel) => break Ok(()),
            _ = reply_tx.closed() => break Ok(()),
            msg = stream.recv() => match msg {
                Some(msg) => msg,
                None => break Ok(()),
            },
            _ = tokio::time::sleep(IDLE_TIMEOUT) => {
                let message = format!("no message for {} seconds; closing the session", IDLE_TIMEOUT.as_secs());
                tracing::info!("{message}");
                send_error(reply_tx, error_codes::TIMEOUT, message).await?;
                break Ok(());
            }
        };
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break Ok(()),
            Message::Binary(_) | Message::Ping(_) | Message::Pong(_) => continue,
        };
        let req: SttRequest = match serde_json::from_str(text.as_str()) {
            Ok(r) => r,
            Err(e) => {
                send_error(reply_tx, error_codes::BAD_REQUEST, format!("invalid request: {e}"))
                    .await?;
                continue;
            }
        };
        match (&worker, req) {
            (None, SttRequest::Setup { input_format, json_config, .. }) => {
                match setup(&app, &input_format, json_config.as_deref()).await {
                    Ok(stream) => {
                        let ready = SttReply::Ready {
                            model_name: app.model_name.clone(),
                            sample_rate: SAMPLE_RATE as u32,
                            frame_size: FRAME_SIZE as u32,
                            delay_in_frames: app.recognizer.delay_frames() as f64,
                            text_stream_names: vec![],
                            request_id: uuid::Uuid::new_v4().to_string(),
                        };
                        if reply_tx.send(ready).await.is_err() {
                            break Ok(());
                        }
                        let (frame_tx, frame_rx) = mpsc::channel(FRAME_QUEUE);
                        let recognizer = app.recognizer.clone();
                        let replies = reply_tx.clone();
                        let join = tokio::task::spawn_blocking(move || {
                            infer(recognizer, stream, frame_rx, replies)
                        });
                        worker = Some((frame_tx, join));
                    }
                    Err((code, message)) => send_error(reply_tx, code, message).await?,
                }
            }
            (None, _) => {
                send_error(
                    reply_tx,
                    error_codes::BAD_REQUEST,
                    "expected setup as first message".into(),
                )
                .await?;
            }
            (Some(_), SttRequest::Setup { .. }) => {
                send_error(
                    reply_tx,
                    error_codes::BAD_REQUEST,
                    "session already initialized".into(),
                )
                .await?;
            }
            (Some((frames, _)), SttRequest::Audio { audio }) => {
                let bytes = match base64_decode(&audio) {
                    Ok(b) => b,
                    Err(e) => {
                        send_error(
                            reply_tx,
                            error_codes::BAD_REQUEST,
                            format!("invalid audio: {e}"),
                        )
                        .await?;
                        continue;
                    }
                };
                push_pcm(&mut pending, &mut odd_byte, &bytes);
                while pending.len() >= FRAME_SIZE {
                    let pcm: Vec<f32> = pending.drain(..FRAME_SIZE).collect();
                    // A full queue waits for the model: that is the backpressure.
                    if frames.send(Input::Frame { pcm, real: FRAME_SIZE }).await.is_err() {
                        // The inference thread stopped; its error is reported below.
                        break 'session Ok(());
                    }
                }
            }
            (Some((frames, _)), SttRequest::EndOfStream) => {
                if !pending.is_empty() {
                    let real = pending.len();
                    let mut pcm = std::mem::take(&mut pending);
                    pcm.resize(FRAME_SIZE, 0.0);
                    let _ = frames.send(Input::Frame { pcm, real }).await;
                }
                let _ = frames.send(Input::Finish).await;
                tracing::info!("asr stream closed by client (end of stream)");
                break Ok(());
            }
        }
    };
    if let Some((frames, join)) = worker {
        // Closing the queue stops the thread after the frames it already has; a cancelled
        // session also has its reply channel closed, which stops it sooner.
        drop(frames);
        tokio::select! {
            joined = join => match joined? {
                Ok(()) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "asr inference failed");
                    let _ = send_error(reply_tx, error_codes::INTERNAL, format!("inference failed: {e}")).await;
                }
            },
            _ = cancelled(cancel) => {}
        }
    }
    outcome
}

/// Validate `setup` and start a stream, which runs the model's leading silence.
async fn setup(
    app: &AsrInner,
    input_format: &str,
    json_config: Option<&str>,
) -> std::result::Result<Box<dyn RecognizerStream>, (u32, String)> {
    if !input_format.eq_ignore_ascii_case("pcm") {
        return Err((
            error_codes::NOT_IMPLEMENTED,
            format!(
                "input_format {input_format:?} is not supported; send pcm (s16le mono at {SAMPLE_RATE} Hz)"
            ),
        ));
    }
    let language = match json_config.map(str::trim).filter(|s| !s.is_empty()) {
        None => None,
        Some(json) => match serde_json::from_str::<serde_json::Value>(json) {
            Ok(serde_json::Value::Object(map)) => match map.get("language") {
                None | Some(serde_json::Value::Null) => None,
                Some(serde_json::Value::String(l)) => Some(l.clone()),
                Some(other) => {
                    return Err((
                        error_codes::BAD_REQUEST,
                        format!("json_config.language must be a string, got {other}"),
                    ));
                }
            },
            _ => {
                return Err((
                    error_codes::BAD_REQUEST,
                    "json_config must be a JSON object".to_string(),
                ));
            }
        },
    };
    // A model that takes no language ignores the hint rather than rejecting it.
    let languages = app.recognizer.languages();
    let language = language.filter(|_| !languages.is_empty());
    if let Some(l) = &language
        && !languages.contains(l)
    {
        return Err((
            error_codes::BAD_REQUEST,
            format!("unsupported language {l:?}; this model takes {}", languages.join(", ")),
        ));
    }
    tracing::info!(language = language.as_deref().unwrap_or("auto"), "starting asr session");
    let recognizer = app.recognizer.clone();
    tokio::task::spawn_blocking(move || recognizer.stream(language.as_deref()))
        .await
        .map_err(|e| (error_codes::INTERNAL, format!("failed to start the stream: {e}")))?
        .map_err(|e| (error_codes::INTERNAL, format!("failed to start the stream: {e}")))
}

/// The inference thread: one model step per frame, replies sent as they are produced.
fn infer(
    recognizer: Arc<dyn Recognizer>,
    mut stream: Box<dyn RecognizerStream>,
    mut frames: mpsc::Receiver<Input>,
    replies: mpsc::Sender<SttReply>,
) -> Result<()> {
    let step_duration_s = 1.0 / FRAME_RATE;
    let mut step_idx = 0u64;
    let mut samples = 0usize;
    while let Some(input) = frames.blocking_recv() {
        if replies.is_closed() {
            return Ok(());
        }
        match input {
            Input::Frame { pcm, real } => {
                let step = stream.step(&pcm)?;
                samples += real;
                let vad = step
                    .vad
                    .iter()
                    .map(|v| VadPrediction {
                        horizon_s: v.horizon_s,
                        inactivity_prob: v.inactivity_prob as f64,
                    })
                    .collect();
                let reply = SttReply::Step {
                    step_idx,
                    step_duration_s,
                    total_duration_s: samples as f64 / SAMPLE_RATE as f64,
                    vad,
                };
                step_idx += 1;
                if replies.blocking_send(reply).is_err() {
                    return Ok(());
                }
                if !send_words(recognizer.as_ref(), step.events, &replies)? {
                    return Ok(());
                }
            }
            Input::Finish => {
                // Only the words: the silence is ours, not the client's, so it gets no steps.
                let silence = vec![0f32; FRAME_SIZE];
                for _ in 0..recognizer.delay_frames() + 1 {
                    let step = stream.step(&silence)?;
                    if !send_words(recognizer.as_ref(), step.events, &replies)? {
                        return Ok(());
                    }
                }
                let _ = replies.blocking_send(SttReply::EndOfStream);
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Send a step's words; false once the client is gone.
fn send_words(
    recognizer: &dyn Recognizer,
    events: Vec<Event>,
    replies: &mpsc::Sender<SttReply>,
) -> Result<bool> {
    for event in events {
        let reply = match event {
            Event::Word { tokens, start_time } => SttReply::Text {
                text: recognizer.decode(&tokens)?,
                start_s: start_time,
                stream_id: 0,
            },
            Event::EndWord { stop_time } => SttReply::EndText { stop_s: stop_time, stream_id: 0 },
            Event::EndOfStream => continue,
        };
        if replies.blocking_send(reply).is_err() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn base64_decode(audio: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    Ok(base64::engine::general_purpose::STANDARD.decode(audio)?)
}

/// Append s16le bytes as f32 samples, carrying an odd trailing byte to the next message.
fn push_pcm(pending: &mut Vec<f32>, odd_byte: &mut Option<u8>, bytes: &[u8]) {
    let mut bytes = bytes;
    if let Some(lo) = odd_byte.take() {
        let Some((&hi, rest)) = bytes.split_first() else {
            *odd_byte = Some(lo);
            return;
        };
        pending.push(i16::from_le_bytes([lo, hi]) as f32 / 32768.0);
        bytes = rest;
    }
    let mut chunks = bytes.chunks_exact(2);
    pending.extend(chunks.by_ref().map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0));
    if let [last] = chunks.remainder() {
        *odd_byte = Some(*last);
    }
}

async fn send_error(tx: &mpsc::Sender<SttReply>, code: u32, message: String) -> Result<()> {
    tx.send(SttReply::Error { code, message })
        .await
        .map_err(|_| anyhow::anyhow!("reply channel closed"))?;
    Ok(())
}
