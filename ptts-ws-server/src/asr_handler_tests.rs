use super::*;
use crate::asr::AsrInner;
use ptts::asr::Step;
use std::sync::Mutex;
use std::time::Duration;

#[test]
fn replies_carry_every_field_and_round_trip() {
    let replies = [
        SttReply::Ready {
            model_name: "asr".into(),
            sample_rate: 24000,
            frame_size: 1920,
            delay_in_frames: 10.0,
            text_stream_names: vec![],
            request_id: "id".into(),
        },
        SttReply::Step {
            step_idx: 3,
            step_duration_s: 0.08,
            total_duration_s: 0.32,
            vad: vec![
                VadPrediction { horizon_s: 0.5, inactivity_prob: 0.25 },
                VadPrediction { horizon_s: 1.0, inactivity_prob: 0.5 },
            ],
        },
        SttReply::Text { text: "hi".into(), start_s: 1.5, stream_id: 0 },
        SttReply::EndText { stop_s: 2.0, stream_id: 0 },
        SttReply::Error { code: 501, message: "m".into() },
        SttReply::EndOfStream,
        SttReply::PeekResult {
            peek_id: 7,
            total_duration_s: 1.25,
            words: vec![
                PeekWord { text: "hello".into(), start_s: 0.5 },
                PeekWord { text: "there".into(), start_s: 0.75 },
            ],
        },
        SttReply::PeekResult { peek_id: 8, total_duration_s: 0.0, words: vec![] },
    ];
    let expected = [
        r#"{"type":"ready","model_name":"asr","sample_rate":24000,"frame_size":1920,"delay_in_frames":10.0,"text_stream_names":[],"request_id":"id"}"#,
        r#"{"type":"step","step_idx":3,"step_duration_s":0.08,"total_duration_s":0.32,"vad":[{"horizon_s":0.5,"inactivity_prob":0.25},{"horizon_s":1.0,"inactivity_prob":0.5}]}"#,
        r#"{"type":"text","text":"hi","start_s":1.5,"stream_id":0}"#,
        r#"{"type":"end_text","stop_s":2.0,"stream_id":0}"#,
        r#"{"type":"error","code":501,"message":"m"}"#,
        r#"{"type":"end_of_stream"}"#,
        r#"{"type":"peek_result","peek_id":7,"total_duration_s":1.25,"words":[{"text":"hello","start_s":0.5},{"text":"there","start_s":0.75}]}"#,
        r#"{"type":"peek_result","peek_id":8,"total_duration_s":0.0,"words":[]}"#,
    ];
    for (reply, json) in replies.iter().zip(expected) {
        assert_eq!(serde_json::to_string(reply).unwrap(), json);
        assert_eq!(&serde_json::from_str::<SttReply>(json).unwrap(), reply);
    }
}

#[test]
fn requests_parse_as_the_voice_agent_sends_them() {
    let setup = setup_json(Some(r#"{"language":"en"}"#), "pcm");
    let SttRequest::Setup { model_name, input_format, json_config } =
        serde_json::from_value(setup).unwrap()
    else {
        panic!("setup")
    };
    assert_eq!(model_name, "default");
    assert_eq!(input_format, "pcm");
    assert_eq!(json_config.as_deref(), Some(r#"{"language":"en"}"#));
    // Optional fields may be left out.
    let bare: SttRequest = serde_json::from_str(r#"{"type":"setup"}"#).unwrap();
    assert!(
        matches!(bare, SttRequest::Setup { json_config: None, ref input_format, .. } if input_format == "pcm")
    );
    let audio: SttRequest =
        serde_json::from_value(serde_json::json!({"type": "audio", "audio": "AAE="})).unwrap();
    assert!(matches!(&audio, SttRequest::Audio { audio } if audio == "AAE="));
    let eos: SttRequest = serde_json::from_str(r#"{"type":"end_of_stream"}"#).unwrap();
    assert!(matches!(eos, SttRequest::EndOfStream));
    let peek: SttRequest = serde_json::from_str(r#"{"type":"peek","peek_id":42}"#).unwrap();
    assert!(matches!(peek, SttRequest::Peek { peek_id: 42, frames_after_speech: None }));
    // u64 ids, the full range.
    let peek: SttRequest =
        serde_json::from_value(serde_json::json!({"type": "peek", "peek_id": u64::MAX})).unwrap();
    assert!(matches!(peek, SttRequest::Peek { peek_id: u64::MAX, frames_after_speech: None }));
    assert!(serde_json::from_str::<SttRequest>(r#"{"type":"peek"}"#).is_err());
    let peek: SttRequest =
        serde_json::from_str(r#"{"type":"peek","peek_id":1,"frames_after_speech":9}"#).unwrap();
    assert!(matches!(peek, SttRequest::Peek { peek_id: 1, frames_after_speech: Some(9) }));
}

#[test]
fn pcm_bytes_split_anywhere_decode_to_the_same_samples() {
    let samples: Vec<i16> = vec![0, 1, -1, i16::MAX, i16::MIN, 1234];
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let expected: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
    for split in 0..=bytes.len() {
        let (mut pending, mut odd) = (vec![], None);
        push_pcm(&mut pending, &mut odd, &bytes[..split]);
        push_pcm(&mut pending, &mut odd, &[]);
        push_pcm(&mut pending, &mut odd, &bytes[split..]);
        assert_eq!(pending, expected, "split at {split}");
        assert_eq!(odd, None);
    }
}

// ---- A fake recognizer ----

/// Reports the frame's mean absolute amplitude as the inactivity of every horizon, so tests
/// can tell frames apart; emits a word on the second frame it sees, ends it on the third,
/// and one more word once it has been fed only silence.
struct FakeRecognizer {
    frames: Arc<Mutex<Vec<Vec<f32>>>>,
    started: Arc<Mutex<Vec<Option<String>>>>,
    forked: Arc<Mutex<Vec<Vec<f32>>>>,
}

struct FakeStream {
    /// Where this stream logs its frames: the recognizer's `frames` for a live stream, its
    /// `forked` for a fork.
    frames: Arc<Mutex<Vec<Vec<f32>>>>,
    forked: Arc<Mutex<Vec<Vec<f32>>>>,
    seen: usize,
}

const DELAY: usize = 2;

impl Recognizer for FakeRecognizer {
    fn stream(&self, language: Option<&str>) -> Result<Box<dyn RecognizerStream>> {
        self.started.lock().unwrap().push(language.map(str::to_owned));
        Ok(Box::new(FakeStream {
            frames: self.frames.clone(),
            forked: self.forked.clone(),
            seen: 0,
        }))
    }

    fn delay_frames(&self) -> usize {
        DELAY
    }

    fn languages(&self) -> Vec<String> {
        vec!["en".into(), "fr".into()]
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(tokens.iter().map(|t| format!("w{t}")).collect::<Vec<_>>().join("+"))
    }
}

impl RecognizerStream for FakeStream {
    fn step(&mut self, pcm: &[f32]) -> Result<Step> {
        assert_eq!(pcm.len(), FRAME_SIZE);
        self.frames.lock().unwrap().push(pcm.to_vec());
        self.seen += 1;
        let level = pcm.iter().map(|x| x.abs()).sum::<f32>() / pcm.len() as f32;
        let events = match self.seen {
            2 => vec![Event::Word { tokens: vec![7, 8], start_time: 0.1 }],
            3 => vec![Event::EndWord { stop_time: 0.2 }],
            _ if level == 0.0 => vec![Event::Word { tokens: vec![9], start_time: 0.3 }],
            _ => vec![],
        };
        let vad = [0.5, 1.0, 2.0, 3.0]
            .iter()
            .map(|&horizon_s| ptts::asr::VadPrediction { horizon_s, inactivity_prob: level })
            .collect();
        Ok(Step { events, vad })
    }

    fn fork(&self) -> Result<Box<dyn RecognizerStream>> {
        Ok(Box::new(FakeStream {
            frames: self.forked.clone(),
            forked: self.forked.clone(),
            seen: self.seen,
        }))
    }
}

#[allow(clippy::type_complexity)]
fn fake_app() -> (Arc<AsrInner>, Arc<Mutex<Vec<Vec<f32>>>>, Arc<Mutex<Vec<Option<String>>>>) {
    let (app, frames, started, _) = fake_app_with_forks();
    (app, frames, started)
}

type FrameLog = Arc<Mutex<Vec<Vec<f32>>>>;

/// [`fake_app`], plus the log of the frames its streams' forks were fed.
#[allow(clippy::type_complexity)]
fn fake_app_with_forks() -> (Arc<AsrInner>, FrameLog, Arc<Mutex<Vec<Option<String>>>>, FrameLog) {
    let frames = Arc::new(Mutex::new(vec![]));
    let started = Arc::new(Mutex::new(vec![]));
    let forked = Arc::new(Mutex::new(vec![]));
    let recognizer = Arc::new(FakeRecognizer {
        frames: frames.clone(),
        started: started.clone(),
        forked: forked.clone(),
    });
    let app = AsrApp::new(recognizer, "fake".into(), std::num::NonZeroUsize::new(1).unwrap());
    (app.0.unwrap(), frames, started, forked)
}

fn text(msg: impl serde::Serialize) -> Message {
    Message::Text(serde_json::to_string(&msg).unwrap().into())
}

fn audio_message(samples: &[i16]) -> Message {
    use base64::Engine;
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let audio = base64::engine::general_purpose::STANDARD.encode(bytes);
    text(serde_json::json!({"type": "audio", "audio": audio}))
}

fn setup_json(json_config: Option<&str>, input_format: &str) -> serde_json::Value {
    let mut setup = serde_json::json!({
        "type": "setup",
        "model_name": "default",
        "input_format": input_format,
    });
    if let Some(json_config) = json_config {
        setup["json_config"] = json_config.into();
    }
    setup
}

fn setup_message(json_config: Option<&str>, input_format: &str) -> Message {
    text(setup_json(json_config, input_format))
}

/// Run a session over `messages` and collect every reply.
async fn run(app: Arc<AsrInner>, messages: Vec<Message>) -> Vec<SttReply> {
    run_within(app, messages, Duration::from_secs(10)).await
}

async fn run_within(app: Arc<AsrInner>, messages: Vec<Message>, limit: Duration) -> Vec<SttReply> {
    let (request_tx, mut request_rx) = mpsc::channel(messages.len() + 1);
    for m in messages {
        request_tx.send(m).await.unwrap();
    }
    drop(request_tx);
    let (reply_tx, mut reply_rx) = mpsc::channel(1024);
    let (_cancel_tx, mut cancel_rx) = watch::channel(false);
    tokio::time::timeout(limit, run_session(app, &mut request_rx, &reply_tx, &mut cancel_rx))
        .await
        .unwrap()
        .unwrap();
    drop(reply_tx);
    let mut replies = vec![];
    while let Some(r) = reply_rx.recv().await {
        replies.push(r);
    }
    replies
}

#[tokio::test]
async fn a_session_streams_steps_and_words_and_flushes_on_end_of_stream() {
    let (app, frames, started) = fake_app();
    // 2.5 frames of a constant level, sent in chunks that do not line up with frames.
    let level = 8192i16; // 0.25
    let samples = vec![level; FRAME_SIZE * 5 / 2];
    let (a, b) = samples.split_at(1000);
    let replies = run(
        app,
        vec![
            setup_message(Some(r#"{"language":"fr","other":1}"#), "pcm"),
            audio_message(a),
            audio_message(b),
            text(serde_json::json!({"type": "end_of_stream"})),
        ],
    )
    .await;
    assert_eq!(started.lock().unwrap().as_slice(), [Some("fr".to_string())]);

    let mut replies = replies.into_iter();
    assert!(
        matches!(replies.next(), Some(SttReply::Ready { frame_size: 1920, sample_rate: 24000, delay_in_frames, .. }) if delay_in_frames == DELAY as f64)
    );
    let mut steps = vec![];
    let mut words = vec![];
    let mut rest = vec![];
    for r in replies {
        match r {
            SttReply::Step { step_idx, total_duration_s, vad, step_duration_s } => {
                assert_eq!(step_duration_s, 0.08);
                assert_eq!(vad.len(), 4);
                assert_eq!(vad[2].horizon_s, 2.0);
                steps.push((step_idx, total_duration_s, vad[2].inactivity_prob));
            }
            SttReply::Text { .. } | SttReply::EndText { .. } => words.push(r),
            other => rest.push(other),
        }
    }
    // Two full frames and the zero-padded half frame; the flush silence gets no steps.
    assert_eq!(steps.len(), 3);
    assert_eq!(steps.iter().map(|s| s.0).collect::<Vec<_>>(), [0, 1, 2]);
    assert!((steps[0].1 - 0.08).abs() < 1e-9 && (steps[1].1 - 0.16).abs() < 1e-9);
    assert!((steps[2].1 - 0.2).abs() < 1e-9, "only the real half frame counts");
    assert!((steps[0].2 - 0.25).abs() < 1e-6 && (steps[2].2 - 0.125).abs() < 1e-6);
    assert_eq!(
        words,
        [
            SttReply::Text { text: "w7+w8".into(), start_s: 0.1, stream_id: 0 },
            SttReply::EndText { stop_s: 0.2, stream_id: 0 },
            // One per flush frame: DELAY + 1 frames of silence.
            SttReply::Text { text: "w9".into(), start_s: 0.3, stream_id: 0 },
            SttReply::Text { text: "w9".into(), start_s: 0.3, stream_id: 0 },
            SttReply::Text { text: "w9".into(), start_s: 0.3, stream_id: 0 },
        ]
    );
    assert_eq!(rest, [SttReply::EndOfStream]);

    let frames = frames.lock().unwrap();
    assert_eq!(frames.len(), 3 + DELAY + 1);
    assert!(frames[..2].iter().all(|f| f.iter().all(|&x| x == 0.25)));
    assert!(frames[2][..FRAME_SIZE / 2].iter().all(|&x| x == 0.25));
    assert!(frames[2][FRAME_SIZE / 2..].iter().all(|&x| x == 0.0));
}

#[tokio::test]
async fn setup_errors_are_reported_and_the_session_continues() {
    let (app, _, started) = fake_app();
    let replies = run(
        app,
        vec![
            audio_message(&[0; 10]),
            setup_message(None, "opus"),
            setup_message(Some(r#"{"language":"xx"}"#), "pcm"),
            setup_message(Some("[1]"), "pcm"),
            setup_message(None, "PCM"),
            setup_message(None, "pcm"),
            text(serde_json::json!({"type": "nonsense"})),
        ],
    )
    .await;
    let codes: Vec<_> = replies
        .iter()
        .map(|r| match r {
            SttReply::Error { code, .. } => *code,
            SttReply::Ready { .. } => 200,
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    assert_eq!(codes, [400, 501, 400, 400, 200, 400, 400]);
    let SttReply::Error { message, .. } = &replies[2] else { unreachable!() };
    assert!(message.contains("en, fr"), "{message}");
    // No language: the model picks.
    assert_eq!(started.lock().unwrap().as_slice(), [None]);
}

fn peek_message(peek_id: u64) -> Message {
    text(serde_json::json!({"type": "peek", "peek_id": peek_id}))
}

/// The replies with every `peek_result` left out, and `ready`'s random request id blanked.
fn without_peeks(replies: &[SttReply]) -> Vec<SttReply> {
    replies
        .iter()
        .filter(|r| !matches!(r, SttReply::PeekResult { .. }))
        .cloned()
        .map(|r| match r {
            SttReply::Ready {
                request_id: _,
                model_name,
                sample_rate,
                frame_size,
                delay_in_frames,
                text_stream_names,
            } => SttReply::Ready {
                request_id: String::new(),
                model_name,
                sample_rate,
                frame_size,
                delay_in_frames,
                text_stream_names,
            },
            r => r,
        })
        .collect()
}

#[tokio::test]
async fn peeks_answer_in_order_with_the_audio_and_leave_the_stream_alone() {
    let level = 8192i16; // 0.25
    let one_and_a_half = vec![level; FRAME_SIZE * 3 / 2];
    let one = vec![level; FRAME_SIZE];
    let session = |peeks: bool| {
        let mut messages = vec![setup_message(None, "pcm"), audio_message(&one_and_a_half)];
        if peeks {
            messages.push(peek_message(1));
        }
        messages.push(audio_message(&one));
        if peeks {
            messages.extend([peek_message(2), peek_message(u64::MAX)]);
        }
        messages.push(text(serde_json::json!({"type": "end_of_stream"})));
        messages
    };

    let (app, frames, _, forked) = fake_app_with_forks();
    let plain = run(app, session(false)).await;
    assert!(forked.lock().unwrap().is_empty());
    let plain_frames = std::mem::take(&mut *frames.lock().unwrap());

    let (app, frames, _, forked) = fake_app_with_forks();
    let peeked = run(app, session(true)).await;
    // The live stream saw exactly the same frames and said exactly the same things.
    assert_eq!(without_peeks(&peeked), without_peeks(&plain));
    assert_eq!(*frames.lock().unwrap(), plain_frames);

    // Each peek lands after the steps of the audio sent before it, and before any later one.
    let kinds: Vec<String> = peeked
        .iter()
        .map(|r| match r {
            SttReply::Step { step_idx, .. } => format!("step{step_idx}"),
            SttReply::PeekResult { peek_id, .. } => format!("peek{peek_id}"),
            SttReply::Ready { .. } => "ready".into(),
            SttReply::Text { .. } => "text".into(),
            SttReply::EndText { .. } => "end_text".into(),
            SttReply::EndOfStream => "eos".into(),
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    let max = format!("peek{}", u64::MAX);
    assert_eq!(
        kinds,
        [
            "ready", "step0", "peek1", "step1", "text", "peek2", &max, "step2", "end_text", "text",
            "text", "text", "eos"
        ]
    );
    let peeks: Vec<_> = peeked
        .iter()
        .filter_map(|r| match r {
            SttReply::PeekResult { peek_id, total_duration_s, words } => {
                Some((*peek_id, *total_duration_s, words.clone()))
            }
            _ => None,
        })
        .collect();
    let word = |text: &str, start_s| PeekWord { text: text.into(), start_s };
    // Peek 1 covers one frame and a half: the fork steps the padded half frame (the fake's
    // second frame, which closes "w7+w8"), then DELAY + 1 frames of silence.
    assert_eq!(peeks[0].0, 1);
    assert!((peeks[0].1 - 0.12).abs() < 1e-9, "the buffered half frame counts");
    assert_eq!(peeks[0].2, [word("w7+w8", 0.1), word("w9", 0.3), word("w9", 0.3)]);
    // "w7+w8" has gone out as `text` by then, so peeks 2 and 3 do not repeat it.
    for (id, (peek_id, total_duration_s, words)) in [2, u64::MAX].into_iter().zip(&peeks[1..]) {
        assert_eq!(*peek_id, id);
        assert!((total_duration_s - 0.2).abs() < 1e-9);
        assert_eq!(*words, [word("w9", 0.3), word("w9", 0.3), word("w9", 0.3)]);
    }

    // Every fork got a copy of the buffered half frame, padded, then the silence.
    let forked = forked.lock().unwrap();
    assert_eq!(forked.len(), 3 * (1 + DELAY + 1));
    for fork in forked.chunks(1 + DELAY + 1) {
        assert!(fork[0][..FRAME_SIZE / 2].iter().all(|&x| x == 0.25));
        assert!(fork[0][FRAME_SIZE / 2..].iter().all(|&x| x == 0.0));
        assert!(fork[1..].iter().all(|f| f.iter().all(|&x| x == 0.0)));
    }
}

/// `frames_after_speech` stops the fork that many frames past the last speech step, never
/// past the full delay, and leaves the live stream alone.
#[tokio::test]
async fn a_trimmed_peek_stops_after_the_speech() {
    // The fake reports a frame's level as its inactivity: 0.25 is speech, 0.75 is not.
    let speech = vec![8192i16; FRAME_SIZE];
    let quiet = vec![24576i16; FRAME_SIZE * 3];
    let trimmed = |after: u32| {
        text(serde_json::json!({"type": "peek", "peek_id": after, "frames_after_speech": after}))
    };
    let (app, _, _, forked) = fake_app_with_forks();
    let replies = run(
        app,
        vec![
            setup_message(None, "pcm"),
            audio_message(&speech),
            audio_message(&quiet),
            // Speech at step 0 and steps 0-3 run: 5 frames past it are steps 4 and 5,
            // 2 frames past it have run already, 100 is capped at the full delay.
            trimmed(5),
            trimmed(2),
            trimmed(100),
            peek_message(0),
            text(serde_json::json!({"type": "end_of_stream"})),
        ],
    )
    .await;
    let peeks: Vec<_> = replies
        .iter()
        .filter_map(|r| match r {
            SttReply::PeekResult { peek_id, words, .. } => Some((*peek_id, words.len())),
            _ => None,
        })
        .collect();
    // The fake emits a word on every frame of silence.
    assert_eq!(peeks, [(5, 2), (2, 0), (100, DELAY + 1), (0, DELAY + 1)]);
    assert_eq!(forked.lock().unwrap().len(), 2 + 2 * (DELAY + 1));
}

#[tokio::test]
async fn a_peek_before_setup_is_a_protocol_error() {
    let (app, frames, _, forked) = fake_app_with_forks();
    let replies = run(
        app,
        vec![
            peek_message(1),
            setup_message(None, "pcm"),
            peek_message(2),
            text(serde_json::json!({"type": "end_of_stream"})),
        ],
    )
    .await;
    let SttReply::Error { code: 400, message } = &replies[0] else {
        panic!("expected a 400, got {:?}", replies[0])
    };
    assert!(message.contains("setup"), "{message}");
    assert!(matches!(replies[1], SttReply::Ready { .. }));
    // No audio yet: the fork runs only the silence.
    let SttReply::PeekResult { peek_id: 2, total_duration_s, .. } = replies[2] else {
        panic!("expected peek 2, got {:?}", replies[2])
    };
    assert_eq!(total_duration_s, 0.0);
    assert_eq!(forked.lock().unwrap().len(), DELAY + 1);
    assert_eq!(frames.lock().unwrap().len(), DELAY + 1, "only end_of_stream's flush");
    assert!(matches!(replies.last(), Some(SttReply::EndOfStream)));
}

#[tokio::test]
async fn the_route_reports_an_unconfigured_recognizer_and_limits_sessions() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite;

    async fn serve(app: AsrApp) -> (String, tokio::task::JoinHandle<()>) {
        let router = axum::Router::new()
            .route("/speech/asr", axum::routing::any(ws_handler))
            .with_state(app);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/speech/asr", listener.local_addr().unwrap());
        (url, tokio::spawn(async move { axum::serve(listener, router).await.unwrap() }))
    }

    let (url, server) = serve(AsrApp(None)).await;
    let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let _ = ws
        .send(tungstenite::Message::Text(
            r#"{"type":"setup","model_name":"default","input_format":"pcm"}"#.into(),
        ))
        .await;
    let reply = ws.next().await.unwrap().unwrap().into_text().unwrap();
    let SttReply::Error { code, message } = serde_json::from_str(&reply).unwrap() else {
        panic!("expected an error, got {reply}")
    };
    assert_eq!(code, 501);
    assert!(message.contains("--asr-config"), "{message}");
    server.abort();

    let (app, _, _) = fake_app();
    let (url, server) = serve(AsrApp(Some(app.clone()))).await;
    let (mut first, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let error = tokio_tungstenite::connect_async(&url).await.unwrap_err();
    let tungstenite::Error::Http(response) = error else {
        panic!("expected HTTP 429, got {error}")
    };
    assert_eq!(response.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
    first
        .send(tungstenite::Message::Text(
            r#"{"type":"setup","model_name":"default","input_format":"pcm"}"#.into(),
        ))
        .await
        .unwrap();
    let ready = first.next().await.unwrap().unwrap().into_text().unwrap();
    assert!(matches!(serde_json::from_str(&ready).unwrap(), SttReply::Ready { .. }));
    first.send(tungstenite::Message::Text(r#"{"type":"end_of_stream"}"#.into())).await.unwrap();
    let mut last = None;
    while let Some(Ok(msg)) = first.next().await {
        if let tungstenite::Message::Text(t) = msg {
            last = Some(t.to_string());
        }
    }
    assert!(matches!(serde_json::from_str(&last.unwrap()).unwrap(), SttReply::EndOfStream));
    let permit = tokio::time::timeout(Duration::from_secs(5), app.requests.clone().acquire_owned())
        .await
        .expect("a closed session leaked its slot")
        .unwrap();
    drop(permit);
    server.abort();
}

/// The local voice agent's reply shapes (`PttsWsStt` in local-voice-agent
/// `crates/agent-core/src/ptts_ws.rs`), copied verbatim, so these tests decode the server's
/// messages exactly as that client does.
mod agent_stt {
    #[derive(serde::Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum SttReply {
        Ready {
            #[serde(default)]
            sample_rate: Option<u32>,
        },
        Step {
            total_duration_s: f64,
            #[serde(default)]
            vad: Vec<VadPrediction>,
        },
        Text {
            text: String,
            start_s: f64,
        },
        Error {
            #[serde(default)]
            code: Option<serde_json::Value>,
            message: String,
        },
        EndOfStream,
        #[serde(other)]
        Other,
    }

    #[derive(serde::Deserialize)]
    pub struct VadPrediction {
        #[serde(default)]
        pub horizon_s: f64,
        pub inactivity_prob: f64,
    }
}

#[tokio::test]
async fn a_session_decodes_with_the_voice_agents_reply_shapes() {
    use agent_stt::SttReply as A;
    let (app, _, _) = fake_app();
    // The agent's setup and audio, exactly as it builds them with `serde_json::json!`.
    let setup = serde_json::json!({
        "type": "setup",
        "model_name": "default",
        "input_format": "pcm",
        "json_config": serde_json::json!({"language": "en", "extra": true}).to_string(),
    });
    let replies = run(
        app,
        vec![
            text(setup),
            audio_message(&vec![8192i16; FRAME_SIZE * 3]),
            text(serde_json::json!({"type": "end_of_stream"})),
        ],
    )
    .await;
    let decoded: Vec<A> = replies
        .iter()
        .map(|r| serde_json::from_str(&serde_json::to_string(r).unwrap()).unwrap())
        .collect();
    assert!(matches!(decoded[0], A::Ready { sample_rate: Some(24000) }));
    let steps: Vec<_> = decoded
        .iter()
        .filter_map(|r| match r {
            A::Step { total_duration_s, vad } => Some((*total_duration_s, vad)),
            _ => None,
        })
        .collect();
    assert_eq!(steps.len(), 3);
    for (i, (current_s, vad)) in steps.iter().enumerate() {
        assert!((current_s - 0.08 * (i + 1) as f64).abs() < 1e-9);
        let horizons: Vec<f64> = vad.iter().map(|v| v.horizon_s).collect();
        assert_eq!(horizons, [0.5, 1.0, 2.0, 3.0], "all four horizons, ascending");
        assert!((vad[2].inactivity_prob - 0.25).abs() < 1e-6);
    }
    let words: Vec<_> = decoded
        .iter()
        .filter_map(|r| match r {
            A::Text { text, start_s } => Some((text.as_str(), *start_s)),
            _ => None,
        })
        .collect();
    assert_eq!(words[0], ("w7+w8", 0.1));
    // `end_text` is not in the agent's enum and lands in `Other`, which it ignores.
    assert!(decoded.iter().any(|r| matches!(r, A::Other)));
    assert!(matches!(decoded.last(), Some(A::EndOfStream)));
    let error = serde_json::to_string(&SttReply::Error { code: 400, message: "m".into() }).unwrap();
    let A::Error { code, message } = serde_json::from_str(&error).unwrap() else { panic!() };
    assert_eq!((code, message.as_str()), (Some(serde_json::json!(400)), "m"));
}

// ---- The real model ----

/// The samples of a 16-bit PCM WAV file's first channel.
fn read_wav_s16(path: &std::path::Path) -> Vec<i16> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..4], b"RIFF", "{} is not a WAV file", path.display());
    let (mut pos, mut channels) = (12, 1);
    while pos + 8 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        match &bytes[pos..pos + 4] {
            b"fmt " => {
                assert_eq!(u16::from_le_bytes([body[0], body[1]]), 1, "not PCM");
                channels = u16::from_le_bytes([body[2], body[3]]) as usize;
                let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                assert_eq!(rate, SAMPLE_RATE as u32, "resample to 24 kHz first");
                assert_eq!(u16::from_le_bytes([body[14], body[15]]), 16, "not 16-bit");
            }
            b"data" => {
                return body
                    .chunks_exact(2 * channels)
                    .map(|s| i16::from_le_bytes([s[0], s[1]]))
                    .collect();
            }
            _ => {}
        }
        pos += 8 + len + (len & 1);
    }
    panic!("{} has no data chunk", path.display())
}

/// Same audio with and without a peek every half second: the live stream's replies must be
/// identical, and the last peek must predict exactly what `end_of_stream` then flushes.
///
/// `cargo test --release -p ptts-ws-server -- --ignored peeks_on_the_real_model`. Set
/// `PTTS_TEST_ASR_DEVICE=cuda` (with the `cuda` feature) to run it on the GPU, and optionally
/// `PTTS_TEST_ASR_DTYPE`; on the CPU the weights are `q8_0`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires PTTS_TEST_ASR_MODEL (a speech-recognition checkpoint folder with a \
            tokenizer.json) and PTTS_TEST_ASR_AUDIO (24 kHz mono 16-bit WAV of speech)"]
async fn peeks_on_the_real_model_leave_the_live_stream_bit_for_bit_alone() {
    use ptts::synth::{DeviceKind, Quant};
    let dir = std::env::var("PTTS_TEST_ASR_MODEL").expect("set PTTS_TEST_ASR_MODEL");
    let audio = std::env::var("PTTS_TEST_ASR_AUDIO").expect("set PTTS_TEST_ASR_AUDIO");
    let device: DeviceKind = match std::env::var("PTTS_TEST_ASR_DEVICE") {
        Ok(name) => name.parse().unwrap(),
        Err(_) => DeviceKind::Cpu,
    };
    let quant = if device.resolve() == DeviceKind::Cpu { Quant::Q80 } else { Quant::F32 };
    let dtype = std::env::var("PTTS_TEST_ASR_DTYPE").ok().map(|d| d.parse().unwrap());
    let app = crate::asr::load_asr(
        std::path::Path::new(&dir),
        None,
        None,
        device,
        quant,
        dtype,
        std::num::NonZeroUsize::new(1).unwrap(),
    )
    .await
    .unwrap()
    .0
    .unwrap();
    let samples = read_wav_s16(std::path::Path::new(&audio));
    // Half a second per message: 6.25 frames, so the peeks see every partial-frame length.
    let chunks: Vec<&[i16]> = samples.chunks(SAMPLE_RATE / 2).collect();
    let session = |peeks: bool| {
        let mut messages = vec![setup_message(Some(r#"{"language":"en"}"#), "pcm")];
        if peeks {
            messages.push(peek_message(0));
        }
        for (i, chunk) in chunks.iter().enumerate() {
            messages.push(audio_message(chunk));
            if peeks {
                messages.push(peek_message(i as u64 + 1));
            }
        }
        messages.push(text(serde_json::json!({"type": "end_of_stream"})));
        messages
    };
    let limit = Duration::from_secs(300);
    let plain = run_within(app.clone(), session(false), limit).await;
    let peeked = run_within(app, session(true), limit).await;

    assert_eq!(without_peeks(&peeked), without_peeks(&plain));
    let words: Vec<_> = plain
        .iter()
        .filter_map(|r| match r {
            SttReply::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(words.len() > 3, "no speech recognized: {words:?}");

    let peeks: Vec<_> = peeked
        .iter()
        .enumerate()
        .filter_map(|(i, r)| match r {
            SttReply::PeekResult { peek_id, total_duration_s, words } => {
                Some((i, *peek_id, *total_duration_s, words))
            }
            _ => None,
        })
        .collect();
    assert_eq!(peeks.len(), chunks.len() + 1);
    let mut sent = 0;
    for (n, (_, peek_id, total_duration_s, _)) in peeks.iter().enumerate() {
        assert_eq!(*peek_id, n as u64);
        assert!((total_duration_s - sent as f64 / SAMPLE_RATE as f64).abs() < 1e-9);
        sent += chunks.get(n).map_or(0, |c| c.len());
    }
    // Before any audio, nothing is in flight.
    assert!(peeks[0].3.is_empty(), "{:?}", peeks[0].3);
    // The last peek covers all the audio, so it runs exactly what end_of_stream then runs on
    // the live stream: the padded partial frame and the delay. Same words, same times.
    let (at, _, _, last) = peeks.last().unwrap();
    let flushed: Vec<PeekWord> = peeked[at + 1..]
        .iter()
        .filter_map(|r| match r {
            SttReply::Text { text, start_s, .. } => {
                Some(PeekWord { text: text.clone(), start_s: *start_s })
            }
            _ => None,
        })
        .collect();
    assert!(!flushed.is_empty(), "the audio should end mid-delay");
    assert_eq!(**last, flushed);
    for (_, peek_id, total_duration_s, words) in &peeks {
        let words: Vec<_> = words.iter().map(|w| format!("{}@{:.2}", w.text, w.start_s)).collect();
        eprintln!("peek {peek_id} at {total_duration_s:.3}s: {}", words.join(" "));
    }
}
