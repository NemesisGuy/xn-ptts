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
    ];
    let expected = [
        r#"{"type":"ready","model_name":"asr","sample_rate":24000,"frame_size":1920,"delay_in_frames":10.0,"text_stream_names":[],"request_id":"id"}"#,
        r#"{"type":"step","step_idx":3,"step_duration_s":0.08,"total_duration_s":0.32,"vad":[{"horizon_s":0.5,"inactivity_prob":0.25},{"horizon_s":1.0,"inactivity_prob":0.5}]}"#,
        r#"{"type":"text","text":"hi","start_s":1.5,"stream_id":0}"#,
        r#"{"type":"end_text","stop_s":2.0,"stream_id":0}"#,
        r#"{"type":"error","code":501,"message":"m"}"#,
        r#"{"type":"end_of_stream"}"#,
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
}

struct FakeStream {
    frames: Arc<Mutex<Vec<Vec<f32>>>>,
    seen: usize,
}

const DELAY: usize = 2;

impl Recognizer for FakeRecognizer {
    fn stream(&self, language: Option<&str>) -> Result<Box<dyn RecognizerStream>> {
        self.started.lock().unwrap().push(language.map(str::to_owned));
        Ok(Box::new(FakeStream { frames: self.frames.clone(), seen: 0 }))
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
}

#[allow(clippy::type_complexity)]
fn fake_app() -> (Arc<AsrInner>, Arc<Mutex<Vec<Vec<f32>>>>, Arc<Mutex<Vec<Option<String>>>>) {
    let frames = Arc::new(Mutex::new(vec![]));
    let started = Arc::new(Mutex::new(vec![]));
    let recognizer = Arc::new(FakeRecognizer { frames: frames.clone(), started: started.clone() });
    let app = AsrApp::new(recognizer, "fake".into(), std::num::NonZeroUsize::new(1).unwrap());
    (app.0.unwrap(), frames, started)
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
    let (request_tx, mut request_rx) = mpsc::channel(messages.len() + 1);
    for m in messages {
        request_tx.send(m).await.unwrap();
    }
    drop(request_tx);
    let (reply_tx, mut reply_rx) = mpsc::channel(1024);
    let (_cancel_tx, mut cancel_rx) = watch::channel(false);
    tokio::time::timeout(
        Duration::from_secs(10),
        run_session(app, &mut request_rx, &reply_tx, &mut cancel_rx),
    )
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
