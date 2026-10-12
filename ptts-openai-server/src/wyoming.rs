//! The Wyoming protocol, which Home Assistant uses to talk to local voice services. A connection
//! carries events, each a JSON header line followed by optional JSON data and a binary payload.
//! This serves the text-to-speech part of it: `describe` lists the voices, `synthesize` speaks a
//! whole text, and `synthesize-start`, `-chunk` and `-stop` speak text while it is still being
//! written, a sentence at a time. Home Assistant uses the last to start speaking an assistant's
//! reply before the reply is finished.
//!
//! A request names a voice but not a language, so a checkpoint that speaks several languages lists
//! each voice once per language, and the name says which one. Requests share the HTTP API's
//! generation slots, but wait for one instead of being turned away.

use crate::api::next_seed;
use crate::model::AppState;
use ptts::preprocess::{Lang, Normalize};
use ptts::synth::SpeechOptions;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{Semaphore, mpsc};

/// Sent in every header, as the Python library sends its own version. Clients do not check it.
const VERSION: &str = "1.7.0";

/// The most a header line, an event's JSON data and an event's payload may take. A request is
/// some text and a voice name, so these are far above what a real client sends.
const MAX_HEADER: usize = 64 * 1024;
const MAX_DATA: usize = 1 << 20;
const MAX_PAYLOAD: usize = 16 << 20;

/// How long a client may stop reading before its connection is closed and its slot freed.
const SEND_TIMEOUT: Duration = Duration::from_secs(30);

/// Streamed text is spoken once a sentence of at least this many characters has ended, so that
/// an abbreviation such as "Dr." is not spoken on its own.
const MIN_PIECE_CHARS: usize = 20;

/// Streamed text with no sentence end is spoken anyway once it is this long, cut at a space.
const MAX_PENDING_BYTES: usize = 1000;

/// Events queued for a connection's writer, and commands for its generation thread.
const QUEUE: usize = 32;

/// Generates speech. A trait so the protocol can be tested without a checkpoint.
pub trait Speaker: Send + Sync + 'static {
    fn speak(&self, text: &str, opts: &SpeechOptions) -> ptts::Result<Pcm>;
}

/// The audio of one request, a chunk at a time.
pub type Pcm = Box<dyn Iterator<Item = ptts::Result<Vec<f32>>>>;

impl Speaker for AppState {
    fn speak(&self, text: &str, opts: &SpeechOptions) -> ptts::Result<Pcm> {
        Ok(Box::new(self.synth.stream_with(text, opts)?))
    }
}

/// Everything a connection needs, shared by all of them.
pub struct Server<S> {
    pub speaker: S,
    pub catalog: Catalog,
    pub requests: Arc<Semaphore>,
    pub seed_base: u64,
    pub sample_rate: u32,
}

impl Server<AppState> {
    pub fn new(app: &AppState) -> anyhow::Result<Self> {
        let catalog = Catalog::new(
            &app.voices,
            app.synth.languages(),
            app.normalize,
            &app.model_name,
            &app.model_url,
        )?;
        Ok(Self {
            speaker: app.clone(),
            catalog,
            requests: app.requests.clone(),
            seed_base: app.seed_base,
            sample_rate: app.sample_rate,
        })
    }
}

/// Accept connections on `listener` until the process ends.
pub async fn serve<S: Speaker>(listener: tokio::net::TcpListener, server: Arc<Server<S>>) {
    loop {
        match listener.accept().await {
            Ok((stream, peer)) => {
                tracing::debug!(%peer, "wyoming connection");
                let _ = stream.set_nodelay(true);
                let (read, write) = stream.into_split();
                tokio::spawn(connection(tokio::io::BufReader::new(read), write, server.clone()));
            }
            Err(e) => {
                // Out of file descriptors, most likely. Wait rather than spin.
                tracing::warn!(error = %e, "cannot accept a wyoming connection");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

/// One connection: this task reads events, a second writes them, and a blocking thread
/// generates. The reader keeps reading while a sentence is spoken, so streamed text never waits
/// on the model.
async fn connection<R, W, S>(read: R, write: W, server: Arc<Server<S>>)
where
    R: AsyncBufRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send + 'static,
    S: Speaker,
{
    let (out_tx, out_rx) = mpsc::channel(QUEUE);
    let writer = tokio::spawn(write_events(write, out_rx, SEND_TIMEOUT));
    let (command_tx, command_rx) = mpsc::channel(QUEUE);
    let worker = {
        let (server, out) = (server.clone(), out_tx.clone());
        let handle = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || generate(&server, command_rx, &out, &handle))
    };
    if let Err(e) = read_events(read, &server, &command_tx, &out_tx).await {
        tracing::debug!(error = %e, "wyoming connection ended");
    }
    drop((command_tx, out_tx));
    let _ = worker.await;
    let _ = writer.await;
}

/// Text in progress between `synthesize-start` and `synthesize-stop`.
struct Streaming {
    voice: Option<String>,
    normalize: Normalize,
    pending: String,
}

enum Command {
    Speak {
        text: String,
        voice: Option<String>,
        normalize: Normalize,
        streamed: bool,
    },
    /// An event to send once the audio before it has been: an error for a later request must
    /// not arrive in the middle of an earlier one's audio.
    Send(Vec<u8>),
    /// The end of a `synthesize`: `audio-stop`.
    EndAudio,
    /// The end of a stream: `audio-stop` if any audio was sent, then `synthesize-stopped`.
    EndStream,
}

async fn read_events<R: AsyncBufRead + Unpin, S: Speaker>(
    mut read: R,
    server: &Server<S>,
    commands: &mpsc::Sender<Command>,
    out: &mpsc::Sender<Vec<u8>>,
) -> anyhow::Result<()> {
    let mut stream: Option<Streaming> = None;
    loop {
        let event = tokio::select! {
            event = read_event(&mut read) => match event? {
                Some(event) => event,
                None => return Ok(()),
            },
            // The writer gave up on this client: stop reading too.
            _ = out.closed() => return Ok(()),
        };
        let text = || event.data.get("text").and_then(Value::as_str).unwrap_or_default();
        match event.kind.as_str() {
            "describe" => out.send(server.catalog.info.clone()).await?,
            "ping" => out.send(encode_event("pong", &json!({ "text": text() }))).await?,
            // Inside a stream, Home Assistant also sends the whole text this way, for servers
            // that predate streaming. It has been spoken already.
            "synthesize" if stream.is_some() => {}
            "synthesize" => match server.catalog.resolve(event.data.get("voice")) {
                Ok((voice, normalize)) => {
                    let text = text().to_string();
                    commands
                        .send(Command::Speak { text, voice, normalize, streamed: false })
                        .await?;
                    commands.send(Command::EndAudio).await?;
                }
                Err(message) => {
                    commands.send(Command::Send(error_event(&message, "unknown-voice"))).await?
                }
            },
            "synthesize-start" => match server.catalog.resolve(event.data.get("voice")) {
                Ok((voice, normalize)) => {
                    stream = Some(Streaming { voice, normalize, pending: String::new() })
                }
                Err(message) => {
                    commands.send(Command::Send(error_event(&message, "unknown-voice"))).await?
                }
            },
            "synthesize-chunk" => {
                if let Some(s) = stream.as_mut() {
                    s.pending.push_str(text());
                    while let Some(piece) = next_piece(&mut s.pending) {
                        if !piece.is_empty() {
                            let (voice, normalize) = (s.voice.clone(), s.normalize);
                            let speak =
                                Command::Speak { text: piece, voice, normalize, streamed: true };
                            commands.send(speak).await?;
                        }
                    }
                }
            }
            "synthesize-stop" => {
                if let Some(s) = stream.take() {
                    let rest = s.pending.trim();
                    if !rest.is_empty() {
                        let (text, voice, normalize) = (rest.to_string(), s.voice, s.normalize);
                        commands
                            .send(Command::Speak { text, voice, normalize, streamed: true })
                            .await?;
                    }
                }
                commands.send(Command::EndStream).await?;
            }
            other => tracing::debug!(event = other, "ignoring a wyoming event"),
        }
    }
}

/// Run the connection's commands in order, on a blocking thread. Returns once the commands end
/// or the client is gone; dropping a speech stream stops its generation.
fn generate<S: Speaker>(
    server: &Server<S>,
    mut commands: mpsc::Receiver<Command>,
    out: &mpsc::Sender<Vec<u8>>,
    handle: &tokio::runtime::Handle,
) {
    let send = |event: Vec<u8>| out.blocking_send(event).is_ok();
    let format = json!({ "rate": server.sample_rate, "width": 2, "channels": 1 });
    let audio_start = encode_event("audio-start", &format);
    let mut started = false;
    while let Some(command) = commands.blocking_recv() {
        match command {
            Command::Speak { text, voice, normalize, streamed } => {
                // Declared before the stream, so it is released after the stream has stopped.
                let Ok(_permit) = handle.block_on(server.requests.clone().acquire_owned()) else {
                    return;
                };
                if out.is_closed() {
                    return;
                }
                let mut opts = SpeechOptions::default().seed(next_seed(server.seed_base));
                opts = opts.normalize(normalize);
                if let Some(voice) = voice {
                    opts = opts.voice(voice);
                }
                let pcm = match server.speaker.speak(&text, &opts) {
                    Ok(pcm) => pcm,
                    // A streamed piece with nothing left to say once normalized, such as an
                    // emoji on its own, is skipped rather than ending the whole reply.
                    Err(ptts::Error::InvalidArgument(e)) if streamed => {
                        tracing::warn!(error = %e, "skipping a piece of streamed text");
                        continue;
                    }
                    Err(e) => {
                        if !send(error_event(&e.to_string(), error_code(&e))) {
                            return;
                        }
                        continue;
                    }
                };
                for chunk in pcm {
                    let samples = match chunk {
                        Ok(samples) => samples,
                        Err(e) => {
                            tracing::warn!(error = %e, "wyoming request failed after it started");
                            send(error_event(&e.to_string(), error_code(&e)));
                            break;
                        }
                    };
                    if !started {
                        if !send(audio_start.clone()) {
                            return;
                        }
                        started = true;
                    }
                    if !send(audio_chunk(&format, &samples)) {
                        return;
                    }
                }
            }
            Command::Send(event) => {
                if !send(event) {
                    return;
                }
            }
            Command::EndAudio => {
                // A client waits for `audio-stop`, so it is sent even with no audio before it.
                if !started && !send(audio_start.clone()) {
                    return;
                }
                started = false;
                if !send(encode_event("audio-stop", &json!({}))) {
                    return;
                }
            }
            Command::EndStream => {
                if started && !send(encode_event("audio-stop", &json!({}))) {
                    return;
                }
                started = false;
                if !send(encode_event("synthesize-stopped", &json!({}))) {
                    return;
                }
            }
        }
    }
}

async fn write_events<W: AsyncWrite + Unpin>(
    mut write: W,
    mut events: mpsc::Receiver<Vec<u8>>,
    timeout: Duration,
) {
    while let Some(event) = events.recv().await {
        match tokio::time::timeout(timeout, write.write_all(&event)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::debug!(error = %e, "wyoming client went away");
                return;
            }
            Err(_) => {
                let secs = timeout.as_secs();
                tracing::warn!(
                    "wyoming client stopped reading for {secs}s; closing the connection"
                );
                return;
            }
        }
    }
    let _ = write.shutdown().await;
}

/// Take the next piece of streamed text worth speaking off the front of `pending`: up to the
/// end of a sentence at least [`MIN_PIECE_CHARS`] long, or to the last space once `pending` is
/// past [`MAX_PENDING_BYTES`]. A sentence ends at a line break, or at `.`, `!`, `?` or `…` (with
/// any closing quotes or brackets) followed by a space. The text after it may not have arrived
/// yet, so a final mark with nothing after it waits: "3." may be the start of "3.5".
fn next_piece(pending: &mut String) -> Option<String> {
    const ENDS: &[char] = &['.', '!', '?', '…', '。', '！', '？'];
    const CLOSERS: &[char] = &['"', '\'', '”', '’', ')', ']', '»'];
    let mut cut = None;
    let mut chars = pending.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let end = if c == '\n' {
            Some(i)
        } else if ENDS.contains(&c) {
            let mut end = i + c.len_utf8();
            while let Some(&(j, d)) = chars.peek()
                && (CLOSERS.contains(&d) || ENDS.contains(&d))
            {
                end = j + d.len_utf8();
                chars.next();
            }
            chars.peek().is_some_and(|&(_, d)| d.is_whitespace()).then_some(end)
        } else {
            None
        };
        if let Some(end) = end
            && pending[..end].trim().chars().count() >= MIN_PIECE_CHARS
        {
            cut = Some(end);
            break;
        }
    }
    let cut = cut.or_else(|| {
        (pending.len() > MAX_PENDING_BYTES).then(|| {
            // The last word may still be arriving, so the cut is at the space before it.
            pending
                .trim_end()
                .rfind(char::is_whitespace)
                .filter(|&i| i > 0)
                .unwrap_or(pending.len())
        })
    })?;
    let piece = pending[..cut].trim().to_string();
    pending.replace_range(..cut, "");
    Some(piece)
}

/// The voices as Wyoming lists them, and which voice and normalization each name stands for.
pub struct Catalog {
    listed: Vec<Listed>,
    /// The checkpoint's voices, for clients that name one directly.
    voices: Vec<String>,
    /// For a request that names no language.
    normalize: Normalize,
    /// The encoded `info` event.
    info: Vec<u8>,
}

struct Listed {
    name: String,
    voice: String,
    lang: String,
    normalize: Normalize,
}

impl Catalog {
    /// `languages` are those the checkpoint's config lists. With none, the voices speak the
    /// `--lang` language; with several, each voice is listed once per language as `VOICE-LANG`.
    pub fn new(
        voices: &[String],
        languages: &[String],
        normalize: Normalize,
        model_name: &str,
        model_url: &str,
    ) -> anyhow::Result<Self> {
        let languages = if languages.is_empty() {
            anyhow::ensure!(
                normalize != Normalize::OFF,
                "Wyoming lists each voice with its language, and this checkpoint's config lists \
                 none: set --lang to the language it speaks"
            );
            vec![normalize.as_str().to_string()]
        } else {
            languages.to_vec()
        };
        let mut listed = Vec::new();
        for lang in &languages {
            let lang_normalize = normalize_for(normalize, lang);
            if normalize != Normalize::OFF && lang_normalize == Normalize::OFF {
                tracing::warn!(
                    lang,
                    "no text normalization for this language; text is read as written"
                );
            }
            for voice in voices {
                let name =
                    if languages.len() > 1 { format!("{voice}-{lang}") } else { voice.clone() };
                listed.push(Listed {
                    name,
                    voice: voice.clone(),
                    lang: lang.clone(),
                    normalize: lang_normalize,
                });
            }
        }
        let attribution = json!({ "name": model_name, "url": model_url });
        let voice_info: Vec<Value> = listed
            .iter()
            .map(|l| {
                json!({
                    "name": l.name, "description": l.voice, "attribution": attribution,
                    "installed": true, "version": null, "languages": [l.lang], "speakers": null,
                })
            })
            .collect();
        let program = json!({
            "name": "Phonon",
            "description": "Phonon text to speech",
            "attribution": { "name": "Gradium", "url": env!("CARGO_PKG_REPOSITORY") },
            "installed": true,
            "version": env!("CARGO_PKG_VERSION"),
            "voices": voice_info,
            "supports_synthesize_streaming": true,
        });
        let info = encode_event(
            "info",
            &json!({
                "asr": [], "tts": [program], "handle": [], "intent": [], "wake": [], "mic": [],
                "snd": [],
            }),
        );
        Ok(Self { listed, voices: voices.to_vec(), normalize, info })
    }

    /// The voice (`None` for the default) and normalization a request's `voice` asks for. It
    /// names a listed voice, a checkpoint voice with or without a language, or only a language.
    fn resolve(&self, voice: Option<&Value>) -> Result<(Option<String>, Normalize), String> {
        let field = |key| voice.and_then(|v| v.get(key)).and_then(Value::as_str);
        let (name, language) = (field("name"), field("language"));
        let normalize = language.map_or(self.normalize, |lang| normalize_for(self.normalize, lang));
        match name {
            Some(name) => {
                if let Some(l) = self.listed.iter().find(|l| l.name == name) {
                    Ok((Some(l.voice.clone()), l.normalize))
                } else if let Some(v) = self.voices.iter().find(|v| v.eq_ignore_ascii_case(name)) {
                    Ok((Some(v.clone()), normalize))
                } else if name.eq_ignore_ascii_case("default") {
                    Ok((None, normalize))
                } else {
                    let names: Vec<_> = self.listed.iter().map(|l| l.name.as_str()).collect();
                    Err(format!("unknown voice '{name}'; this server has {}", names.join(", ")))
                }
            }
            None => match language.and_then(|lang| self.listed.iter().find(|l| l.lang == lang)) {
                Some(l) => Ok((None, l.normalize)),
                None => Ok((None, normalize)),
            },
        }
    }
}

/// How text in `lang` is normalized on a server whose `--lang` gave `server`: not at all if
/// that was `none`, else as `lang` with the same rewrite rules. A region such as `en-US` or
/// `en_US` counts as its language. A language with no normalizer of its own is read as written.
fn normalize_for(server: Normalize, lang: &str) -> Normalize {
    if server == Normalize::OFF {
        return Normalize::OFF;
    }
    let base = lang.split(['-', '_']).next().unwrap_or(lang);
    match base.parse::<Lang>() {
        Ok(lang) => Normalize::for_lang(lang).with_rules(server.rules()),
        Err(_) => Normalize::OFF,
    }
}

/// One event received.
#[derive(Debug, PartialEq)]
struct Event {
    kind: String,
    /// The header's `data` merged with the data that follows it. Always an object.
    data: Value,
}

/// Read the next event, or `None` at the end of the connection. Payloads are read and dropped:
/// no event this server handles carries one.
async fn read_event<R: AsyncBufRead + Unpin>(read: &mut R) -> std::io::Result<Option<Event>> {
    let invalid = |message: String| std::io::Error::new(std::io::ErrorKind::InvalidData, message);
    let mut line = Vec::new();
    while line.iter().all(u8::is_ascii_whitespace) {
        line.clear();
        let limit = MAX_HEADER as u64 + 1;
        if (&mut *read).take(limit).read_until(b'\n', &mut line).await? == 0 {
            return Ok(None);
        }
        if line.len() > MAX_HEADER {
            return Err(invalid(format!("an event header is over {MAX_HEADER} bytes")));
        }
    }
    let header: Value =
        serde_json::from_slice(&line).map_err(|e| invalid(format!("bad event header: {e}")))?;
    let kind = header
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("an event header has no type".into()))?
        .to_string();
    let mut data = match header.get("data") {
        Some(Value::Object(data)) => data.clone(),
        _ => serde_json::Map::new(),
    };
    let length = |key| header.get(key).and_then(Value::as_u64).unwrap_or(0) as usize;
    let data_length = length("data_length");
    if data_length > 0 {
        if data_length > MAX_DATA {
            return Err(invalid(format!(
                "event data is {data_length} bytes; the limit is {MAX_DATA}"
            )));
        }
        let mut bytes = vec![0; data_length];
        read.read_exact(&mut bytes).await?;
        match serde_json::from_slice(&bytes) {
            Ok(Value::Object(more)) => data.extend(more),
            _ => return Err(invalid(format!("the data of a {kind} event is not a JSON object"))),
        }
    }
    let payload_length = length("payload_length");
    if payload_length > 0 {
        if payload_length > MAX_PAYLOAD {
            return Err(invalid(format!(
                "a payload is {payload_length} bytes; the limit is {MAX_PAYLOAD}"
            )));
        }
        let skipped =
            tokio::io::copy(&mut (&mut *read).take(payload_length as u64), &mut tokio::io::sink())
                .await?;
        if skipped < payload_length as u64 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
    }
    Ok(Some(Event { kind, data: Value::Object(data) }))
}

/// An event with no payload. Data that is empty is left out, as the Python library does.
fn encode_event(kind: &str, data: &Value) -> Vec<u8> {
    encode_with_payload(kind, data, &[])
}

fn encode_with_payload(kind: &str, data: &Value, payload: &[u8]) -> Vec<u8> {
    let mut header = json!({ "type": kind, "version": VERSION });
    let data = match data {
        Value::Object(map) if map.is_empty() => Vec::new(),
        Value::Null => Vec::new(),
        data => serde_json::to_vec(data).expect("JSON values serialize"),
    };
    if !data.is_empty() {
        header["data_length"] = data.len().into();
    }
    if !payload.is_empty() {
        header["payload_length"] = payload.len().into();
    }
    let mut event = serde_json::to_vec(&header).expect("JSON values serialize");
    event.push(b'\n');
    event.extend_from_slice(&data);
    event.extend_from_slice(payload);
    event
}

/// 16-bit little-endian PCM, which is what Home Assistant writes into its WAV files.
fn audio_chunk(format: &Value, samples: &[f32]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    ptts::wav::write_samples(&mut pcm, samples).expect("writing to a Vec cannot fail");
    encode_with_payload("audio-chunk", format, &pcm)
}

fn error_event(text: &str, code: &str) -> Vec<u8> {
    encode_event("error", &json!({ "text": text, "code": code }))
}

fn error_code(e: &ptts::Error) -> &'static str {
    match e {
        ptts::Error::InvalidArgument(_) => "invalid-argument",
        ptts::Error::UnknownVoice { .. } => "unknown-voice",
        _ => "synthesis-failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    fn catalog(languages: &[&str]) -> Catalog {
        let en = "en".parse().unwrap();
        Catalog::new(&names(&["alba", "default"]), &names(languages), en, "model", "url").unwrap()
    }

    fn info_voices(catalog: &Catalog) -> Vec<Value> {
        let newline = catalog.info.iter().position(|&b| b == b'\n').unwrap();
        let header: Value = serde_json::from_slice(&catalog.info[..newline]).unwrap();
        assert_eq!(header["type"], "info");
        let data: Value = serde_json::from_slice(&catalog.info[newline + 1..]).unwrap();
        data["tts"][0]["voices"].as_array().unwrap().clone()
    }

    #[test]
    fn a_checkpoint_with_no_languages_speaks_the_server_language() {
        let catalog = catalog(&[]);
        let voices = info_voices(&catalog);
        assert_eq!(voices.len(), 2);
        assert_eq!(voices[0]["name"], "alba");
        assert_eq!(voices[0]["languages"], json!(["en"]));
        assert_eq!(voices[0]["installed"], true);
        let alba = json!({ "name": "alba" });
        assert_eq!(
            catalog.resolve(Some(&alba)).unwrap(),
            (Some("alba".into()), "en".parse().unwrap())
        );
        let off = Catalog::new(&names(&["alba"]), &[], Normalize::OFF, "model", "url");
        assert!(off.is_err(), "with --lang none there is no language to list");
    }

    #[test]
    fn several_languages_list_each_voice_once_per_language() {
        let catalog = catalog(&["en", "fr", "it"]);
        let voices = info_voices(&catalog);
        let listed: Vec<_> = voices.iter().map(|v| v["name"].as_str().unwrap()).collect();
        assert_eq!(
            listed,
            ["alba-en", "default-en", "alba-fr", "default-fr", "alba-it", "default-it"]
        );
        assert_eq!(voices[2]["description"], "alba");
        assert_eq!(voices[2]["languages"], json!(["fr"]));

        let resolve = |voice: Value| catalog.resolve(Some(&voice));
        let fr = Normalize::for_lang(Lang::Fr);
        assert_eq!(resolve(json!({ "name": "alba-fr" })).unwrap(), (Some("alba".into()), fr));
        // No normalizer for Italian: read as written.
        assert_eq!(resolve(json!({ "name": "alba-it" })).unwrap().1, Normalize::OFF);
        // A plain voice name, with or without a language.
        let en = Normalize::for_lang(Lang::En);
        assert_eq!(resolve(json!({ "name": "ALBA" })).unwrap(), (Some("alba".into()), en));
        assert_eq!(
            resolve(json!({ "name": "alba", "language": "fr_FR" })).unwrap(),
            (Some("alba".into()), fr)
        );
        assert_eq!(resolve(json!({ "language": "fr" })).unwrap(), (None, fr));
        assert_eq!(catalog.resolve(None).unwrap(), (None, en));
        assert!(resolve(json!({ "name": "alloy" })).unwrap_err().contains("alba-fr"));
    }

    #[test]
    fn the_server_rules_carry_over_to_each_language() {
        use ptts::preprocess::Rules;
        let none = Normalize::for_lang(Lang::En).with_rules(Rules::NONE);
        assert_eq!(normalize_for(none, "de").rules(), Rules::NONE);
        assert_eq!(normalize_for(Normalize::OFF, "de"), Normalize::OFF);
    }

    #[tokio::test]
    async fn events_round_trip_and_payloads_are_skipped() {
        let mut bytes = encode_with_payload("audio-chunk", &json!({ "rate": 24000 }), &[1, 2, 3]);
        bytes.extend(encode_event("synthesize", &json!({ "text": "Hi ✓" })));
        // Data in the header itself, merged with the data after it, and blank lines between.
        bytes.extend(
            b"\n{\"type\": \"ping\", \"data\": {\"a\": 1}, \"data_length\": 8}\n{\"b\": 2}",
        );
        let mut read = &bytes[..];
        let chunk = read_event(&mut read).await.unwrap().unwrap();
        assert_eq!(chunk, Event { kind: "audio-chunk".into(), data: json!({ "rate": 24000 }) });
        let synthesize = read_event(&mut read).await.unwrap().unwrap();
        assert_eq!(synthesize.data["text"], "Hi ✓");
        let ping = read_event(&mut read).await.unwrap().unwrap();
        assert_eq!(ping.data, json!({ "a": 1, "b": 2 }));
        assert_eq!(read_event(&mut read).await.unwrap(), None);

        // An empty event has no data_length, and a header too long is refused.
        assert_eq!(
            encode_event("audio-stop", &json!({})),
            b"{\"type\":\"audio-stop\",\"version\":\"1.7.0\"}\n"
        );
        let long = [b"{\"type\": \"", &vec![b'a'; MAX_HEADER][..], b"\"}\n"].concat();
        assert!(read_event(&mut &long[..]).await.is_err());
    }

    #[test]
    fn streamed_text_is_cut_at_sentence_ends() {
        let mut pending = String::new();
        let mut pieces = vec![];
        for chunk in
            ["Sure. Dr. Smith is ", "in at 3.", "5 today! Turning on the", " lights now.", " Done"]
        {
            pending.push_str(chunk);
            while let Some(piece) = next_piece(&mut pending) {
                pieces.push(piece);
            }
        }
        assert_eq!(pieces, ["Sure. Dr. Smith is in at 3.5 today!", "Turning on the lights now."]);
        assert_eq!(pending.trim(), "Done");

        // Closing quotes stay with their sentence, and a line break ends one.
        let mut pending = "He said \"turn it off now.\" Then left\nfor the evening. ".to_string();
        assert_eq!(next_piece(&mut pending).unwrap(), "He said \"turn it off now.\"");
        assert_eq!(next_piece(&mut pending).unwrap(), "Then left\nfor the evening.");
        let mut pending = "These lights are now on:\nKitchen".to_string();
        assert_eq!(next_piece(&mut pending).unwrap(), "These lights are now on:");
        assert_eq!(next_piece(&mut pending), None);

        // Text with no sentence end is spoken once it is long, without splitting a word.
        let mut pending = "word ".repeat(MAX_PENDING_BYTES / 5 + 1) + "half";
        let piece = next_piece(&mut pending).unwrap();
        assert!(piece.ends_with("word") && pending.trim() == "half", "{pending}");
    }

    /// Speaks every text as one chunk per word, and records what it was asked.
    #[derive(Default)]
    struct Words(Mutex<Vec<(String, Normalize)>>);

    impl Speaker for Arc<Words> {
        fn speak(&self, text: &str, opts: &SpeechOptions) -> ptts::Result<Pcm> {
            // As normalization leaves nothing of text without letters.
            if !text.chars().any(char::is_alphabetic) {
                return Err(ptts::Error::InvalidArgument("nothing to synthesize".into()));
            }
            self.0.lock().unwrap().push((text.to_string(), opts.normalize.unwrap()));
            let words = text.split_whitespace().count();
            Ok(Box::new((0..words).map(|_| Ok(vec![0.5; 4]))))
        }
    }

    /// Send `events` over a fresh connection and return the event types that came back, with
    /// the texts spoken.
    async fn exchange(events: &[Vec<u8>]) -> (Vec<String>, Vec<(String, Normalize)>) {
        let words = Arc::new(Words::default());
        let server = Arc::new(Server {
            speaker: words.clone(),
            catalog: catalog(&["en", "fr"]),
            requests: Arc::new(Semaphore::new(1)),
            seed_base: 0,
            sample_rate: 24000,
        });
        let (client, served) = tokio::io::duplex(1 << 16);
        let (read, write) = tokio::io::split(served);
        let task = tokio::spawn(connection(tokio::io::BufReader::new(read), write, server));
        let (mut from_server, mut to_server) = tokio::io::split(client);
        to_server.write_all(&events.concat()).await.unwrap();
        to_server.shutdown().await.unwrap();
        let mut received = vec![];
        let mut from_server = tokio::io::BufReader::new(&mut from_server);
        while let Some(event) = read_event(&mut from_server).await.unwrap() {
            if event.kind == "audio-chunk" {
                assert_eq!(event.data, json!({ "rate": 24000, "width": 2, "channels": 1 }));
            }
            received.push(event.kind);
        }
        task.await.unwrap();
        let spoken = words.0.lock().unwrap().clone();
        (received, spoken)
    }

    fn event(kind: &str, data: Value) -> Vec<u8> {
        encode_event(kind, &data)
    }

    #[tokio::test]
    async fn a_synthesize_is_answered_with_audio() {
        let (received, spoken) = exchange(&[
            event("describe", json!({})),
            event(
                "synthesize",
                json!({ "text": "Bonjour à tous.", "voice": { "name": "alba-fr" } }),
            ),
            event("synthesize", json!({ "text": "Hi.", "voice": { "name": "nobody" } })),
            event("synthesize", json!({ "text": " " })),
        ])
        .await;
        let expected = [
            "info",
            "audio-start",
            "audio-chunk",
            "audio-chunk",
            "audio-chunk",
            "audio-stop",
            "error",
            "error",
            "audio-start",
            "audio-stop",
        ];
        assert_eq!(received, expected);
        assert_eq!(spoken, [("Bonjour à tous.".to_string(), Normalize::for_lang(Lang::Fr))]);
    }

    #[tokio::test]
    async fn streamed_text_is_spoken_a_sentence_at_a_time() {
        // What Home Assistant sends: the chunks, then the whole text again, then the stop.
        let (received, spoken) = exchange(&[
            event("synthesize-start", json!({ "voice": { "name": "default-fr" } })),
            event("synthesize-chunk", json!({ "text": "Les lumières du salon " })),
            event("synthesize-chunk", json!({ "text": "sont allumées. Et" })),
            event("synthesize-chunk", json!({ "text": " voilà, c'est fait. 😀" })),
            event(
                "synthesize",
                json!({ "text": "Les lumières du salon sont allumées. Et voilà, c'est fait. 😀" }),
            ),
            event("synthesize-stop", json!({})),
        ])
        .await;
        // The emoji left at the end has nothing to say, and is skipped without an error.
        let fr = Normalize::for_lang(Lang::Fr);
        let texts: Vec<_> = spoken.iter().map(|(t, n)| (t.as_str(), *n)).collect();
        assert_eq!(
            texts,
            [("Les lumières du salon sont allumées.", fr), ("Et voilà, c'est fait.", fr)]
        );
        let mut expected = vec!["audio-start"];
        expected.extend(["audio-chunk"; 10]);
        expected.extend(["audio-stop", "synthesize-stopped"]);
        assert_eq!(received, expected);

        // A stream with nothing in it still ends.
        let (received, spoken) =
            exchange(&[event("synthesize-start", json!({})), event("synthesize-stop", json!({}))])
                .await;
        assert_eq!((received, spoken), (vec!["synthesize-stopped".to_string()], vec![]));
    }
}
