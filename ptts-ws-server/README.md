# Phonon WebSocket server

Run with a checkpoint folder or Hugging Face repo ID:

```bash
cargo run --release -p ptts-ws-server -- \
  --config "$MODEL_DIR" --quant q8 --lang en
```

The server listens on `0.0.0.0:8080`, with WebSocket connections at `/speech/tts`. `--addr` changes the address. The checkpoint supplies its config, tokenizer, weights, and voices; `--revision` selects a Hub revision. Building requires libopus (`apt-get install libopus-dev` or `brew install opus`).

By default, one WebSocket session is accepted at a time. `--max-concurrent-requests N` sets a positive limit. Excess connections receive HTTP 429 before the WebSocket upgrade, with a message asking the client to retry after an active session closes.

A session retains its primed voice and KV cache between messages. Its slot is held for the entire connection, and released after the session and generation workers stop. Close connections when finished so other clients can connect. A session that sends no message for five minutes while no audio is being generated gets an error with code 408 and is closed. A client that stops reading for 30 seconds is disconnected. The limit applies to connections, including those awaiting their first setup message.

## Speech recognition (`/speech/asr`)

With `--asr-config`, the same server also transcribes speech at `/speech/asr`, with JSON messages in the style of `/speech/tts`; the local voice agent's `PttsWsStt` client speaks it. It reports words as they are recognized and, every 80 ms, the probability that the speaker has stopped, which is what a voice agent's turn-taking reads.

```bash
cargo run --release -p ptts-ws-server -- \
  --config "$MODEL_DIR" --quant q8 --lang en \
  --asr-config "$ASR_DIR" --asr-quant q8
```

`--asr-config` takes a folder (or its `config.json`) or a Hugging Face repo ID holding `config.json`, `model.safetensors`, `mimi.safetensors` and `tokenizer.json`; `--asr-revision` selects a Hub revision. A checkpoint that ships only a SentencePiece `tokenizer.model` needs it converted once with `uv run scripts/convert-tokenizer.py tokenizer.model -o tokenizer.json`, passed with `--asr-tokenizer` if it is not beside the config. `--asr-quant q8` quantizes the transformer at load time and runs about twice as fast as the unquantized default, on the CPU only. Recognition runs on `--asr-device` (`auto`, `cpu` or `cuda`), which defaults to `--device`: a build with the `cuda` feature puts it on the GPU unless told otherwise, so keep a quantized ASR on the CPU there with `--asr-device cpu`. On CUDA the LM computes in f32 unless `--asr-dtype bf16` or `f16` says otherwise; the Mimi encoder always runs in f32. f32 matches the CPU's VAD probabilities to within 0.007 and its transcripts exactly; bf16 and f16 take about 0.7 GB less GPU memory and are about 12% faster on an RTX 5070, at the cost of VAD probabilities up to 0.04 (bf16) or 0.01 (f16) away from the CPU's. Without `--asr-config`, a connection to `/speech/asr` gets an `error` message with code 501 saying so, and is closed.

Sessions are limited separately from `/speech/tts`: `--asr-max-concurrent-requests` (default 1), with the same HTTP 429, idle and slow-reader rules as above. A voice agent holds one ASR connection for the conversation, plus one TTS connection per reply and briefly two while it is interrupted, so serve it with `--max-concurrent-requests 2` or more.

### Protocol

Every message is a JSON text frame with a `type`.

Client to server:

| `type` | Fields | |
|---|---|---|
| `setup` | `model_name` (ignored), `input_format`, `json_config` | First message. `input_format` must be `pcm`. `json_config` is a JSON object *as a string*; its `language` (`en`, `fr`, `de`, `es`, `pt`, or the checkpoint's other values) pins the spoken language, and without it the model picks. Other keys are ignored. |
| `audio` | `audio` | Base64 of s16le mono PCM at 24 kHz, in chunks of any length. |
| `peek` | `peek_id` (an unsigned 64-bit integer) | Ask for the words still in flight: answered with one `peek_result`. The stream carries on undisturbed. |
| `end_of_stream` | | No more audio. |

Server to client:

| `type` | Fields | |
|---|---|---|
| `ready` | `model_name`, `sample_rate` (24000), `frame_size` (1920), `delay_in_frames`, `text_stream_names`, `request_id` | Answer to `setup`. |
| `step` | `step_idx`, `step_duration_s` (0.08), `total_duration_s`, `vad` | One per 1920-sample frame of audio, in order. `total_duration_s` is the client audio consumed so far. `vad` is a list of `{"horizon_s", "inactivity_prob"}`, one per horizon of the checkpoint in ascending order (`0.5, 1, 2, 3` s for `asr-67f06783.100`). `inactivity_prob` is the probability that the speaker says nothing more within the horizon: near 0 while they speak, rising towards 1 a few hundred milliseconds after they stop. |
| `text` | `text`, `start_s`, `stream_id` (0) | A recognized word (sometimes two short ones), sent once the model has closed it, which is `delay_in_frames` frames (0.8 s) after it was spoken. |
| `end_text` | `stop_s`, `stream_id` (0) | When the previous word ended. |
| `error` | `code`, `message` | 400 for a bad request, 408 for an idle session, 500 for an inference failure, 501 for an unsupported input format or no recognizer. Setup errors leave the session waiting for another `setup`. |
| `peek_result` | `peek_id`, `total_duration_s`, `words` | Answer to `peek`, with its `peek_id`. `words` is a list of `{"text", "start_s"}`, the words the server has not sent as `text` yet but would if the audio ended at the peek; empty when none are pending. `total_duration_s` is the client audio the peek covers. |
| `end_of_stream` | | After `end_of_stream` from the client: the last partial frame is padded with silence and gets its `step`, then the model's text delay is run out on silence so the last words arrive (as `text`/`end_text` only, no further `step`s), then this, and the server closes the connection. |

The first two `step`s report `inactivity_prob` 1.0 for every horizon: the model's output is meaningless for its first four steps (its initial token, the two frames of silence it starts on, and the first frame of audio), and nobody has been heard speaking yet. Afterwards, the probability drops about 0.3 s after speech starts.

#### Peeking at the words in flight

A word arrives as `text` `delay_in_frames` frames (0.8 s) after it was spoken, so when the speaker stops, their last words are still inside the model. `end_of_stream` gets them out by running that delay on silence, but ends the session; sending silence instead would feed the voice activity heads a pause that never happened. `peek` gets them without either: the server copies the stream's state (the Mimi encoder's and the language model's caches, the word in progress), runs the copy on the buffered partial frame padded with silence and then on `delay_in_frames + 1` frames of silence, exactly as `end_of_stream` would, answers with the words that come out, and drops the copy. The live stream is untouched: its `step`s and `text`s are the same, bit for bit, as without the peek. So a voice agent can start its reply speculatively a moment after the speaker pauses, with everything sent as `text` so far followed by the `peek_result`'s `words` as the full transcript, and keep listening.

A `peek` is handled in order with the audio: its `peek_result` comes after the `step`s and `text`s of all the audio sent before it, and covers that audio including a partial frame not yet stepped; `total_duration_s` counts it. The copy's frames produce no `step`s. The words carry the same `text` and `start_s` as their `text` messages would; the end of the stream can still change how a word closes, typically its punctuation ("Hello," while more speech follows, "Hello." at a peek), and a word can appear in several peeks until it is sent as `text`. A peek costs the model 11 extra steps on the session's inference thread, about 0.2 s on an 8-core desktop CPU with `--asr-quant q8` (`RAYON_NUM_THREADS=8`) and about 60 ms on an RTX 5070, during which that session's next `step`s wait; a peek every half second keeps such a stream real time but delays its `step`s by up to that much. `peek` before `setup` is an error like `audio`, and peeks may be sent back to back: each is answered in turn.

Inference runs on a blocking thread per session. Audio arriving faster than the model consumes it is held in a bounded queue, after which the server stops reading the socket, so a client streaming a file faster than real time is slowed to the model's speed rather than buffered without limit.
