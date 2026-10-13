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

`--asr-config` takes a folder (or its `config.json`) or a Hugging Face repo ID holding `config.json`, `model.safetensors`, `mimi.safetensors` and `tokenizer.json`; `--asr-revision` selects a Hub revision. A checkpoint that ships only a SentencePiece `tokenizer.model` needs it converted once with `uv run scripts/convert-tokenizer.py tokenizer.model -o tokenizer.json`, passed with `--asr-tokenizer` if it is not beside the config. `--asr-quant q8` quantizes the transformer at load time and runs about twice as fast as the unquantized default. Recognition runs on the CPU whatever `--device` is. Without `--asr-config`, a connection to `/speech/asr` gets an `error` message with code 501 saying so, and is closed.

Sessions are limited separately from `/speech/tts`: `--asr-max-concurrent-requests` (default 1), with the same HTTP 429, idle and slow-reader rules as above. A voice agent holds one ASR connection for the conversation, plus one TTS connection per reply and briefly two while it is interrupted, so serve it with `--max-concurrent-requests 2` or more.

### Protocol

Every message is a JSON text frame with a `type`.

Client to server:

| `type` | Fields | |
|---|---|---|
| `setup` | `model_name` (ignored), `input_format`, `json_config` | First message. `input_format` must be `pcm`. `json_config` is a JSON object *as a string*; its `language` (`en`, `fr`, `de`, `es`, `pt`, or the checkpoint's other values) pins the spoken language, and without it the model picks. Other keys are ignored. |
| `audio` | `audio` | Base64 of s16le mono PCM at 24 kHz, in chunks of any length. |
| `end_of_stream` | | No more audio. |

Server to client:

| `type` | Fields | |
|---|---|---|
| `ready` | `model_name`, `sample_rate` (24000), `frame_size` (1920), `delay_in_frames`, `text_stream_names`, `request_id` | Answer to `setup`. |
| `step` | `step_idx`, `step_duration_s` (0.08), `total_duration_s`, `vad` | One per 1920-sample frame of audio, in order. `total_duration_s` is the client audio consumed so far. `vad` is a list of `{"horizon_s", "inactivity_prob"}`, one per horizon of the checkpoint in ascending order (`0.5, 1, 2, 3` s for `asr-67f06783.100`). `inactivity_prob` is the probability that the speaker says nothing more within the horizon: near 0 while they speak, rising towards 1 a few hundred milliseconds after they stop. |
| `text` | `text`, `start_s`, `stream_id` (0) | A recognized word (sometimes two short ones), sent once the model has closed it, which is `delay_in_frames` frames (0.8 s) after it was spoken. |
| `end_text` | `stop_s`, `stream_id` (0) | When the previous word ended. |
| `error` | `code`, `message` | 400 for a bad request, 408 for an idle session, 500 for an inference failure, 501 for an unsupported input format or no recognizer. Setup errors leave the session waiting for another `setup`. |
| `end_of_stream` | | After `end_of_stream` from the client: the last partial frame is padded with silence and gets its `step`, then the model's text delay is run out on silence so the last words arrive (as `text`/`end_text` only, no further `step`s), then this, and the server closes the connection. |

The first two `step`s report `inactivity_prob` 1.0 for every horizon: the model's output is meaningless for its first four steps (its initial token, the two frames of silence it starts on, and the first frame of audio), and nobody has been heard speaking yet. Afterwards, the probability drops about 0.3 s after speech starts. A client that wants the last words without ending the session sends about a second of silence instead of `end_of_stream`.

Inference runs on a blocking thread per session. Audio arriving faster than the model consumes it is held in a bounded queue, after which the server stops reading the socket, so a client streaming a file faster than real time is slowed to the model's speed rather than buffered without limit.
