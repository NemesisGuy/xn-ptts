# /// script
# requires-python = ">=3.9"
# dependencies = [
#    "requests",
#    "huggingface_hub",
# ]
# ///
"""Clone a voice with the Gradium API and download it for a Phonon checkpoint.

Uploads a recording of the speaker (10 seconds of clear speech are used, from --start on),
waits for Gradium to make the voice for the checkpoint given as --model, and writes its voice
file. The voice is saved in your Gradium account: download-voice.py fetches it again by the ID
this prints. Needs a Gradium API key in GRADIUM_API_KEY. Clone only voices you have the
speaker's consent to use.

    uv run scripts/clone-voice.py speaker.wav --lang en --model hf://OWNER/MODEL -o voice.safetensors
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://api.gradium.ai/api"


def check(r: requests.Response, *args, **kwargs) -> None:
    """Exits with the API's own message when a request fails."""
    if not r.ok:
        try:
            detail = r.json()["detail"]
        except (ValueError, KeyError):
            detail = r.text
        sys.exit(f"Gradium API error {r.status_code}: {detail}")


def model_ext(model: str) -> str:
    """The checkpoint's model id, `sig@epoch`, from its config.json."""
    if model.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo, _, revision = model.removeprefix("hf://").partition("@")
        path = Path(hf_hub_download(repo, "config.json", revision=revision or None))
    else:
        path = Path(model)
        if path.is_dir():
            path = path / "config.json"
    model_id = json.loads(path.read_text()).get("model_id")
    if not model_id:
        sys.exit(f"{model} has no model_id in its config.json: Gradium voices need a Phonon checkpoint")
    return f"{model_id['sig']}@{model_id['epoch']}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path, help="a recording of the speaker (wav, mp3, flac, ...)")
    p.add_argument("--lang", required=True, choices=["en", "fr", "de", "es", "pt"], help="the recording's language")
    p.add_argument("--model", required=True, help="checkpoint folder, config.json, or hf://OWNER/MODEL[@REVISION]")
    p.add_argument("--name", help="the voice's name in your account (default: the audio file's name)")
    p.add_argument("--start", type=float, default=0.0, help="where the speech starts in the recording, in seconds")
    p.add_argument("-o", "--out", type=Path, help="output file (default: VOICE_ID.safetensors)")
    args = p.parse_args()

    key = os.environ.get("GRADIUM_API_KEY")
    if not key:
        sys.exit("set GRADIUM_API_KEY to your Gradium API key")
    ext = model_ext(args.model)
    s = requests.Session()
    s.headers["x-api-key"] = key
    s.hooks["response"] = check

    with args.audio.open("rb") as f:
        form = {"name": args.name or args.audio.stem, "language": args.lang, "start_s": args.start}
        r = s.post(f"{API}/voices/", files={"audio_file": (args.audio.name, f)}, data=form)
    voice_id = r.json().get("uid")
    if not voice_id:
        sys.exit(f"cloning failed: {r.json().get('error')}")
    print(f"created voice {voice_id}")

    # The voice is made for each model in turn: wait for this checkpoint's.
    for _ in range(60):
        r = s.get(f"{API}/voices/embeddings/{voice_id}/{ext}")
        if r.json().get("found"):
            break
        if not r.json().get("model_ext"):
            sys.exit(f"the Gradium API has no voices for model {ext}")
        time.sleep(2)
    else:
        sys.exit(f"voice {voice_id} is not ready for model {ext} yet: try download-voice.py later")

    r = s.get(f"{API}/voices/embeddings/{voice_id}/{ext}/blob", params={"as_safetensors": "1"})
    out = args.out or Path(f"{voice_id}.safetensors")
    out.write_bytes(r.content)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
