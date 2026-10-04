"""Grok Voice for one push-to-talk utterance.

Confirmed 2026-10-04 from the xAI voice docs:

* Speech to text: POST https://api.x.ai/v1/stt
  model ``grok-voice-transcribe-2.0``
  https://docs.x.ai/developers/model-capabilities/audio/speech-to-text
* Text to speech: POST https://api.x.ai/v1/tts
  voice ``eve``, language ``en``, response body is audio bytes (MP3 by default)
  https://docs.x.ai/developers/model-capabilities/audio/text-to-speech

The recording is transcribed, the transcript goes through ``assistant.run_turn``
so tools still hit the planner, and the final assistant text is synthesized.
The API key stays on the server.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from assistant import AssistantError, _error_text, _public_error, _read_request

STT_URL = "https://api.x.ai/v1/stt"
TTS_URL = "https://api.x.ai/v1/tts"
STT_MODEL = "grok-voice-transcribe-2.0"
TTS_VOICE_ID = "eve"
TTS_LANGUAGE = "en"
_TTS_CHAR_LIMIT = 60_000


def transcribe_audio(
    audio: bytes,
    api_key: str,
    *,
    filename: str = "question.wav",
    mime: str = "audio/wav",
    timeout: float = 60.0,
) -> str:
    """Send one recording to the speech-to-text endpoint and return the transcript."""

    key = str(api_key or "").strip()
    if not key:
        raise AssistantError(
            "The assistant needs XAI_API_KEY. Set that environment variable, "
            "or set xai_api_key in Streamlit secrets."
        )
    if not audio:
        raise AssistantError("That recording was empty.")
    safe_name = Path(str(filename or "question.wav")).name.replace('"', "") or "question.wav"
    safe_mime = mime if mime and "/" in str(mime) else "audio/wav"
    fields = [
        ("model", STT_MODEL),
        ("language", "en"),
        ("format", "true"),
        ("keyterm", "EVA"),
        ("keyterm", "resupply"),
    ]
    body, content_type = _multipart(fields, "file", safe_name, audio, safe_mime)
    request = urllib.request.Request(
        STT_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": content_type,
        },
        method="POST",
    )
    raw = _read_request(request, timeout)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        raise AssistantError("Speech to text returned a response that was not JSON.") from None
    if not isinstance(parsed, dict):
        raise AssistantError("Speech to text returned a response that was not JSON.")
    text = str(parsed.get("text") or "").strip()
    if not text:
        raise AssistantError("No speech was heard in that recording.")
    return text


def synthesize_speech(text: str, api_key: str, *, timeout: float = 60.0) -> bytes:
    """Synthesize the assistant reply with the text-to-speech endpoint. Returns MP3 bytes."""

    key = str(api_key or "").strip()
    if not key:
        raise AssistantError(
            "The assistant needs XAI_API_KEY. Set that environment variable, "
            "or set xai_api_key in Streamlit secrets."
        )
    spoken = str(text or "").strip()
    if not spoken:
        raise AssistantError("There was no answer text to speak.")
    if len(spoken) > _TTS_CHAR_LIMIT:
        spoken = spoken[: _TTS_CHAR_LIMIT - 1].rstrip() + "."
    payload = {
        "text": spoken,
        "voice_id": TTS_VOICE_ID,
        "language": TTS_LANGUAGE,
        "text_normalization": True,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        TTS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = ""
            headers = getattr(response, "headers", None)
            if headers is not None:
                content_type = str(headers.get("Content-Type") or "")
            audio = response.read()
    except urllib.error.HTTPError as exc:
        detail = _public_error(_error_text(exc.read()))
        raise AssistantError(f"Text to speech returned {exc.code}. {detail}".rstrip()) from None
    except urllib.error.URLError as exc:
        raise AssistantError(f"Text to speech could not be reached. {_public_error(str(exc.reason))}") from None
    if "json" in content_type.lower():
        detail = _public_error(_error_text(audio))
        raise AssistantError(f"Text to speech did not return audio. {detail}".rstrip())
    if not audio:
        raise AssistantError("Text to speech returned empty audio.")
    return audio


def _multipart(
    fields: list[tuple[str, str]],
    file_field: str,
    filename: str,
    content: bytes,
    mime: str,
) -> tuple[bytes, str]:
    """Multipart body with option fields before the file, as the speech-to-text API requires."""

    boundary = "----issfood" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    header = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {mime}\r\n\r\n"
    )
    chunks.append(header.encode() + content + f"\r\n--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
