import wave
from pathlib import Path

from config import DOG_BREED, DOG_NAME, DOG_PERSONALITY


def build_performance_prompt(transcript: str, outro_tag: str = "") -> str:
    outro = f" {outro_tag}" if outro_tag else ""
    outro_direction = (
        "\nOutro: After the final spoken word, perform the bracketed laugh clearly for "
        "about one second. The laugh is required, nonverbal and must not be read as text."
        if outro_tag
        else ""
    )
    return f"""
# AUDIO PROFILE
The character is a dog named {DOG_NAME}. Breed profile: {DOG_BREED}.
Personality: {DOG_PERSONALITY}.
The voice is youthful and slightly androgynous, but not childish, squeaky or feminine.

# DIRECTOR'S NOTES
Language: Native Polish with careful pronunciation of every Polish character.
Style: Natural spoken storytelling with a confident, cheeky smile in the voice.
Sound as if the dog knows exactly what he is doing, even when the story proves otherwise.
Keep the warmth, but avoid a timid, overly sweet or neutral delivery.
Pacing: Medium-slow with an elastic, lightly bouncing rhythm. Give the setup crisp
energy, use short intentional pauses, then land the punchline with playful swagger.
Gently lengthen only its key word. Do not sing, shout or overact.
Originality: Do not imitate any known fictional character, actor or existing voice.
{outro_direction}

# TRANSCRIPT
[mischievously] {transcript.strip()}{outro}
""".strip()


def save_pcm_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(pcm)


def extract_audio_bytes(response) -> bytes:
    try:
        parts = response.candidates[0].content.parts
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemini TTS nie zwrócił kompletnej odpowiedzi audio.") from exc
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        data = getattr(inline_data, "data", None)
        if data:
            return data
    raise RuntimeError("Gemini TTS nie zwrócił danych audio.")
