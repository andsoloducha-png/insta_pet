import argparse
import time
from pathlib import Path

from google import genai
from google.genai import types

from config import BASE_DIR, GEMINI_API_KEY, GEMINI_TTS_MODEL
from tts_support import build_performance_prompt, extract_audio_bytes, save_pcm_wav


OUTPUT_DIR = BASE_DIR / "output" / "gemini_voice_previews"
VOICE_CANDIDATES = ("Puck", "Fenrir", "Sadachbia", "Achird")
SIGNATURE_LAUGH_VARIANTS = {
    "warm": "[laughs briefly: a warm, buoyant two-beat chuckle, playful and original, no words]",
    "springy": "[laughs briefly: a light springy three-beat laugh with a gentle rise, original and nonverbal]",
    "mischief": "[chuckles briefly: mischievous, relaxed and warm, two beats, original and no words]",
    "mischief_v2": "[laughs]",
}
DEFAULT_TEXT = (
    "Cześć, tu Jogi! Miałem tylko grzecznie przywitać się z babcią, "
    "ale ten puszysty dywan wyglądał zdecydowanie zbyt podejrzanie."
)


def generate_preview(
    client: genai.Client,
    voice: str,
    transcript: str,
    output_name: str | None = None,
    outro_tag: str = "",
) -> Path:
    prompt = build_performance_prompt(transcript, outro_tag)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=GEMINI_TTS_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                ),
            )
            output_path = OUTPUT_DIR / (output_name or f"gemini_{voice.lower()}.wav")
            save_pcm_wav(output_path, extract_audio_bytes(response))
            return output_path
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt)
    raise RuntimeError(f"Nie udało się wygenerować głosu {voice}: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Porównuje cztery głosy Gemini TTS.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Polski tekst próbki.")
    parser.add_argument(
        "--achird-laughs",
        action="store_true",
        help="Generuje trzy oryginalne warianty końcowego śmiechu głosem Achird.",
    )
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        raise SystemExit("Brak GEMINI_API_KEY w pliku .env.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"Model: {GEMINI_TTS_MODEL}")
    if args.achird_laughs:
        for variant, outro_tag in SIGNATURE_LAUGH_VARIANTS.items():
            path = generate_preview(
                client,
                "Achird",
                args.text,
                output_name=f"gemini_achird_laugh_{variant}.wav",
                outro_tag=outro_tag,
            )
            print(f"  Achird / {variant}: {path.resolve()}")
        return
    for voice in VOICE_CANDIDATES:
        path = generate_preview(client, voice, args.text)
        print(f"  {voice}: {path.resolve()}")


if __name__ == "__main__":
    main()
