import argparse
from pathlib import Path

from media import VOICE_PRESETS, generate_audio_with_timings


DEFAULT_TEXT = (
    "Cześć, tu Jogi! Miałem tylko grzecznie przywitać się z babcią, "
    "ale ten puszysty dywan wyglądał zdecydowanie zbyt podejrzanie."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generuje porównawcze próbki głosu Jogiego.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Tekst czytany w każdej próbce.")
    args = parser.parse_args()

    print("Generuję autorskie warianty głosu Jogiego...")
    for preset_name in VOICE_PRESETS:
        result = generate_audio_with_timings(
            args.text,
            output_name=f"voice_previews/{preset_name}.mp3",
            preset_name=preset_name,
            provider="edge",
        )
        print(f"  {preset_name}: {Path(result.path).resolve()}")


if __name__ == "__main__":
    main()
