"""Krótki test aktualnie wybranego dostawcy TTS bez arkusza i renderowania rolki."""

import argparse
from pathlib import Path

from config import TTS_PROVIDER
from media import generate_audio_with_timings


DEFAULT_TEXT = (
    "No dobrze, człowieki. Mieliście jeden dywan i jedno zadanie. "
    "Ja tylko sprawdziłem, czy na pewno mnie obserwujecie."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generuje próbkę głosu wskazanego przez TTS_PROVIDER w pliku .env."
    )
    parser.add_argument(
        "--tekst",
        default=DEFAULT_TEXT,
        help="Tekst próbki. Bez parametru zostanie użyty krótki tekst Jogiego.",
    )
    args = parser.parse_args()

    result = generate_audio_with_timings(
        args.tekst,
        output_name=f"voice_test/preview_{TTS_PROVIDER}.mp3",
    )
    print(f"Gotowe: {Path(result.path).resolve()}")
    print(f"Dostawca: {result.provider}; model: {result.model or 'nie dotyczy'}")
    print(f"Zsynchronizowane słowa: {len(result.words)}")


if __name__ == "__main__":
    main()
