"""Ręczny test całego pipeline bez odczytu i zapisu Google Sheets."""

import argparse
from pathlib import Path

from ai import generate_reel_content
from config import TTS_SIGNATURE_LAUGH_FILE
from gemini_voice_preview import DEFAULT_TEXT as VOICE_PREVIEW_TEXT
from media import (
    AudioResult,
    _append_signature_laugh,
    _approximate_words,
    _audio_duration,
    audio_result_to_dict,
    build_caption_cues,
    create_cover,
    generate_audio_with_timings,
    limit_caption_cues,
    render_reel_video,
    save_instagram_caption,
    save_manifest,
    save_srt,
)


def _offline_audio_and_content() -> tuple[AudioResult, dict]:
    narration = Path("output/gemini_voice_previews/gemini_achird.wav")
    if not narration.exists():
        raise SystemExit(
            "Brak próbki Achird. Uruchom wcześniej gemini_voice_preview.py."
        )
    output = Path("output/smoke/audio.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    _append_signature_laugh(
        narration.resolve(),
        Path(TTS_SIGNATURE_LAUGH_FILE),
        output.resolve(),
    )
    narration_duration = _audio_duration(narration)
    words = _approximate_words(VOICE_PREVIEW_TEXT, narration_duration)
    cues = limit_caption_cues(build_caption_cues(words), narration_duration)
    audio = AudioResult(
        str(output.resolve()),
        tuple(words),
        tuple(cues),
        "Achird",
        "achird_warm_signature",
        "gemini",
    )
    content = {
        "lektor": VOICE_PREVIEW_TEXT,
        "naglowek": "TEST GŁOSU ACHIRD",
        "cover_title": "ACHIRD I WARM",
        "caption": "Lokalny test głosu Achird z podpisem dźwiękowym warm.",
        "asset_order": [0],
        "format_id": "punchline",
    }
    return audio, content


def main() -> None:
    parser = argparse.ArgumentParser(description="Testuje kompletną rolkę bez Google Sheets.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Nie wywołuje Gemini; używa istniejącej próbki Achird.",
    )
    args = parser.parse_args()

    candidates = [
        Path("output/temp_5.jpg"),
        Path("output/wiersz_27/media_1.jpg"),
        Path("output/media_27_1.jpg"),
    ]
    generated_media = sorted(
        (
            path
            for path in Path("output").glob("wiersz_*/media_*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates.extend(generated_media)
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise SystemExit(
            "Brak medium testowego. Dodaj obraz do output/temp_5.jpg "
            "albo najpierw wygeneruj dowolną rolkę."
        )

    media_paths = [str(source.resolve())]
    if args.offline:
        audio, content = _offline_audio_and_content()
    else:
        content = generate_reel_content(
            "Spotkanie na spacerze",
            "Podczas spaceru Jogi spotkał drugiego psa. Oba psy zatrzymały się obok siebie.",
            media_paths,
        )
        audio = generate_audio_with_timings(content["lektor"], "smoke/audio.mp3")
    subtitles = save_srt(audio.cues, "smoke/napisy.srt")
    video = render_reel_video(
        media_paths,
        audio.path,
        "smoke/reel.mp4",
        headline=content["naglowek"],
        caption_cues=audio.cues,
        asset_order=content["asset_order"],
        format_id=content["format_id"],
    )
    cover = create_cover(media_paths, content["cover_title"], "smoke/okladka.jpg")
    caption = save_instagram_caption(content["caption"], "smoke/opis.txt")
    manifest = save_manifest(
        {
            "test": True,
            "content": content,
            "audio": audio_result_to_dict(audio),
            "outputs": {
                "video": video,
                "cover": cover,
                "caption": caption,
                "subtitles": subtitles,
            },
        },
        "smoke/manifest.json",
    )
    print("Test end-to-end zakończony:")
    for path in (video, cover, caption, subtitles, manifest):
        print(Path(path).resolve())


if __name__ == "__main__":
    main()
