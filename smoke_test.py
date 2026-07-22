"""Ręczny test całego pipeline bez odczytu i zapisu Google Sheets."""

from pathlib import Path

from ai import generate_reel_content
from media import (
    audio_result_to_dict,
    create_cover,
    generate_audio_with_timings,
    render_reel_video,
    save_instagram_caption,
    save_manifest,
    save_srt,
)


def main() -> None:
    source = Path("output/temp_5.jpg")
    if not source.exists():
        raise SystemExit("Brak output/temp_5.jpg — test potrzebuje lokalnego zdjęcia.")

    media_paths = [str(source.resolve())]
    content = generate_reel_content(
        "Leśny spacer Jogiego",
        "Jogi zatrzymał się przy dużym drzewie podczas spokojnego spaceru.",
        media_paths,
    )
    audio = generate_audio_with_timings(content["lektor"], "smoke_audio.mp3")
    subtitles = save_srt(audio.cues, "smoke_napisy.srt")
    video = render_reel_video(
        media_paths,
        audio.path,
        "smoke_reel.mp4",
        headline=content["naglowek"],
        caption_cues=audio.cues,
        asset_order=content["asset_order"],
    )
    cover = create_cover(media_paths, content["cover_title"], "smoke_okladka.jpg")
    caption = save_instagram_caption(content["caption"], "smoke_opis.txt")
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
        "smoke_manifest.json",
    )
    print("Test end-to-end zakończony:")
    for path in (video, cover, caption, subtitles, manifest):
        print(Path(path).resolve())


if __name__ == "__main__":
    main()
