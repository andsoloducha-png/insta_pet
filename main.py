import logging
from pathlib import Path

from ai import generate_reel_content
from media import (
    audio_result_to_dict,
    create_cover,
    download_media_links,
    generate_audio_with_timings,
    render_reel_video,
    save_instagram_caption,
    save_manifest,
    save_srt,
)
from sheets import fetch_pending_entry, update_status


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def process_entry(entry: dict) -> dict:
    """Przetwarza jeden wpis i zwraca manifest gotowych plików."""
    row_idx = entry["row_idx"]
    nazwa = str(entry.get("nazwa") or "").strip()
    opis = str(entry.get("opis") or "").strip()

    media_paths = download_media_links(entry.get("link"), row_idx)
    content = generate_reel_content(nazwa, opis, media_paths)

    audio = generate_audio_with_timings(
        content["lektor"],
        output_name=f"audio_{row_idx}.mp3",
    )
    subtitles_path = save_srt(audio.cues, f"napisy_{row_idx}.srt")
    video_path = render_reel_video(
        media_paths,
        audio.path,
        output_name=f"reel_wiersz_{row_idx}.mp4",
        headline=content["naglowek"],
        caption_cues=audio.cues,
        asset_order=content["asset_order"],
    )
    cover_path = create_cover(
        media_paths,
        content["cover_title"],
        output_name=f"okladka_{row_idx}.jpg",
        asset_order=content["asset_order"],
    )
    caption_path = save_instagram_caption(content["caption"], f"opis_wiersz_{row_idx}.txt")

    manifest = {
        "row_idx": row_idx,
        "source": {"nazwa": nazwa, "opis": opis},
        "content": content,
        "media_paths": media_paths,
        "audio": audio_result_to_dict(audio),
        "outputs": {
            "video": video_path,
            "cover": cover_path,
            "caption": caption_path,
            "subtitles": subtitles_path,
        },
    }
    manifest_path = save_manifest(manifest, f"manifest_{row_idx}.json")
    manifest["outputs"]["manifest"] = manifest_path
    return manifest


def main() -> None:
    print("Szukam pierwszego wpisu z wartością 'tak' w kolumnie instagram...")
    entry = fetch_pending_entry()
    if not entry:
        print("Brak nowych wpisów do przetworzenia.")
        return

    row_idx = entry["row_idx"]
    print(f"Przetwarzanie wiersza {row_idx}: {entry.get('nazwa') or '(bez nazwy)'}")
    try:
        manifest = process_entry(entry)
        update_status(entry["worksheet"], row_idx, entry["col_idx"], "wygenerowane")
    except Exception as exc:
        LOGGER.exception("Nie udało się przetworzyć wiersza %s", row_idx)
        try:
            update_status(entry["worksheet"], row_idx, entry["col_idx"], "błąd")
        except Exception:
            LOGGER.warning("Nie udało się zapisać statusu błędu w arkuszu.")
        raise SystemExit(f"Błąd: {exc}") from exc

    outputs = manifest["outputs"]
    print("\nGotowe pliki:")
    for label in ("video", "cover", "caption", "subtitles", "manifest"):
        print(f"  {label}: {Path(outputs[label]).resolve()}")


if __name__ == "__main__":
    main()
