import hashlib
import json
import logging
from pathlib import Path

from ai import generate_reel_content
from config import GOOGLE_DRIVE_OUTPUT_FOLDER_ID, GOOGLE_DRIVE_UPLOAD_ENABLED
from drive_upload import DriveUploadError, GoogleDriveUploader
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
from reel_formats import choose_reel_format, normalise_reel_format
from sheets import fetch_pending_entry, update_status


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _content_cache_path(row_idx: int) -> Path:
    return OUTPUT_DIR / f"wiersz_{row_idx}" / "content_cache.json"


def _recent_reel_formats(row_idx: int, limit: int = 2) -> list[str]:
    manifests = [
        path
        for path in OUTPUT_DIR.glob("wiersz_*/manifest.json")
        if path.parent.name != f"wiersz_{row_idx}"
    ]
    # Uwzględniamy również manifesty utworzone przed wprowadzeniem podfolderów.
    manifests.extend(OUTPUT_DIR.glob("manifest_*.json"))
    formats: list[str] = []
    for path in sorted(manifests, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            value = data.get("content", {}).get("format_id")
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        if value:
            formats.append(normalise_reel_format(value))
        if len(formats) >= limit:
            break
    return formats


def _source_fingerprint(entry: dict) -> str:
    source = {
        "row_idx": entry["row_idx"],
        "nazwa": str(entry.get("nazwa") or "").strip(),
        "opis": str(entry.get("opis") or "").strip(),
        "link": str(entry.get("link") or "").strip(),
    }
    payload = json.dumps(source, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cached_content(entry: dict) -> dict | None:
    path = _content_cache_path(entry["row_idx"])
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if cached.get("fingerprint") != _source_fingerprint(entry):
        return None
    content = cached.get("content")
    return content if isinstance(content, dict) and content.get("lektor") else None


def _save_content_cache(entry: dict, content: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _content_cache_path(entry["row_idx"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"fingerprint": _source_fingerprint(entry), "content": content},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "resource_exhausted" in message or "quota exceeded" in message


def _upload_manifest_outputs(manifest: dict, uploader: GoogleDriveUploader) -> dict:
    manifest_path = Path(manifest["outputs"]["manifest"]).resolve()
    local_folder = manifest_path.parent
    drive = uploader.sync_folder(
        local_folder,
        GOOGLE_DRIVE_OUTPUT_FOLDER_ID,
        exclude_names={manifest_path.name},
    )
    manifest["drive"] = drive
    saved_manifest_path = save_manifest(
        manifest,
        f"{local_folder.name}/{manifest_path.name}",
    )
    uploaded_manifest = uploader.sync_file(saved_manifest_path, drive["folder_id"])
    drive["files"][manifest_path.name] = uploaded_manifest.as_dict()
    return drive


def process_entry(entry: dict) -> dict:
    """Przetwarza jeden wpis i zwraca manifest gotowych plików."""
    row_idx = entry["row_idx"]
    nazwa = str(entry.get("nazwa") or "").strip()
    opis = str(entry.get("opis") or "").strip()

    media_paths = download_media_links(entry.get("link"), row_idx)
    content = _load_cached_content(entry)
    if content:
        LOGGER.info("Używam zachowanego tekstu dla wiersza %s; pomijam ponowne wywołania AI.", row_idx)
        content["format_id"] = normalise_reel_format(content.get("format_id"))
    else:
        content = generate_reel_content(nazwa, opis, media_paths)
        requested_format = content.get("format_id")
        content["format_id"] = choose_reel_format(
            requested_format,
            len(media_paths),
            _recent_reel_formats(row_idx),
        )
        if content["format_id"] != requested_format:
            LOGGER.info(
                "Format %s zmieniony na %s, aby dopasować media lub uniknąć powtórki.",
                requested_format,
                content["format_id"],
            )
        _save_content_cache(entry, content)

    output_prefix = f"wiersz_{row_idx}"
    audio = generate_audio_with_timings(
        content["lektor"],
        output_name=f"{output_prefix}/audio.mp3",
    )
    subtitles_path = save_srt(audio.cues, f"{output_prefix}/napisy.srt")
    video_path = render_reel_video(
        media_paths,
        audio.path,
        output_name=f"{output_prefix}/reel.mp4",
        headline=content["naglowek"],
        caption_cues=audio.cues,
        asset_order=content["asset_order"],
        format_id=content["format_id"],
    )
    cover_path = create_cover(
        media_paths,
        content["cover_title"],
        output_name=f"{output_prefix}/okladka.jpg",
        asset_order=content["asset_order"],
    )
    caption_path = save_instagram_caption(content["caption"], f"{output_prefix}/opis.txt")

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
    manifest_path = save_manifest(manifest, f"{output_prefix}/manifest.json")
    manifest["outputs"]["manifest"] = manifest_path
    _content_cache_path(row_idx).unlink(missing_ok=True)
    return manifest


def main() -> None:
    uploader = None
    if GOOGLE_DRIVE_UPLOAD_ENABLED:
        try:
            uploader = GoogleDriveUploader()
            folder = uploader.verify_folder(GOOGLE_DRIVE_OUTPUT_FOLDER_ID)
            LOGGER.info("Google Drive: folder docelowy „%s” jest gotowy.", folder["name"])
        except DriveUploadError as exc:
            raise SystemExit(f"Błąd konfiguracji Google Drive: {exc}") from exc

    print("Szukam pierwszego wpisu z wartością 'tak' w kolumnie instagram...")
    entry = fetch_pending_entry()
    if not entry:
        print("Brak nowych wpisów do przetworzenia.")
        return

    row_idx = entry["row_idx"]
    print(f"Przetwarzanie wiersza {row_idx}: {entry.get('nazwa') or '(bez nazwy)'}")
    manifest = None
    try:
        manifest = process_entry(entry)
        if uploader:
            drive = _upload_manifest_outputs(manifest, uploader)
            LOGGER.info("Google Drive: gotowa rolka jest dostępna pod %s", drive["folder_url"])
        update_status(entry["worksheet"], row_idx, entry["col_idx"], "wygenerowane")
    except DriveUploadError as exc:
        LOGGER.exception("Nie udało się wysłać wiersza %s na Google Drive", row_idx)
        if manifest:
            _save_content_cache(entry, manifest["content"])
        raise SystemExit(
            "Błąd wysyłania na Google Drive. Pliki lokalne zostały zachowane, "
            "a status w arkuszu pozostaje bez zmian, aby można było bezpiecznie ponowić próbę. "
            f"Szczegóły: {exc}"
        ) from exc
    except Exception as exc:
        LOGGER.exception("Nie udało się przetworzyć wiersza %s", row_idx)
        if _is_quota_error(exc):
            LOGGER.warning(
                "Limit Gemini został wyczerpany. Status w arkuszu pozostaje bez zmian, "
                "aby można było bezpiecznie ponowić wiersz później."
            )
        else:
            try:
                update_status(entry["worksheet"], row_idx, entry["col_idx"], "błąd")
            except Exception:
                LOGGER.warning("Nie udało się zapisać statusu błędu w arkuszu.")
        raise SystemExit(f"Błąd: {exc}") from exc

    outputs = manifest["outputs"]
    print("\nGotowe pliki:")
    for label in ("video", "cover", "caption", "subtitles", "manifest"):
        print(f"  {label}: {Path(outputs[label]).resolve()}")
    if manifest.get("drive"):
        print(f"  Google Drive: {manifest['drive']['folder_url']}")


if __name__ == "__main__":
    main()
