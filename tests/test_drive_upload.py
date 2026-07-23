import unittest
from pathlib import Path
from unittest.mock import Mock

from drive_upload import DriveFile, GoogleDriveUploader, _escape_query_value
from media import OUTPUT_DIR


class DriveUploadTests(unittest.TestCase):
    def test_escapes_drive_query_values(self):
        self.assertEqual(_escape_query_value("Jogi's folder"), "Jogi\\'s folder")

    def test_sync_folder_uploads_files_and_skips_manifest(self):
        uploader = object.__new__(GoogleDriveUploader)
        uploader.ensure_folder = Mock(
            return_value=DriveFile("folder-id", "wiersz_2", "folder-url")
        )
        uploader.sync_file = Mock(
            side_effect=lambda path, _folder_id: DriveFile(
                f"id-{Path(path).name}",
                Path(path).name,
                f"url-{Path(path).name}",
            )
        )

        folder = OUTPUT_DIR / "_test_drive_sync"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            (folder / "reel.mp4").write_bytes(b"video")
            (folder / "opis.txt").write_text("Opis", encoding="utf-8")
            (folder / "manifest.json").write_text("{}", encoding="utf-8")

            result = uploader.sync_folder(
                folder,
                "parent-id",
                exclude_names={"manifest.json"},
            )
        finally:
            for name in ("reel.mp4", "opis.txt", "manifest.json"):
                (folder / name).unlink(missing_ok=True)
            folder.rmdir()

        self.assertEqual(result["folder_id"], "folder-id")
        self.assertEqual(set(result["files"]), {"opis.txt", "reel.mp4"})
        uploader.ensure_folder.assert_called_once_with("parent-id", "_test_drive_sync")
        self.assertEqual(uploader.sync_file.call_count, 2)


if __name__ == "__main__":
    unittest.main()
