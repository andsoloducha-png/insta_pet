import unittest
from unittest.mock import patch

from media import AudioResult, CaptionCue, TimedWord
from main import (
    _content_cache_path,
    _load_cached_content,
    _save_content_cache,
    _upload_manifest_outputs,
    process_entry,
)


class ProcessEntryTests(unittest.TestCase):
    @patch("main.GOOGLE_DRIVE_OUTPUT_FOLDER_ID", "folder-rodzic")
    @patch("main.save_manifest", return_value="output/wiersz_7/manifest.json")
    def test_uploads_complete_reel_folder_and_manifest(self, save_manifest):
        uploaded_manifest = type(
            "Uploaded",
            (),
            {
                "as_dict": lambda self: {
                    "id": "manifest-id",
                    "name": "manifest.json",
                    "url": "manifest-url",
                }
            },
        )()
        uploader = type(
            "Uploader",
            (),
            {
                "sync_folder": lambda self, *args, **kwargs: {
                    "folder_id": "folder-7",
                    "folder_url": "https://drive.google.com/drive/folders/folder-7",
                    "files": {},
                },
                "sync_file": lambda self, *args, **kwargs: uploaded_manifest,
            },
        )()
        manifest = {
            "outputs": {"manifest": "output/wiersz_7/manifest.json"},
            "content": {"lektor": "Test"},
        }

        result = _upload_manifest_outputs(manifest, uploader)

        self.assertEqual(result["folder_id"], "folder-7")
        self.assertEqual(result["files"]["manifest.json"]["id"], "manifest-id")
        self.assertEqual(manifest["drive"]["folder_url"], result["folder_url"])
        save_manifest.assert_called_once()

    def test_content_cache_is_reused_only_for_unchanged_source(self):
        entry = {
            "row_idx": 9876,
            "nazwa": "Spacer",
            "opis": "Jogi spotkał Zuzię.",
            "link": "https://example.com/a.jpg",
        }
        content = {"lektor": "Mam plan.", "caption": "Mam plan."}
        cache_path = _content_cache_path(entry["row_idx"])
        try:
            _save_content_cache(entry, content)
            self.assertEqual(_load_cached_content(entry), content)
            self.assertIsNone(_load_cached_content({**entry, "opis": "Zmieniony opis"}))
        finally:
            cache_path.unlink(missing_ok=True)

    @patch("main._recent_reel_formats", return_value=[])
    @patch("main.save_manifest", return_value="output/wiersz_7/manifest.json")
    @patch("main.save_instagram_caption", return_value="output/wiersz_7/opis.txt")
    @patch("main.create_cover", return_value="output/wiersz_7/okladka.jpg")
    @patch("main.render_reel_video", return_value="output/wiersz_7/reel.mp4")
    @patch("main.save_srt", return_value="output/wiersz_7/napisy.srt")
    @patch("main.generate_audio_with_timings")
    @patch("main.generate_reel_content")
    @patch(
        "main.download_media_links",
        return_value=["output/wiersz_7/media_1.jpg", "output/wiersz_7/media_2.gif"],
    )
    def test_builds_complete_manifest_for_multiple_media(
        self,
        download_media,
        generate_content,
        generate_audio,
        save_srt,
        render_video,
        create_cover,
        save_caption,
        save_manifest,
        _recent_formats,
    ):
        word = TimedWord("Cześć!", 0.0, 0.5)
        cue = CaptionCue(0.0, 0.7, (word,))
        generate_audio.return_value = AudioResult("output/wiersz_7/audio.wav", (word,), (cue,))
        generate_content.return_value = {
            "lektor": "Cześć!",
            "naglowek": "JOGI W AKCJI",
            "cover_title": "LEŚNY NOS",
            "caption_body": "Jogi poznaje leśne zapachy.",
            "hashtags": ["#JogiPudel", "#PsiSpacer"],
            "caption": "Jogi poznaje leśne zapachy.\n\n#JogiPudel #PsiSpacer",
            "alt_text": "Pudel podczas spaceru w lesie.",
            "asset_order": [1, 0],
            "format_id": "mini_story",
        }

        result = process_entry(
            {
                "row_idx": 7,
                "nazwa": "Leśny spacer",
                "opis": "Jogi węszy przy drzewie.",
                "link": "https://example.com/a.jpg, https://example.com/b.gif",
            }
        )

        download_media.assert_called_once()
        generate_content.assert_called_once_with(
            "Leśny spacer",
            "Jogi węszy przy drzewie.",
            ["output/wiersz_7/media_1.jpg", "output/wiersz_7/media_2.gif"],
        )
        self.assertEqual(
            render_video.call_args.kwargs["output_name"],
            "wiersz_7/reel.mp4",
        )
        self.assertEqual(render_video.call_args.kwargs["format_id"], "mini_story")
        create_cover.assert_called_once()
        save_caption.assert_called_once()
        save_srt.assert_called_once()
        save_manifest.assert_called_once()
        self.assertEqual(result["content"]["asset_order"], [1, 0])
        self.assertEqual(result["content"]["format_id"], "mini_story")
        self.assertEqual(result["outputs"]["manifest"], "output/wiersz_7/manifest.json")
        self.assertEqual(result["outputs"]["video"], "output/wiersz_7/reel.mp4")


if __name__ == "__main__":
    unittest.main()
