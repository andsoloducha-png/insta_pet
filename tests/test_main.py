import unittest
from unittest.mock import patch

from media import AudioResult, CaptionCue, TimedWord
from main import process_entry


class ProcessEntryTests(unittest.TestCase):
    @patch("main.save_manifest", return_value="output/manifest_7.json")
    @patch("main.save_instagram_caption", return_value="output/opis_wiersz_7.txt")
    @patch("main.create_cover", return_value="output/okladka_7.jpg")
    @patch("main.render_reel_video", return_value="output/reel_wiersz_7.mp4")
    @patch("main.save_srt", return_value="output/napisy_7.srt")
    @patch("main.generate_audio_with_timings")
    @patch("main.generate_reel_content")
    @patch("main.download_media_links", return_value=["output/a.jpg", "output/b.gif"])
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
    ):
        word = TimedWord("Cześć!", 0.0, 0.5)
        cue = CaptionCue(0.0, 0.7, (word,))
        generate_audio.return_value = AudioResult("output/audio_7.mp3", (word,), (cue,))
        generate_content.return_value = {
            "lektor": "Cześć!",
            "naglowek": "JOGI W AKCJI",
            "cover_title": "LEŚNY NOS",
            "caption_body": "Jogi poznaje leśne zapachy.",
            "hashtags": ["#JogiPudel", "#PsiSpacer"],
            "caption": "Jogi poznaje leśne zapachy.\n\n#JogiPudel #PsiSpacer",
            "alt_text": "Pudel podczas spaceru w lesie.",
            "asset_order": [1, 0],
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
            ["output/a.jpg", "output/b.gif"],
        )
        render_video.assert_called_once()
        create_cover.assert_called_once()
        save_caption.assert_called_once()
        save_srt.assert_called_once()
        save_manifest.assert_called_once()
        self.assertEqual(result["content"]["asset_order"], [1, 0])
        self.assertEqual(result["outputs"]["manifest"], "output/manifest_7.json")
        self.assertEqual(result["outputs"]["video"], "output/reel_wiersz_7.mp4")


if __name__ == "__main__":
    unittest.main()
