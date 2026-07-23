import unittest
import wave

from gemini_voice_preview import OUTPUT_DIR
from tts_support import build_performance_prompt, save_pcm_wav


class GeminiVoicePreviewTests(unittest.TestCase):
    def test_prompt_separates_direction_from_polish_transcript(self):
        prompt = build_performance_prompt("Zażółć gęślą jaźń.")
        self.assertIn("Native Polish", prompt)
        self.assertIn("# TRANSCRIPT", prompt)
        self.assertIn("Zażółć gęślą jaźń.", prompt)
        self.assertIn("Do not imitate any known fictional character", prompt)
        self.assertIn("confident, cheeky smile", prompt)
        self.assertIn("playful swagger", prompt)

    def test_optional_laugh_is_added_only_after_transcript(self):
        prompt = build_performance_prompt("To jest puenta.", "[laughs briefly]")
        self.assertIn("To jest puenta. [laughs briefly]", prompt)
        self.assertIn("The laugh is required", prompt)

    def test_saves_pcm_as_24khz_mono_wav(self):
        path = OUTPUT_DIR / "_test_preview.wav"
        try:
            save_pcm_wav(path, b"\x00\x00" * 240)
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getsampwidth(), 2)
                self.assertEqual(audio.getframerate(), 24_000)
                self.assertEqual(audio.getnframes(), 240)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
