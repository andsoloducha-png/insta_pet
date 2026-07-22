import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import types

from ai import ReelContent, _normalise_content, _parse_response, generate_reel_content


class ReelContentTests(unittest.TestCase):
    @staticmethod
    def _content() -> ReelContent:
        return ReelContent(
            voiceover="Jestem Jogi i sprawdzam nową trasę.",
            headline="Mój leśny plan",
            cover_title="Leśny plan",
            caption_body="Pudel Jogi ruszył dziś do lasu.",
            hashtags=["JogiPudel", "pudel", "instapies"],
            alt_text="Pudel siedzi przy drzewie.",
            asset_order=[],
        )

    def test_normalisation_deduplicates_order_and_hashtags(self):
        content = ReelContent(
            voiceover="Jestem Jogi i sprawdzam nową trasę.",
            headline="Mój leśny plan",
            cover_title="Leśny plan",
            caption_body="Pudel Jogi ruszył dziś do lasu.",
            hashtags=["JogiPudel", "#pudel", "pudel", "instapies", "las", "szósty"],
            alt_text="Pudel siedzi przy drzewie.",
            asset_order=[2, 2, 99, 0],
        )

        result = _normalise_content(content, media_count=3)

        self.assertEqual(result["asset_order"], [2, 0, 1])
        self.assertEqual(result["hashtags"], ["#JogiPudel", "#pudel", "#instapies", "#las", "#szósty"])
        self.assertIn("\n\n#JogiPudel", result["caption"])
        self.assertEqual(result["naglowek"], "MÓJ LEŚNY PLAN")

    def test_parse_error_does_not_expose_generated_content(self):
        response = SimpleNamespace(
            parsed=None,
            text='{"voiceover":"tajna treść ucięta',
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(value="MAX_TOKENS"))],
        )

        with self.assertRaises(ValueError) as raised:
            _parse_response(response, media_count=0)

        message = str(raised.exception)
        self.assertIn("finish_reason=MAX_TOKENS", message)
        self.assertNotIn("tajna treść", message)

    @patch("ai.time.sleep", return_value=None)
    @patch("ai.GEMINI_MAX_RETRIES", 3)
    @patch("ai.GEMINI_FALLBACK_MODELS", ())
    @patch("ai.GEMINI_MODEL", "gemini-3.5-flash")
    @patch("ai.GEMINI_API_KEY", "test-key")
    def test_retries_truncated_json_on_same_model(self, _sleep):
        truncated = SimpleNamespace(
            parsed=None,
            text='{"voiceover":"ucięte',
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(value="MAX_TOKENS"))],
        )
        complete = SimpleNamespace(parsed=self._content(), text=None, candidates=[])
        generate = Mock(side_effect=[truncated, complete])
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))

        with patch("ai.genai.Client", return_value=client):
            result = generate_reel_content("Spacer", "Jogi przy drzewie", [])

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result["naglowek"], "MÓJ LEŚNY PLAN")
        first_config = generate.call_args_list[0].kwargs["config"]
        self.assertEqual(first_config.max_output_tokens, 4096)
        self.assertEqual(first_config.thinking_config.thinking_level, types.ThinkingLevel.MINIMAL)


if __name__ == "__main__":
    unittest.main()
