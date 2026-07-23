import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import types

from ai import (
    ReelContent,
    _breed_is_relevant,
    _clean_cover_title,
    _content_quality_issues,
    _normalise_content,
    _parse_response,
    generate_reel_content,
)


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
            voiceover="Jestem pudlem toy i sprawdzam nową trasę.",
            headline="Mój leśny plan",
            cover_title="Leśny plan",
            caption_body="Jako pudel toy ruszyłem dziś do lasu.",
            hashtags=["JogiPudel", "#pudeltoy", "pudel", "życiepsa", "las", "viral"],
            alt_text="Pudel toy siedzi przy drzewie.",
            asset_order=[2, 2, 99, 0],
        )

        result = _normalise_content(content, media_count=3)

        self.assertEqual(result["asset_order"], [2, 0, 1])
        self.assertEqual(
            result["hashtags"],
            ["#pudelminiaturowy", "#pies", "#pudel", "#zyciepsa", "#las"],
        )
        self.assertNotIn("JogiPudel", result["caption"])
        self.assertNotIn("toy", result["caption"].lower())
        self.assertIn("pudlem miniaturowym", result["lektor"].lower())
        self.assertIn("pudel miniaturowy", result["alt_text"].lower())
        self.assertEqual(result["naglowek"], "MÓJ LEŚNY PLAN")
        self.assertEqual(result["format_id"], "punchline")

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

    def test_quality_gate_detects_mixed_perspective_and_wrong_breed(self):
        content = ReelContent(
            voiceover="Jogi wchodzi do domu. Potem zacząłem oglądać każdy kąt.",
            headline="Pierwsza wizyta",
            cover_title="U babci",
            caption_body=(
                "Pudel Jogi składa wizytę u babci. Później sprawdziłem puszysty dywan "
                "i zacząłem poznawać nowe miejsce."
            ),
            hashtags=["JogiPudel", "pudeltoy", "pies"],
            alt_text="Pudel w domu.",
            asset_order=[0],
        )

        issues = _content_quality_issues(content)

        self.assertTrue(any("trzeciej osobie" in issue for issue in issues))
        self.assertTrue(any("toy" in issue for issue in issues))

    def test_cover_title_removes_redundant_possessive_and_limits_words(self):
        self.assertEqual(_clean_cover_title("Moja pierwsza wizyta u babci"), "PIERWSZA WIZYTA U BABCI")

    def test_breed_is_only_relevant_for_breed_related_topics(self):
        self.assertFalse(_breed_is_relevant("Wizyta u babci", "Jogi nasikał na dywan"))
        self.assertTrue(_breed_is_relevant("Pierwsze strzyżenie", "Czesanie sierści pudla"))

    @patch("ai.time.sleep", return_value=None)
    @patch("ai.GEMINI_MAX_RETRIES", 3)
    @patch("ai.GEMINI_FALLBACK_MODELS", ())
    @patch("ai.GEMINI_MODEL", "gemini-3.5-flash")
    @patch("ai.GEMINI_THINKING_LEVEL", "minimal")
    @patch("ai.GEMINI_API_KEY", "test-key")
    @patch("ai.GEMINI_EDITOR_ENABLED", False)
    @patch("ai.GEMINI_PROOFREADER_ENABLED", False)
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
        self.assertEqual(first_config.max_output_tokens, 8192)
        self.assertEqual(first_config.thinking_config.thinking_level, types.ThinkingLevel.MINIMAL)

    @patch("ai.GEMINI_MAX_RETRIES", 3)
    @patch("ai.GEMINI_FALLBACK_MODELS", ("gemini-3.5-flash-lite",))
    @patch("ai.GEMINI_MODEL", "gemini-3.5-flash")
    @patch("ai.GEMINI_API_KEY", "test-key")
    @patch("ai.GEMINI_EDITOR_ENABLED", False)
    @patch("ai.GEMINI_PROOFREADER_ENABLED", False)
    def test_quota_error_switches_model_without_repeating_request(self):
        complete = SimpleNamespace(parsed=self._content(), text=None, candidates=[])
        generate = Mock(side_effect=[RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"), complete])
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))

        with patch("ai.genai.Client", return_value=client):
            generate_reel_content("Spacer", "Jogi przy drzewie", [])

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(
            [call.kwargs["model"] for call in generate.call_args_list],
            ["gemini-3.5-flash", "gemini-3.5-flash-lite"],
        )

    @patch("ai.GEMINI_MAX_RETRIES", 1)
    @patch("ai.GEMINI_FALLBACK_MODELS", ())
    @patch("ai.GEMINI_MODEL", "gemini-3.5-flash")
    @patch("ai.GEMINI_API_KEY", "test-key")
    @patch("ai.GEMINI_EDITOR_ENABLED", True)
    @patch("ai.GEMINI_PROOFREADER_ENABLED", False)
    def test_editor_unifies_perspective_and_removes_personal_hashtags(self):
        draft = ReelContent(
            voiceover="Jogi pierwszy raz odwiedza babcię. Trochę się zestresowałem.",
            headline="Pierwsza wizyta",
            cover_title="U babci",
            caption_body="Pudel Jogi odwiedza babcię. Trochę się zestresowałem.",
            hashtags=["JogiPudel", "pudeltoy", "pieswgosciach"],
            alt_text="Pudel w mieszkaniu.",
            asset_order=[0],
        )
        edited = ReelContent(
            voiceover=(
                "Pierwszy raz odwiedziłem babcię i od progu poczułem lekkie napięcie. "
                "Nowe miejsce bardzo mnie zaciekawiło, lecz spokojnie obserwowałem "
                "otoczenie i poznawałem dom krok po kroku."
            ),
            headline="Pierwsza wizyta",
            cover_title="U babci",
            caption_body=(
                "Pierwszy raz odwiedziłem babcię i od razu zacząłem poznawać nowe miejsce. "
                "Nowe miejsce trochę mnie zestresowało, ale spokojnie zacząłem poznawać "
                "każdy kąt. Drugi pies zachował dystans, więc również obserwowałem "
                "sytuację z daleka i niczego nie przyspieszałem. Taka wizyta była dla "
                "mnie zupełnie nowym doświadczeniem. "
                "Jak Wasze psy reagują na pierwszą wizytę w nowym domu?"
            ),
            hashtags=["pudelminiaturowy", "pies", "pieswgosciach", "psiezycie"],
            alt_text="Pudel miniaturowy w mieszkaniu.",
            asset_order=[0],
        )
        generate = Mock(
            side_effect=[
                SimpleNamespace(parsed=draft, text=None, candidates=[]),
                SimpleNamespace(parsed=edited, text=None, candidates=[]),
            ]
        )
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))

        with patch("ai.genai.Client", return_value=client):
            result = generate_reel_content("Pierwsza wizyta", "Wizyta u babci", [])

        self.assertEqual(generate.call_count, 2)
        self.assertTrue(result["caption_body"].startswith("Pierwszy raz odwiedziłem"))
        self.assertTrue(all("jogi" not in tag.lower() for tag in result["hashtags"]))
        self.assertTrue(all("toy" not in tag.lower() for tag in result["hashtags"]))
        editor_config = generate.call_args_list[1].kwargs["config"]
        self.assertEqual(editor_config.thinking_config.thinking_level, types.ThinkingLevel.LOW)

    @patch("ai.GEMINI_MAX_RETRIES", 1)
    @patch("ai.GEMINI_FALLBACK_MODELS", ())
    @patch("ai.GEMINI_MODEL", "gemini-3.5-flash")
    @patch("ai.GEMINI_API_KEY", "test-key")
    @patch("ai.GEMINI_EDITOR_ENABLED", False)
    @patch("ai.GEMINI_PROOFREADER_ENABLED", True)
    def test_proofreader_keeps_better_text_without_extra_api_round(self):
        valid_voiceover = (
            "Pierwszy raz odwiedziłem babcię i spokojnie wszedłem do nowego domu. "
            "Sisi nie zaakceptowała mojej obecności, a ja później nasikałem na dywan. "
            "Tak wyglądał mój debiut."
        )
        valid_caption = (
            "Pierwszy raz odwiedziłem babcię, ale domowa rezydentka Sisi nie "
            "zaakceptowała mojej obecności. Później nasikałem na dywan i właśnie tak "
            "zapisała się moja pierwsza wizyta w gościach. Bez dodatkowych sukcesów, "
            "bez wielkiego finału — tylko ja, Sisi i ten nieszczęsny dywan. Czy Wasz "
            "pies też kiedyś zaliczył podobną wpadkę poza domem?"
        )
        valid = ReelContent(
            voiceover=valid_voiceover,
            headline="Pierwsza wizyta",
            cover_title="Wizyta u babci",
            caption_body=valid_caption,
            hashtags=["pies", "pieswgosciach", "psiezycie"],
            alt_text="Pies w domu.",
            asset_order=[],
        )
        shortened = ReelContent(
            voiceover="Pierwszy raz odwiedziłem babcię.",
            headline="Pierwsza wizyta",
            cover_title="Wizyta u babci",
            caption_body="Pierwszy raz odwiedziłem babcię. Czy Wasz pies też tak miał?",
            hashtags=["pies", "pieswgosciach"],
            alt_text="Pies w domu.",
            asset_order=[],
        )
        generate = Mock(
            side_effect=[
                SimpleNamespace(parsed=valid, text=None, candidates=[]),
                SimpleNamespace(parsed=shortened, text=None, candidates=[]),
            ]
        )
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))

        with patch("ai.genai.Client", return_value=client):
            result = generate_reel_content(
                "Pierwsza wizyta u babci",
                "Sisi nie toleruje Jogiego, a ten nasikał na dywan.",
                [],
            )

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result["caption_body"], valid_caption)
        proofreader_config = generate.call_args_list[1].kwargs["config"]
        self.assertEqual(
            proofreader_config.thinking_config.thinking_level,
            types.ThinkingLevel.MEDIUM,
        )


if __name__ == "__main__":
    unittest.main()
