import unittest

from reel_formats import choose_reel_format, eligible_reel_formats, normalise_reel_format


class ReelFormatTests(unittest.TestCase):
    def test_single_medium_disables_sequence_formats(self):
        self.assertEqual(eligible_reel_formats(1), ("punchline", "diary_mood"))
        self.assertEqual(choose_reel_format("comparison", 1), "punchline")

    def test_multiple_media_accept_every_format(self):
        for format_id in ("punchline", "mini_story", "comparison", "diary_mood"):
            self.assertEqual(choose_reel_format(format_id, 3), format_id)

    def test_third_identical_format_is_changed(self):
        self.assertEqual(
            choose_reel_format("punchline", 3, ["punchline", "punchline"]),
            "mini_story",
        )
        self.assertEqual(
            choose_reel_format("diary_mood", 1, ["diary_mood", "diary_mood"]),
            "punchline",
        )

    def test_unknown_format_has_stable_default(self):
        self.assertEqual(normalise_reel_format("unknown"), "punchline")


if __name__ == "__main__":
    unittest.main()
