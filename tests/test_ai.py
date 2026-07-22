import unittest

from ai import ReelContent, _normalise_content


class ReelContentTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
