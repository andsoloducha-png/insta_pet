import math
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from media import (
    INSTAGRAM_COVER_BRAND_TOP,
    INSTAGRAM_COVER_TITLE_TOP,
    INSTAGRAM_HEADLINE_TOP,
    OUTPUT_DIR,
    TimedWord,
    VisualAsset,
    _apply_voice_effect,
    _cover_title_lines,
    _fit_wrapped_text,
    _font,
    _generate_gemini_pcm,
    _request_elevenlabs_audio,
    _retime_words_after_cuts,
    _silence_cuts,
    _scale_words,
    _timed_words_from_alignment,
    _voice_preset,
    build_caption_cues,
    download_media_links,
    extract_drive_id,
    generate_audio_with_timings,
    load_visual_asset,
    limit_caption_cues,
    parse_media_links,
    render_frame,
    render_reel_video,
    resolve_output_path,
    save_instagram_caption,
)

try:
    from moviepy import AudioFileClip, VideoFileClip
except ImportError:
    from moviepy.editor import AudioFileClip, VideoFileClip


class MediaTests(unittest.TestCase):
    def test_instagram_title_positions_match_mobile_and_grid_views(self):
        profile_crop_top = (1920 - 1080 * 4 // 3) // 2
        feed_crop_top = (1920 - 1080 * 5 // 4) // 2
        self.assertLess(INSTAGRAM_HEADLINE_TOP, feed_crop_top)
        self.assertGreater(INSTAGRAM_COVER_BRAND_TOP, profile_crop_top)
        self.assertGreater(INSTAGRAM_COVER_TITLE_TOP, feed_crop_top)

    def test_long_headline_is_fitted_without_losing_words(self):
        draw = ImageDraw.Draw(Image.new("RGB", (1080, 1920), "white"))
        _font_value, lines, _size = _fit_wrapped_text(
            draw,
            "JAK OWINĄĆ SOBIE LUDZI WOKÓŁ ŁAPY",
            max_width=880,
            max_lines=2,
            start_font_size=76,
            min_font_size=52,
        )
        self.assertLessEqual(len(lines), 2)
        self.assertEqual(
            [word for line in lines for word in line],
            "JAK OWINĄĆ SOBIE LUDZI WOKÓŁ ŁAPY".split(),
        )

    def test_cover_title_breaks_before_short_polish_preposition(self):
        draw = ImageDraw.Draw(Image.new("RGB", (1080, 1920), "white"))
        lines = _cover_title_lines(
            draw,
            "PIERWSZA WIZYTA U BABCI",
            _font(78),
            max_width=840,
        )
        self.assertEqual(lines, [["PIERWSZA", "WIZYTA"], ["U", "BABCI"]])

    def test_parses_multiple_plain_and_markdown_links(self):
        value = """
        [zdjęcie](https://drive.google.com/file/d/1234567890123456789012345/view)
        https://example.com/jogi.gif, https://example.com/jogi-2.jpg
        """
        self.assertEqual(
            parse_media_links(value),
            [
                "https://drive.google.com/file/d/1234567890123456789012345/view",
                "https://example.com/jogi.gif",
                "https://example.com/jogi-2.jpg",
            ],
        )

    def test_extracts_drive_id_from_common_urls(self):
        expected = "1234567890123456789012345"
        self.assertEqual(extract_drive_id(f"https://drive.google.com/file/d/{expected}/view"), expected)
        self.assertEqual(extract_drive_id(f"https://drive.google.com/open?id={expected}"), expected)

    def test_downloaded_media_are_named_inside_reel_folder(self):
        with patch("media.download_media", side_effect=lambda _link, stem: stem) as download:
            paths = download_media_links(
                "https://example.com/a.jpg https://example.com/b.gif",
                row_idx=12,
            )
        self.assertEqual(paths, ["wiersz_12/media_1", "wiersz_12/media_2"])
        self.assertEqual(
            [call.args[1] for call in download.call_args_list],
            ["wiersz_12/media_1", "wiersz_12/media_2"],
        )

    def test_loads_gif_and_always_renders_fixed_size(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gif_path = OUTPUT_DIR / "_test_sample.gif"
        try:
            frames = [
                Image.new("RGB", (320, 180), (220, 40, 40)),
                Image.new("RGB", (320, 180), (40, 180, 80)),
                Image.new("RGB", (320, 180), (50, 80, 220)),
            ]
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=[100, 150, 200],
                loop=0,
            )
            asset = load_visual_asset(gif_path)
            image = render_frame([asset], duration=2.0, current_time=0.8, headline="TEST")
        finally:
            gif_path.unlink(missing_ok=True)

        self.assertTrue(asset.animated)
        self.assertEqual(len(asset.frames), 3)
        self.assertEqual(image.shape, (1920, 1080, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_four_formats_render_distinct_frames(self):
        x = np.linspace(0, 255, 1080, dtype=np.uint8)
        first_array = np.zeros((1920, 1080, 3), dtype=np.uint8)
        first_array[:, :, 0] = x
        first_array[:, :, 1] = 50
        second_array = np.zeros((1920, 1080, 3), dtype=np.uint8)
        second_array[:, :, 1] = x
        second_array[:, :, 2] = 190
        assets = [
            VisualAsset("first", [Image.fromarray(first_array)], [1.0]),
            VisualAsset("second", [Image.fromarray(second_array)], [1.0]),
        ]

        frames = {
            format_id: render_frame(
                assets,
                duration=4.0,
                current_time=1.78,
                format_id=format_id,
            )
            for format_id in ("punchline", "mini_story", "comparison", "diary_mood")
        }

        self.assertTrue(all(frame.shape == (1920, 1080, 3) for frame in frames.values()))
        self.assertFalse(np.array_equal(frames["punchline"], frames["mini_story"]))
        self.assertFalse(np.array_equal(frames["mini_story"], frames["comparison"]))
        self.assertFalse(np.array_equal(frames["mini_story"], frames["diary_mood"]))

    def test_nested_output_path_is_created_and_cannot_escape_output(self):
        nested = OUTPUT_DIR / "_test_nested" / "opis.txt"
        try:
            saved = save_instagram_caption("Test", "_test_nested/opis.txt")
            self.assertEqual(Path(saved), nested)
            self.assertEqual(nested.read_text(encoding="utf-8"), "Test\n")
            with self.assertRaises(ValueError):
                resolve_output_path("../poza-output.txt")
        finally:
            nested.unlink(missing_ok=True)
            nested.parent.rmdir()

    def test_caption_cues_cover_all_words(self):
        words = [TimedWord(f"słowo{i}", i * 0.3, i * 0.3 + 0.25) for i in range(11)]
        cues = build_caption_cues(words, max_words=4)
        flattened = [word.text for cue in cues for word in cue.words]
        self.assertEqual(flattened, [word.text for word in words])
        self.assertTrue(all(len(cue.words) <= 4 for cue in cues))

    def test_caption_cues_end_before_signature_audio(self):
        words = [TimedWord("Puenta.", 0.5, 1.0)]
        cues = limit_caption_cues(build_caption_cues(words), maximum_end=1.0)
        self.assertEqual(cues[-1].end, 1.0)

    def test_short_video_is_valid_and_compact(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        image_path = OUTPUT_DIR / "_test_portrait.jpg"
        audio_path = OUTPUT_DIR / "_test_tone.wav"
        output_name = "_test_short_reel.mp4"
        output_path = OUTPUT_DIR / output_name
        try:
            Image.new("RGB", (720, 1280), (90, 150, 210)).save(image_path, quality=90)
            self._write_tone(audio_path, duration=1.1)
            rendered = render_reel_video(
                [str(image_path)],
                str(audio_path),
                output_name=output_name,
                headline="KRÓTKI TEST",
                logger=None,
            )
            with VideoFileClip(rendered) as video:
                frame = video.get_frame(min(0.7, video.duration / 2))
                self.assertEqual(list(video.size), [1080, 1920])
                self.assertAlmostEqual(video.fps, 30, delta=0.1)
                self.assertGreater(video.duration, 1.0)
            self.assertEqual(frame.shape, (1920, 1080, 3))
            self.assertLess(output_path.stat().st_size, 5 * 1024 * 1024)
        finally:
            image_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    @patch("media.time.sleep", return_value=None)
    def test_tts_retries_and_uses_fallback_voice(self, _sleep):
        output_name = "_test_tts_fallback.mp3"
        output_path = OUTPUT_DIR / output_name
        attempted_voices: list[str] = []

        async def fake_stream(_text, path, voice, _rate, _pitch):
            attempted_voices.append(voice)
            if voice == "pl-PL-UsunietyNeural":
                raise RuntimeError("No audio was received")
            path.write_bytes(b"fake-mp3-data")
            return [TimedWord("Test", 0.0, 0.5)]

        try:
            with (
                patch("media.TTS_VOICE", "pl-PL-UsunietyNeural"),
                patch("media.TTS_FALLBACK_VOICES", ("pl-PL-ZofiaNeural",)),
                patch("media.TTS_MAX_RETRIES", 2),
                patch("media.TTS_EFFECTS_ENABLED", False),
                patch("media._stream_edge_audio", side_effect=fake_stream),
            ):
                result = generate_audio_with_timings("Test", output_name, provider="edge")
        finally:
            output_path.unlink(missing_ok=True)

        self.assertEqual(
            attempted_voices,
            ["pl-PL-UsunietyNeural", "pl-PL-UsunietyNeural", "pl-PL-ZofiaNeural"],
        )
        self.assertEqual(result.voice, "pl-PL-ZofiaNeural")
        self.assertEqual(result.preset, "jogi_playful_soft")
        self.assertEqual(result.provider, "edge")
        self.assertEqual(result.words[0].text, "Test")

    def test_gemini_provider_appends_signature_after_captioned_narration(self):
        narration_name = "_test_gemini_audio.mp3"
        output_path = OUTPUT_DIR / "_test_gemini_audio.wav"
        laugh_path = OUTPUT_DIR / "_test_signature.wav"

        def fake_gemini(_text, path):
            self._write_tone(path, duration=0.6)

        try:
            self._write_tone(laugh_path, duration=0.25)
            with (
                patch("media._generate_gemini_pcm", side_effect=fake_gemini),
                patch("media.TTS_SIGNATURE_LAUGH_ENABLED", True),
                patch("media.TTS_SIGNATURE_LAUGH_FILE", str(laugh_path)),
            ):
                result = generate_audio_with_timings(
                    "Cześć, tu Jogi!",
                    narration_name,
                    provider="gemini",
                )
            with AudioFileClip(result.path) as audio:
                final_duration = audio.duration
        finally:
            output_path.unlink(missing_ok=True)
            laugh_path.unlink(missing_ok=True)
            (OUTPUT_DIR / "_test_gemini_audio.narration.wav").unlink(missing_ok=True)

        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.voice, "Achird")
        self.assertEqual(result.preset, "achird_warm_signature")
        self.assertGreater(final_duration, 0.75)
        self.assertLessEqual(result.cues[-1].end, 0.6)

    def test_elevenlabs_alignment_is_grouped_into_caption_words(self):
        text = "Cześć, tu Jogi!"
        characters = list(text)
        alignment = {
            "characters": characters,
            "character_start_times_seconds": [
                index * 0.05 for index in range(len(characters))
            ],
            "character_end_times_seconds": [
                (index + 1) * 0.05 for index in range(len(characters))
            ],
        }

        words = _timed_words_from_alignment(alignment)

        self.assertEqual([word.text for word in words], ["Cześć,", "tu", "Jogi!"])
        self.assertAlmostEqual(words[0].start, 0.0)
        self.assertAlmostEqual(words[-1].end, len(characters) * 0.05)

    def test_elevenlabs_request_uses_clone_settings_and_exact_alignment(self):
        output_path = OUTPUT_DIR / "_test_elevenlabs_response.mp3"
        response = Mock(ok=True)
        response.json.return_value = {
            "audio_base64": "ZmFrZS1tcDM=",
            "normalized_alignment": {
                "characters": ["J", "o", "g", "i"],
                "character_start_times_seconds": [0.0, 0.05, 0.10, 0.15],
                "character_end_times_seconds": [0.05, 0.10, 0.15, 0.20],
            },
        }

        try:
            with (
                patch("media.requests.post", return_value=response) as post,
                patch("media.ELEVENLABS_API_KEY", "test-api-key"),
                patch("media.ELEVENLABS_VOICE_ID", "test voice/id"),
                patch("media.ELEVENLABS_MODEL", "eleven_multilingual_v2"),
                patch("media.ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128"),
                patch("media.ELEVENLABS_SPEED", 0.96),
            ):
                words = _request_elevenlabs_audio("Jogi", output_path)
        finally:
            output_path.unlink(missing_ok=True)

        self.assertEqual([word.text for word in words], ["Jogi"])
        self.assertEqual(post.call_args.kwargs["params"]["output_format"], "mp3_44100_128")
        self.assertEqual(
            post.call_args.kwargs["json"]["model_id"],
            "eleven_multilingual_v2",
        )
        self.assertEqual(post.call_args.kwargs["json"]["language_code"], "pl")
        self.assertEqual(post.call_args.kwargs["json"]["voice_settings"]["speed"], 0.96)
        self.assertIn("test%20voice%2Fid", post.call_args.args[0])

    def test_elevenlabs_provider_preserves_timings_and_appends_signature(self):
        output_name = "_test_elevenlabs_audio.mp3"
        output_path = OUTPUT_DIR / "_test_elevenlabs_audio.wav"
        narration_path = OUTPUT_DIR / "_test_elevenlabs_audio.narration.mp3"
        laugh_path = OUTPUT_DIR / "_test_elevenlabs_signature.wav"
        expected_words = [
            TimedWord("Cześć,", 0.05, 0.25),
            TimedWord("Jogi!", 0.30, 0.58),
        ]

        def fake_elevenlabs(_text, path):
            self._write_tone(path, duration=0.6)
            return expected_words

        try:
            self._write_tone(laugh_path, duration=0.25)
            with (
                patch("media._request_elevenlabs_audio", side_effect=fake_elevenlabs),
                patch("media.ELEVENLABS_MAX_RETRIES", 2),
                patch("media.ELEVENLABS_MODEL", "eleven_multilingual_v2"),
                patch("media.ELEVENLABS_VOICE_ID", "test-voice-id"),
                patch("media.TTS_SIGNATURE_LAUGH_ENABLED", True),
                patch("media.TTS_SIGNATURE_LAUGH_FILE", str(laugh_path)),
            ):
                result = generate_audio_with_timings(
                    "Cześć, Jogi!",
                    output_name,
                    provider="elevenlabs",
                )
            with AudioFileClip(result.path) as audio:
                final_duration = audio.duration
        finally:
            output_path.unlink(missing_ok=True)
            narration_path.unlink(missing_ok=True)
            laugh_path.unlink(missing_ok=True)

        self.assertEqual(result.provider, "elevenlabs")
        self.assertEqual(result.model, "eleven_multilingual_v2")
        self.assertEqual(result.voice, "test-voice-id")
        self.assertEqual(result.preset, "elevenlabs_clone_signature")
        self.assertEqual(list(result.words), expected_words)
        self.assertGreater(final_duration, 0.75)
        self.assertLessEqual(result.cues[-1].end, 0.6)

    def test_gemini_tts_quota_switches_to_fallback_model_immediately(self):
        output_path = OUTPUT_DIR / "_test_gemini_fallback.wav"
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(inline_data=SimpleNamespace(data=b"\x00\x00" * 240))]
                    )
                )
            ]
        )
        generate = Mock(
            side_effect=[RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"), response]
        )
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        try:
            with (
                patch("media.genai.Client", return_value=client),
                patch("media.GEMINI_API_KEY", "test-key"),
                patch("media.GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts"),
                patch("media.GEMINI_TTS_FALLBACK_MODELS", ("gemini-3.1-flash-tts-preview",)),
                patch("media.GEMINI_TTS_MAX_RETRIES", 3),
            ):
                selected_model = _generate_gemini_pcm("Test", output_path)
        finally:
            output_path.unlink(missing_ok=True)

        self.assertEqual(selected_model, "gemini-3.1-flash-tts-preview")
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(
            [call.kwargs["model"] for call in generate.call_args_list],
            ["gemini-2.5-flash-preview-tts", "gemini-3.1-flash-tts-preview"],
        )
        self.assertTrue(
            all(
                call.kwargs["config"]
                .speech_config.voice_config.prebuilt_voice_config.voice_name
                == "Achird"
                for call in generate.call_args_list
            )
        )

    def test_voice_presets_are_distinct_and_invalid_name_is_rejected(self):
        soft = _voice_preset("jogi_playful_soft")
        regular = _voice_preset("jogi_playful")
        wild = _voice_preset("jogi_playful_wild")
        self.assertNotEqual((soft.rate, soft.pitch), (regular.rate, regular.pitch))
        self.assertNotEqual((regular.rate, regular.pitch), (wild.rate, wild.pitch))
        with self.assertRaisesRegex(ValueError, "Nieznany preset"):
            _voice_preset("capcut_trickster")

    def test_scales_word_timings_after_audio_processing(self):
        words = [TimedWord("Jogi", 0.25, 0.75)]
        scaled = _scale_words(words, 1.1)
        self.assertAlmostEqual(scaled[0].start, 0.275)
        self.assertAlmostEqual(scaled[0].end, 0.825)

    def test_compacts_only_excess_part_of_long_pause(self):
        words = [TimedWord("A", 0.2, 0.5), TimedWord("B", 2.0, 2.3)]
        cuts = _silence_cuts([(0.0, 0.1), (0.5, 2.0), (2.3, 2.5)], 0.4, 2.5)
        self.assertEqual(cuts, [(0.7, 1.8)])
        retimed = _retime_words_after_cuts(words, cuts)
        self.assertAlmostEqual(retimed[1].start - retimed[0].end, 0.4)

    def test_each_voice_preset_produces_valid_audio(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        source_path = OUTPUT_DIR / "_test_voice_source.wav"
        output_paths = []
        try:
            self._write_tone(source_path, duration=0.5)
            for preset_name in (
                "jogi_playful_soft",
                "jogi_playful",
                "jogi_playful_wild",
                "jogi_urwis",
            ):
                output_path = OUTPUT_DIR / f"_test_{preset_name}.mp3"
                output_paths.append(output_path)
                factor = _apply_voice_effect(
                    source_path,
                    output_path,
                    _voice_preset(preset_name).audio_filter,
                )
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 1000)
                self.assertAlmostEqual(factor, 1.0, delta=0.08)
        finally:
            source_path.unlink(missing_ok=True)
            for output_path in output_paths:
                output_path.unlink(missing_ok=True)

    @staticmethod
    def _write_tone(path: Path, duration: float, sample_rate: int = 48_000):
        sample_count = int(duration * sample_rate)
        samples = bytearray()
        for index in range(sample_count):
            value = int(6000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            samples.extend(value.to_bytes(2, byteorder="little", signed=True))
        with wave.open(str(path), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(sample_rate)
            file.writeframes(bytes(samples))


if __name__ == "__main__":
    unittest.main()
