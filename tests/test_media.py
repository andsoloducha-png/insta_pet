import math
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from media import (
    OUTPUT_DIR,
    TimedWord,
    build_caption_cues,
    extract_drive_id,
    load_visual_asset,
    parse_media_links,
    render_frame,
    render_reel_video,
)

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


class MediaTests(unittest.TestCase):
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

    def test_caption_cues_cover_all_words(self):
        words = [TimedWord(f"słowo{i}", i * 0.3, i * 0.3 + 0.25) for i in range(11)]
        cues = build_caption_cues(words, max_words=4)
        flattened = [word.text for cue in cues for word in cue.words]
        self.assertEqual(flattened, [word.text for word in words])
        self.assertTrue(all(len(cue.words) <= 4 for cue in cues))

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
