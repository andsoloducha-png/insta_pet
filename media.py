import asyncio
import io
import json
import logging
import math
import mimetypes
import os
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import edge_tts
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from config import (
    BASE_DIR,
    DOG_NAME,
    TTS_FALLBACK_VOICES,
    TTS_MAX_RETRIES,
    TTS_PITCH,
    TTS_RATE,
    TTS_VOICE,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)

try:
    from moviepy import AudioFileClip, VideoClip
except ImportError:  # MoviePy 1.x
    from moviepy.editor import AudioFileClip, VideoClip


LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = BASE_DIR / "output"
MAX_DOWNLOAD_BYTES = 60 * 1024 * 1024
MAX_GIF_FRAMES = 450


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class CaptionCue:
    start: float
    end: float
    words: tuple[TimedWord, ...]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


@dataclass(frozen=True)
class AudioResult:
    path: str
    words: tuple[TimedWord, ...]
    cues: tuple[CaptionCue, ...]
    voice: str = ""


@dataclass
class VisualAsset:
    path: str
    frames: list[Image.Image]
    frame_durations: list[float]
    animated: bool = False

    @property
    def animation_duration(self) -> float:
        return sum(self.frame_durations)

    def frame_at(self, local_time: float) -> Image.Image:
        if len(self.frames) == 1:
            return self.frames[0]
        duration = self.animation_duration
        position = local_time % duration if duration > 0 else 0
        elapsed = 0.0
        for frame, frame_duration in zip(self.frames, self.frame_durations):
            elapsed += frame_duration
            if position < elapsed:
                return frame
        return self.frames[-1]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def parse_media_links(value: object) -> list[str]:
    """Wyciąga wszystkie linki HTTP(S) z tekstu, także z Markdowna."""
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []

    markdown_urls = re.findall(r"\[[^\]]*\]\((https?://[^)]+)\)", text, flags=re.I)
    raw_urls = re.findall(r"https?://[^\s<>\[\]]+", text, flags=re.I)
    links: list[str] = []
    for url in [*markdown_urls, *raw_urls]:
        clean = unquote(url).strip().rstrip(".,;!?)\"'")
        if clean and clean not in links:
            links.append(clean)
    return links


def extract_drive_id(link: str) -> str:
    """Wyciąga identyfikator pliku z typowych linków Google Drive."""
    if not link:
        return ""
    clean = str(link).strip()
    parsed = urlparse(clean)
    query_id = parse_qs(parsed.query).get("id", [])
    if query_id and re.fullmatch(r"[A-Za-z0-9_-]{20,}", query_id[0]):
        return query_id[0]

    patterns = (
        r"/file/d/([A-Za-z0-9_-]{20,})",
        r"/d/([A-Za-z0-9_-]{20,})",
        r"/open/([A-Za-z0-9_-]{20,})",
    )
    for pattern in patterns:
        match = re.search(pattern, clean)
        if match:
            return match.group(1)

    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", clean):
        return clean
    return ""


def _download_url(link: str) -> str:
    if "drive.google.com" in link or "docs.google.com" in link:
        file_id = extract_drive_id(link)
        if not file_id:
            raise ValueError(f"Nie rozpoznano identyfikatora Google Drive w linku: {link}")
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    return link


def _extension_from_image(data: bytes, content_type: str) -> str:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            image_format = (image.format or "").upper()
    except Exception as exc:
        raise ValueError("Pobrany plik nie jest obsługiwanym obrazem ani GIF-em.") from exc

    format_extensions = {
        "JPEG": ".jpg",
        "MPO": ".jpg",
        "PNG": ".png",
        "GIF": ".gif",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "TIFF": ".tif",
    }
    extension = format_extensions.get(image_format)
    if extension:
        return extension
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
    return guessed or ".img"


def download_media(link: str, output_stem: str) -> str:
    """Pobiera pojedyncze medium i zachowuje jego prawdziwy typ pliku."""
    local_candidate = Path(link)
    if local_candidate.exists():
        return str(local_candidate.resolve())

    ensure_output_dir()
    url = _download_url(link)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; JogiReels/1.0)"}
    try:
        response = requests.get(url, headers=headers, timeout=(10, 60), stream=True)
        response.raise_for_status()
        buffer = io.BytesIO()
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            if buffer.tell() + len(chunk) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Plik przekracza limit 60 MB.")
            buffer.write(chunk)
    except requests.RequestException as exc:
        raise ValueError(f"Nie udało się pobrać medium: {link}") from exc

    data = buffer.getvalue()
    extension = _extension_from_image(data, response.headers.get("Content-Type", ""))
    output_path = OUTPUT_DIR / f"{output_stem}{extension}"
    with output_path.open("wb") as file:
        file.write(data)
    return str(output_path)


def download_media_links(value: object, row_idx: int) -> list[str]:
    links = parse_media_links(value)
    if not links:
        raise ValueError("Komórka Link nie zawiera żadnego poprawnego adresu HTTP(S).")
    paths = []
    for index, link in enumerate(links, start=1):
        LOGGER.info("Pobieranie medium %s/%s", index, len(links))
        paths.append(download_media(link, f"media_{row_idx}_{index}"))
    return paths


def download_drive_image(link: str, output_name: str = "temp_image.jpg") -> str:
    """Wstecznie zgodny wrapper dla starszego kodu."""
    stem = Path(output_name).stem
    return download_media(link, stem)


def _cover_crop(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(target_w, round(image.width * scale)), max(target_h, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - target_w) // 2)
    top = max(0, (resized.height - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


def _normalise_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    target_w, target_h = VIDEO_WIDTH, VIDEO_HEIGHT
    ratio_delta = abs(math.log((image.width / image.height) / (target_w / target_h)))

    if ratio_delta <= 0.38:
        canvas = _cover_crop(image, target_w, target_h)
    else:
        background = _cover_crop(image, target_w, target_h)
        background = background.filter(ImageFilter.GaussianBlur(radius=32))
        background = ImageEnhance.Brightness(background).enhance(0.62)
        scale = min((target_w * 0.94) / image.width, (target_h * 0.94) / image.height)
        foreground = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas = background
        left = (target_w - foreground.width) // 2
        top = (target_h - foreground.height) // 2
        canvas.paste(foreground, (left, top))

    canvas = ImageEnhance.Contrast(canvas).enhance(1.025)
    canvas = ImageEnhance.Color(canvas).enhance(1.02)
    return ImageEnhance.Sharpness(canvas).enhance(1.06)


def load_visual_asset(path: str | Path) -> VisualAsset:
    path = str(Path(path).resolve())
    frames: list[Image.Image] = []
    durations: list[float] = []
    with Image.open(path) as image:
        animated = bool(getattr(image, "is_animated", False)) and (
            (image.format or "").upper() in {"GIF", "WEBP"}
        )
        frame_count = min(getattr(image, "n_frames", 1), MAX_GIF_FRAMES) if animated else 1
        for index in range(frame_count):
            if animated:
                image.seek(index)
            frame = _normalise_image(image.copy())
            duration_ms = image.info.get("duration", 100) if animated else 1000
            frames.append(frame)
            durations.append(min(1.0, max(0.04, float(duration_ms) / 1000.0)))
    return VisualAsset(path=path, frames=frames, frame_durations=durations, animated=animated)


def load_visual_assets(paths: list[str]) -> list[VisualAsset]:
    if not paths:
        raise ValueError("Brak mediów do zbudowania rolki.")
    return [load_visual_asset(path) for path in paths]


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf"),
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeuib.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _wrap_words(draw: ImageDraw.ImageDraw, words: list[str], font, max_width: int) -> list[list[str]]:
    lines: list[list[str]] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(current)
    return lines


def _cover_title_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[list[str]]:
    words = text.split()
    # Krótkie polskie przyimki naturalnie rozpoczynają drugą linię, np. „U BABCI”.
    for index in range(1, len(words)):
        if words[index].casefold() not in {"u", "w", "na", "do", "z", "bez", "o"}:
            continue
        lines = [words[:index], words[index:]]
        if all(draw.textlength(" ".join(line), font=font) <= max_width for line in lines):
            return lines
    return _wrap_words(draw, words, font, max_width)


def _ken_burns(image: Image.Image, progress: float, segment_index: int, animated: bool) -> Image.Image:
    progress = min(1.0, max(0.0, progress))
    smooth = progress * progress * (3 - 2 * progress)
    if segment_index % 2:
        smooth = 1.0 - smooth
    strength = 0.026 if animated else 0.058
    scale = 1.0 + strength * smooth
    width = max(VIDEO_WIDTH, round(VIDEO_WIDTH * scale))
    height = max(VIDEO_HEIGHT, round(VIDEO_HEIGHT * scale))
    enlarged = image.resize((width, height), Image.Resampling.LANCZOS)
    extra_x = width - VIDEO_WIDTH
    extra_y = height - VIDEO_HEIGHT
    x_bias = 0.35 if segment_index % 3 == 0 else 0.65
    y_bias = 0.42 if segment_index % 2 == 0 else 0.58
    left = min(extra_x, max(0, round(extra_x * x_bias)))
    top = min(extra_y, max(0, round(extra_y * y_bias)))
    return enlarged.crop((left, top, left + VIDEO_WIDTH, top + VIDEO_HEIGHT))


def _draw_headline(image: Image.Image, text: str) -> None:
    if not text:
        return
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(76)
    lines = _wrap_words(draw, text.split(), font, 880)[:2]
    line_height = 88
    total_height = len(lines) * line_height
    top = 205
    widths = [draw.textlength(" ".join(line), font=font) for line in lines]
    box_width = max(widths, default=0) + 80
    left = (VIDEO_WIDTH - box_width) / 2
    draw.rounded_rectangle(
        (left, top - 30, left + box_width, top + total_height + 20),
        radius=32,
        fill=(8, 12, 18, 205),
    )
    draw.rounded_rectangle((left, top - 30, left + 14, top + total_height + 20), radius=7, fill=(46, 111, 255, 255))
    for line_index, line in enumerate(lines):
        line_text = " ".join(line)
        width = draw.textlength(line_text, font=font)
        draw.text(
            ((VIDEO_WIDTH - width) / 2, top + line_index * line_height),
            line_text,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill=(0, 0, 0, 190),
        )
    image.alpha_composite(overlay)


def _draw_caption(image: Image.Image, cue: CaptionCue, current_time: float) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(58)
    words = [word.text for word in cue.words]
    lines = _wrap_words(draw, words, font, 900)[:2]
    line_height = 72
    start_y = 1370 - (len(lines) - 1) * line_height // 2
    line_widths = [draw.textlength(" ".join(line), font=font) for line in lines]
    box_width = max(line_widths, default=0) + 72
    box_height = len(lines) * line_height + 38
    box_left = (VIDEO_WIDTH - box_width) / 2
    draw.rounded_rectangle(
        (box_left, start_y - 22, box_left + box_width, start_y - 22 + box_height),
        radius=28,
        fill=(0, 0, 0, 180),
    )

    timed_by_text: dict[str, list[TimedWord]] = {}
    for timed_word in cue.words:
        timed_by_text.setdefault(timed_word.text, []).append(timed_word)

    for line_index, line in enumerate(lines):
        line_width = draw.textlength(" ".join(line), font=font)
        x = (VIDEO_WIDTH - line_width) / 2
        y = start_y + line_index * line_height
        for index, word_text in enumerate(line):
            candidates = timed_by_text.get(word_text, [])
            active = any(word.start <= current_time <= word.end for word in candidates)
            color = (255, 214, 61, 255) if active else (255, 255, 255, 255)
            draw.text(
                (x, y),
                word_text,
                font=font,
                fill=color,
                stroke_width=3,
                stroke_fill=(0, 0, 0, 220),
            )
            x += draw.textlength(word_text, font=font)
            if index < len(line) - 1:
                x += draw.textlength(" ", font=font)
    image.alpha_composite(overlay)


def build_caption_cues(words: list[TimedWord], max_words: int = 5) -> list[CaptionCue]:
    if not words:
        return []
    groups: list[list[TimedWord]] = []
    current: list[TimedWord] = []
    for word in words:
        current.append(word)
        duration = current[-1].end - current[0].start
        sentence_end = word.text.rstrip().endswith((".", "!", "?", ":", ";"))
        if len(current) >= max_words or duration >= 1.65 or sentence_end:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    cues: list[CaptionCue] = []
    for index, group in enumerate(groups):
        start = max(0.0, group[0].start - 0.06)
        end = group[-1].end + 0.18
        if index + 1 < len(groups):
            end = min(end, groups[index + 1][0].start + 0.03)
        cues.append(CaptionCue(start=start, end=end, words=tuple(group)))
    return cues


def _approximate_words(text: str, duration: float) -> list[TimedWord]:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return []
    weights = [max(1, len(token.strip(".,!?;:"))) for token in tokens]
    total = sum(weights)
    cursor = 0.0
    words: list[TimedWord] = []
    for token, weight in zip(tokens, weights):
        word_duration = duration * weight / total
        words.append(TimedWord(token, cursor, min(duration, cursor + word_duration)))
        cursor += word_duration
    return words


async def _stream_edge_audio(text: str, output_path: Path, voice: str) -> list[TimedWord]:
    communicate = edge_tts.Communicate(text, voice, rate=TTS_RATE, pitch=TTS_PITCH)
    words: list[TimedWord] = []
    with output_path.open("wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = float(chunk["offset"]) / 10_000_000
                duration = float(chunk["duration"]) / 10_000_000
                words.append(TimedWord(str(chunk["text"]), start, start + duration))
    return words


def generate_audio_with_timings(text: str, output_name: str = "temp_audio.mp3") -> AudioResult:
    ensure_output_dir()
    output_path = OUTPUT_DIR / output_name
    voices = list(dict.fromkeys(voice for voice in (TTS_VOICE, *TTS_FALLBACK_VOICES) if voice))
    errors: list[str] = []
    selected_voice = ""
    words: list[TimedWord] = []

    for voice in voices:
        for attempt in range(1, max(1, TTS_MAX_RETRIES) + 1):
            output_path.unlink(missing_ok=True)
            try:
                words = asyncio.run(_stream_edge_audio(text, output_path, voice))
                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise RuntimeError("usługa nie zwróciła danych audio")
                selected_voice = voice
                break
            except Exception as exc:
                output_path.unlink(missing_ok=True)
                message = f"{voice}, próba {attempt}/{max(1, TTS_MAX_RETRIES)}: {exc}"
                LOGGER.warning("Edge TTS nie wygenerował nagrania (%s)", message)
                errors.append(message)
                if attempt < max(1, TTS_MAX_RETRIES):
                    time.sleep(attempt)
        if selected_voice:
            break

    if not selected_voice:
        raise RuntimeError(
            "Edge TTS nie wygenerował nagrania żadnym polskim głosem. " + " | ".join(errors)
        )
    if not words:
        with AudioFileClip(str(output_path)) as audio:
            words = _approximate_words(text, audio.duration)
    cues = build_caption_cues(words)
    return AudioResult(str(output_path), tuple(words), tuple(cues), selected_voice)


def generate_audio(text: str, output_name: str = "temp_audio.mp3") -> str:
    """Wstecznie zgodny wrapper zwracający samą ścieżkę."""
    return generate_audio_with_timings(text, output_name).path


def _ordered_assets(assets: list[VisualAsset], asset_order: list[int] | None) -> list[VisualAsset]:
    if not asset_order:
        return assets
    valid: list[int] = []
    for index in asset_order:
        if 0 <= index < len(assets) and index not in valid:
            valid.append(index)
    valid.extend(index for index in range(len(assets)) if index not in valid)
    return [assets[index] for index in valid]


def render_frame(
    assets: list[VisualAsset],
    duration: float,
    current_time: float,
    headline: str = "",
    caption_cues: list[CaptionCue] | tuple[CaptionCue, ...] = (),
    asset_order: list[int] | None = None,
) -> np.ndarray:
    ordered = _ordered_assets(assets, asset_order)
    segment_duration = duration / len(ordered)
    segment_index = min(len(ordered) - 1, int(current_time / max(segment_duration, 0.001)))
    segment_start = segment_index * segment_duration
    local_time = max(0.0, current_time - segment_start)
    progress = local_time / max(segment_duration, 0.001)
    asset = ordered[segment_index]
    image = _ken_burns(asset.frame_at(local_time), progress, segment_index, asset.animated).convert("RGBA")

    headline_duration = min(1.8, max(1.1, duration * 0.2))
    if current_time <= headline_duration:
        _draw_headline(image, headline)
    for cue in caption_cues:
        if cue.start <= current_time <= cue.end:
            _draw_caption(image, cue, current_time)
            break
    return np.asarray(image.convert("RGB"))


def render_reel_video(
    image_paths: str | list[str],
    audio_path: str,
    output_name: str = "final_reel.mp4",
    headline: str = "",
    caption_cues: list[CaptionCue] | tuple[CaptionCue, ...] = (),
    asset_order: list[int] | None = None,
    logger: str | None = "bar",
) -> str:
    ensure_output_dir()
    paths = [image_paths] if isinstance(image_paths, str) else image_paths
    assets = load_visual_assets(paths)
    output_path = OUTPUT_DIR / output_name
    audio = AudioFileClip(audio_path)
    duration = float(audio.duration)
    if duration <= 0:
        audio.close()
        raise ValueError("Nagranie lektora ma niepoprawną długość.")

    def frame_function(t: float) -> np.ndarray:
        return render_frame(assets, duration, float(t), headline, caption_cues, asset_order)

    video = VideoClip(frame_function=frame_function, duration=duration).with_audio(audio)
    try:
        video.write_videofile(
            str(output_path),
            fps=VIDEO_FPS,
            codec="libx264",
            audio_codec="aac",
            audio_fps=48_000,
            audio_bitrate="192k",
            preset="medium",
            pixel_format="yuv420p",
            ffmpeg_params=["-crf", "20", "-movflags", "+faststart"],
            threads=min(4, os.cpu_count() or 1),
            logger=logger,
        )
    finally:
        video.close()
        audio.close()
    return str(output_path)


def _draw_cover_text(image: Image.Image, text: str) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Dwustronnie wygaszony cień nie tworzy widocznej poziomej krawędzi.
    gradient_top = 150
    gradient_peak = 330
    gradient_bottom = 700
    for y in range(gradient_top, gradient_bottom):
        if y <= gradient_peak:
            progress = (y - gradient_top) / (gradient_peak - gradient_top)
        else:
            progress = 1.0 - (y - gradient_peak) / (gradient_bottom - gradient_peak)
        alpha = round(100 * max(0.0, min(1.0, progress)))
        draw.line((0, y, VIDEO_WIDTH, y), fill=(0, 0, 0, alpha))

    brand_font = _font(30)
    brand = f"{DOG_NAME.upper()} • DZIENNIK PSA"
    brand_y = 270
    title_bar_left = 76
    title_bar_right = 88
    title_text_x = 112
    draw.text((title_text_x, brand_y), brand, font=brand_font, fill=(215, 227, 255, 255))

    title_font_size = 78
    while title_font_size >= 56:
        title_font = _font(title_font_size)
        lines = _cover_title_lines(draw, text, title_font, max_width=840)
        if len(lines) <= 2:
            break
        title_font_size -= 2
    else:
        title_font_size = 56
        title_font = _font(title_font_size)
        lines = _wrap_words(draw, text.split(), title_font, 840)[:2]
    start_x = title_text_x
    start_y = 336
    line_height = round(title_font_size * 1.18)
    for index, line in enumerate(lines):
        line_text = " ".join(line)
        draw.text(
            (start_x, start_y + index * line_height),
            line_text,
            font=title_font,
            fill="white",
            stroke_width=4,
            stroke_fill=(0, 0, 0, 210),
        )
    bar_bottom = start_y + max(1, len(lines)) * line_height - 12
    draw.rounded_rectangle(
        (title_bar_left, start_y - 12, title_bar_right, bar_bottom),
        radius=6,
        fill=(46, 111, 255, 255),
    )
    image.alpha_composite(overlay)


def create_cover(
    image_paths: str | list[str],
    title: str,
    output_name: str = "cover.jpg",
    asset_order: list[int] | None = None,
) -> str:
    ensure_output_dir()
    paths = [image_paths] if isinstance(image_paths, str) else image_paths
    assets = load_visual_assets(paths)
    asset = _ordered_assets(assets, asset_order)[0]
    image = asset.frames[0].convert("RGBA")
    _draw_cover_text(image, title)
    output_path = OUTPUT_DIR / output_name
    image.convert("RGB").save(output_path, quality=93, optimize=True)
    return str(output_path)


def prepare_916_image_with_text(image_path: str, text: str, output_name: str = "processed_image.jpg") -> str:
    """Wstecznie zgodny eksport pojedynczej planszy 9:16."""
    ensure_output_dir()
    asset = load_visual_asset(image_path)
    image = asset.frames[0].convert("RGBA")
    _draw_headline(image, text)
    output_path = OUTPUT_DIR / output_name
    image.convert("RGB").save(output_path, quality=93, optimize=True)
    return str(output_path)


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def save_srt(cues: list[CaptionCue] | tuple[CaptionCue, ...], output_name: str) -> str:
    ensure_output_dir()
    output_path = OUTPUT_DIR / output_name
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n{_srt_timestamp(cue.start)} --> {_srt_timestamp(cue.end)}\n{cue.text}\n"
        )
    output_path.write_text("\n".join(blocks), encoding="utf-8")
    return str(output_path)


def save_instagram_caption(caption_text: str, output_name: str = "post_caption.txt") -> str:
    ensure_output_dir()
    output_path = OUTPUT_DIR / output_name
    output_path.write_text(caption_text.strip() + "\n", encoding="utf-8")
    return str(output_path)


def save_manifest(data: dict, output_name: str) -> str:
    ensure_output_dir()
    output_path = OUTPUT_DIR / output_name
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def audio_result_to_dict(result: AudioResult) -> dict:
    return {
        "path": result.path,
        "voice": result.voice,
        "words": [asdict(word) for word in result.words],
        "cues": [
            {"start": cue.start, "end": cue.end, "text": cue.text}
            for cue in result.cues
        ],
    }
