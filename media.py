import asyncio
import base64
import binascii
import io
import json
import logging
import math
import mimetypes
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import edge_tts
import imageio_ffmpeg
import numpy as np
import requests
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from config import (
    BASE_DIR,
    DOG_NAME,
    ELEVENLABS_API_KEY,
    ELEVENLABS_MAX_RETRIES,
    ELEVENLABS_MODEL,
    ELEVENLABS_OUTPUT_FORMAT,
    ELEVENLABS_SIMILARITY_BOOST,
    ELEVENLABS_SPEAKER_BOOST,
    ELEVENLABS_SPEED,
    ELEVENLABS_STABILITY,
    ELEVENLABS_STYLE,
    ELEVENLABS_VOICE_ID,
    GEMINI_API_KEY,
    GEMINI_TTS_FALLBACK_MODELS,
    GEMINI_TTS_MAX_RETRIES,
    GEMINI_TTS_MODEL,
    GEMINI_TTS_VOICE,
    TTS_FALLBACK_VOICES,
    TTS_EFFECTS_ENABLED,
    TTS_MAX_RETRIES,
    TTS_PITCH,
    TTS_PRESET,
    TTS_PROVIDER,
    TTS_RATE,
    TTS_SIGNATURE_LAUGH_ENABLED,
    TTS_SIGNATURE_LAUGH_FILE,
    TTS_VOICE,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from tts_support import build_performance_prompt, extract_audio_bytes, save_pcm_wav
from reel_formats import normalise_reel_format

try:
    from moviepy import AudioFileClip, VideoClip
except ImportError:  # MoviePy 1.x
    from moviepy.editor import AudioFileClip, VideoClip


LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = BASE_DIR / "output"
MAX_DOWNLOAD_BYTES = 60 * 1024 * 1024
MAX_GIF_FRAMES = 450

# Pełnoekranowy widok mobilny 9:16 pozostaje głównym kadrem rolki. Okładka ma
# osobne położenie tekstu, które nadal mieści się w centralnym podglądzie 3:4.
INSTAGRAM_HEADLINE_TOP = 205
INSTAGRAM_COVER_BRAND_TOP = 270
INSTAGRAM_COVER_TITLE_TOP = 336
REEL_HEADLINE_MAX_WIDTH = 880
REEL_CAPTION_MAX_WIDTH = 900
COVER_TITLE_MAX_WIDTH = 840


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
    preset: str = ""
    provider: str = ""
    model: str = ""


@dataclass(frozen=True)
class VoicePreset:
    """Autorski profil ekspresji, niezależny od nazw i materiałów CapCut."""

    rate: str
    pitch: str
    audio_filter: str
    voice: str = ""
    max_pause: float = 0.0


VOICE_PRESETS: dict[str, VoicePreset] = {
    "jogi_playful_soft": VoicePreset(
        rate="+2%",
        pitch="+18Hz",
        audio_filter=(
            "highpass=f=90,lowpass=f=12000,"
            "equalizer=f=2800:t=q:w=1.2:g=1.5,"
            "acompressor=threshold=0.18:ratio=2.2:attack=8:release=90:makeup=1.25,"
            "alimiter=limit=0.95"
        ),
    ),
    "jogi_playful": VoicePreset(
        rate="+10%",
        pitch="+28Hz",
        audio_filter=(
            "highpass=f=105,lowpass=f=11500,"
            "equalizer=f=180:t=q:w=1:g=-1.5,"
            "equalizer=f=3200:t=q:w=1.1:g=2.8,"
            "acompressor=threshold=0.16:ratio=2.8:attack=6:release=80:makeup=1.35,"
            "alimiter=limit=0.94"
        ),
    ),
    "jogi_playful_wild": VoicePreset(
        rate="+14%",
        pitch="+42Hz",
        audio_filter=(
            "highpass=f=120,lowpass=f=10500,"
            "equalizer=f=200:t=q:w=1:g=-2.5,"
            "equalizer=f=3600:t=q:w=1:g=4,"
            "acompressor=threshold=0.14:ratio=3.2:attack=5:release=70:makeup=1.45,"
            "alimiter=limit=0.93"
        ),
    ),
    "jogi_urwis": VoicePreset(
        rate="-4%",
        pitch="+34Hz",
        voice="pl-PL-MarekNeural",
        max_pause=0.42,
        audio_filter=(
            "highpass=f=85,lowpass=f=11500,"
            "equalizer=f=180:t=q:w=1:g=1.2,"
            "equalizer=f=2900:t=q:w=1.15:g=2.2,"
            "acompressor=threshold=0.17:ratio=2.5:attack=7:release=95:makeup=1.3,"
            "alimiter=limit=0.95"
        ),
    ),
}


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


def resolve_output_path(output_name: str | Path) -> Path:
    """Zwraca bezpieczną ścieżkę wewnątrz output i tworzy jej katalog nadrzędny."""
    relative = Path(output_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Nazwa pliku wyjściowego musi być ścieżką względną wewnątrz output.")
    path = OUTPUT_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
    output_path_for_download = resolve_output_path(f"{output_stem}{extension}")
    with output_path_for_download.open("wb") as file:
        file.write(data)
    return str(output_path_for_download)


def download_media_links(value: object, row_idx: int) -> list[str]:
    links = parse_media_links(value)
    if not links:
        raise ValueError("Komórka Link nie zawiera żadnego poprawnego adresu HTTP(S).")
    paths = []
    for index, link in enumerate(links, start=1):
        LOGGER.info("Pobieranie medium %s/%s", index, len(links))
        paths.append(download_media(link, f"wiersz_{row_idx}/media_{index}"))
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


def _fit_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_lines: int,
    start_font_size: int,
    min_font_size: int,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[list[str]], int]:
    """Zmniejsza tekst do zadanej liczby linii, ale nigdy nie usuwa słów."""
    words = text.split()
    for font_size in range(start_font_size, min_font_size - 1, -2):
        font = _font(font_size)
        lines = _wrap_words(draw, words, font, max_width)
        if len(lines) <= max_lines:
            return font, lines, font_size
    font = _font(min_font_size)
    return font, _wrap_words(draw, words, font, max_width), min_font_size


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


def _ken_burns(
    image: Image.Image,
    progress: float,
    segment_index: int,
    animated: bool,
    strength_scale: float = 1.0,
) -> Image.Image:
    progress = min(1.0, max(0.0, progress))
    smooth = progress * progress * (3 - 2 * progress)
    if segment_index % 2:
        smooth = 1.0 - smooth
    strength = (0.026 if animated else 0.058) * max(0.0, strength_scale)
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
    font, lines, font_size = _fit_wrapped_text(
        draw,
        text,
        max_width=REEL_HEADLINE_MAX_WIDTH,
        max_lines=2,
        start_font_size=76,
        min_font_size=52,
    )
    line_height = round(font_size * 1.16)
    total_height = len(lines) * line_height
    top = INSTAGRAM_HEADLINE_TOP
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
    lines = _wrap_words(draw, words, font, REEL_CAPTION_MAX_WIDTH)[:2]
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


def limit_caption_cues(
    cues: list[CaptionCue],
    maximum_end: float,
) -> list[CaptionCue]:
    """Kończy napisy przed podpisem dźwiękowym lub innym outro bez mowy."""
    return [
        CaptionCue(cue.start, min(cue.end, maximum_end), cue.words)
        for cue in cues
        if cue.start < maximum_end
    ]


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


def _voice_preset(name: str) -> VoicePreset:
    try:
        return VOICE_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(VOICE_PRESETS)
        raise ValueError(f"Nieznany preset głosu {name!r}. Dostępne: {available}") from exc


async def _stream_edge_audio(
    text: str,
    output_path: Path,
    voice: str,
    rate: str,
    pitch: str,
) -> list[TimedWord]:
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
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


def _audio_duration(path: Path) -> float:
    with AudioFileClip(str(path)) as audio:
        return float(audio.duration)


def _apply_voice_effect(source_path: Path, output_path: Path, audio_filter: str) -> float:
    """Nakłada neutralną obróbkę i zwraca stosunek nowej długości do źródłowej."""
    source_duration = _audio_duration(source_path)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-af",
        audio_filter,
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        details = completed.stderr.strip() or "FFmpeg nie zwrócił szczegółów błędu"
        raise RuntimeError(f"Nie udało się nałożyć efektu głosu: {details}")
    output_duration = _audio_duration(output_path)
    if source_duration <= 0 or output_duration <= 0:
        raise RuntimeError("Obróbka głosu zwróciła nagranie o niepoprawnej długości.")
    return output_duration / source_duration


def _silence_cuts(
    silences: list[tuple[float, float]],
    max_pause: float,
    duration: float,
) -> list[tuple[float, float]]:
    """Wyznacza części wewnętrznych pauz do usunięcia, bez cięcia początku i końca."""
    if max_pause <= 0:
        return []
    cuts: list[tuple[float, float]] = []
    half_pause = max_pause / 2
    for silence_start, silence_end in silences:
        if silence_start <= 0.05 or silence_end >= duration - 0.05:
            continue
        if silence_end - silence_start <= max_pause:
            continue
        cut_start = silence_start + half_pause
        cut_end = silence_end - half_pause
        if cut_end - cut_start >= 0.02:
            cuts.append((cut_start, cut_end))
    return cuts


def _detect_silences(source_path: Path) -> list[tuple[float, float]]:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-i",
        str(source_path),
        "-af",
        "silencedetect=noise=-36dB:d=0.08",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    starts = [
        float(value)
        for value in re.findall(r"silence_start:\s*([0-9.]+)", completed.stderr)
    ]
    ends = [
        float(value)
        for value in re.findall(r"silence_end:\s*([0-9.]+)", completed.stderr)
    ]
    return list(zip(starts, ends))


def _retime_words_after_cuts(
    words: list[TimedWord],
    cuts: list[tuple[float, float]],
) -> list[TimedWord]:
    def removed_before(timestamp: float) -> float:
        return sum(max(0.0, min(timestamp, end) - start) for start, end in cuts)

    return [
        TimedWord(
            word.text,
            word.start - removed_before(word.start),
            word.end - removed_before(word.end),
        )
        for word in words
    ]


def _compact_long_pauses(
    source_path: Path,
    output_path: Path,
    words: list[TimedWord],
    max_pause: float,
) -> list[TimedWord]:
    duration = _audio_duration(source_path)
    cuts = _silence_cuts(_detect_silences(source_path), max_pause, duration)
    if not cuts:
        return words

    kept_intervals: list[tuple[float, float]] = []
    cursor = 0.0
    for cut_start, cut_end in cuts:
        kept_intervals.append((cursor, cut_start))
        cursor = cut_end
    kept_intervals.append((cursor, duration))

    filters = []
    labels = []
    for index, (start, end) in enumerate(kept_intervals):
        label = f"part{index}"
        filters.append(
            f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]")

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        details = completed.stderr.strip() or "FFmpeg nie zwrócił szczegółów błędu"
        raise RuntimeError(f"Nie udało się skrócić pauz lektora: {details}")
    return _retime_words_after_cuts(words, cuts)


def _scale_words(words: list[TimedWord], factor: float) -> list[TimedWord]:
    if math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=0.002):
        return words
    return [TimedWord(word.text, word.start * factor, word.end * factor) for word in words]


def _generate_edge_audio_with_timings(
    text: str,
    output_name: str = "temp_audio.mp3",
    preset_name: str | None = None,
) -> AudioResult:
    ensure_output_dir()
    output_path = resolve_output_path(output_name)
    selected_preset = (preset_name or TTS_PRESET).strip().lower()
    preset = _voice_preset(selected_preset)
    rate = TTS_RATE or preset.rate
    pitch = TTS_PITCH or preset.pitch
    source_path = output_path.with_name(f"{output_path.stem}.source{output_path.suffix}")
    voices = list(
        dict.fromkeys(
            voice for voice in (preset.voice, TTS_VOICE, *TTS_FALLBACK_VOICES) if voice
        )
    )
    errors: list[str] = []
    selected_voice = ""
    words: list[TimedWord] = []

    for voice in voices:
        for attempt in range(1, max(1, TTS_MAX_RETRIES) + 1):
            output_path.unlink(missing_ok=True)
            source_path.unlink(missing_ok=True)
            try:
                words = asyncio.run(_stream_edge_audio(text, source_path, voice, rate, pitch))
                if not source_path.exists() or source_path.stat().st_size == 0:
                    raise RuntimeError("usługa nie zwróciła danych audio")
                if preset.max_pause:
                    paced_path = source_path.with_name(
                        f"{source_path.stem}.paced{source_path.suffix}"
                    )
                    paced_path.unlink(missing_ok=True)
                    paced_words = _compact_long_pauses(
                        source_path,
                        paced_path,
                        words,
                        preset.max_pause,
                    )
                    if paced_path.exists():
                        source_path.unlink(missing_ok=True)
                        paced_path.replace(source_path)
                        words = paced_words
                if TTS_EFFECTS_ENABLED:
                    duration_factor = _apply_voice_effect(source_path, output_path, preset.audio_filter)
                    words = _scale_words(words, duration_factor)
                    source_path.unlink(missing_ok=True)
                else:
                    source_path.replace(output_path)
                selected_voice = voice
                break
            except Exception as exc:
                output_path.unlink(missing_ok=True)
                source_path.unlink(missing_ok=True)
                source_path.with_name(f"{source_path.stem}.paced{source_path.suffix}").unlink(
                    missing_ok=True
                )
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
    return AudioResult(
        str(output_path),
        tuple(words),
        tuple(cues),
        selected_voice,
        selected_preset,
        "edge",
    )


def _is_gemini_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "resource_exhausted" in message or "quota exceeded" in message


def _generate_gemini_pcm(text: str, output_path: Path) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Brak GEMINI_API_KEY potrzebnego do syntezy mowy Gemini.")
    prompt = build_performance_prompt(text)
    client = genai.Client(api_key=GEMINI_API_KEY)
    errors: list[str] = []
    requested_models = [GEMINI_TTS_MODEL, *GEMINI_TTS_FALLBACK_MODELS]
    models = list(dict.fromkeys(model for model in requested_models if model))
    retries = max(1, GEMINI_TTS_MAX_RETRIES)
    last_error: Exception | None = None
    for model_name in models:
        for attempt in range(1, retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=GEMINI_TTS_VOICE,
                                )
                            )
                        ),
                    ),
                )
                save_pcm_wav(output_path, extract_audio_bytes(response))
                return model_name
            except Exception as exc:
                last_error = exc
                output_path.unlink(missing_ok=True)
                details = f"{model_name}, próba {attempt}/{retries}: {exc}"
                errors.append(details)
                LOGGER.warning("Gemini TTS nie wygenerował nagrania (%s)", details)
                # Limit jest przypisany do modelu. Kolejne natychmiastowe próby tylko
                # wydłużają pracę, więc przechodzimy od razu do modelu zapasowego.
                if _is_gemini_quota_error(exc):
                    break
                if attempt < retries:
                    time.sleep(attempt)
    raise RuntimeError("Gemini TTS nie wygenerował nagrania: " + " | ".join(errors)) from last_error


def _append_signature_laugh(
    narration_path: Path,
    laugh_path: Path,
    output_path: Path,
) -> None:
    if not laugh_path.exists():
        raise FileNotFoundError(f"Brak pliku podpisu dźwiękowego: {laugh_path}")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(narration_path),
        "-i",
        str(laugh_path),
        "-filter_complex",
        (
            "[0:a]aresample=24000,aformat=sample_fmts=s16:channel_layouts=mono[voice];"
            "[1:a]aresample=24000,aformat=sample_fmts=s16:channel_layouts=mono[laugh];"
            "[voice][laugh]concat=n=2:v=0:a=1[out]"
        ),
        "-map",
        "[out]",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        details = completed.stderr.strip() or "FFmpeg nie zwrócił szczegółów błędu"
        raise RuntimeError(f"Nie udało się dodać podpisu dźwiękowego: {details}")


def _timed_words_from_alignment(alignment: dict | None) -> list[TimedWord]:
    """Zamienia znaczniki czasu znaków ElevenLabs na znaczniki całych słów."""
    if not alignment:
        return []
    characters = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []
    if not characters or not (len(characters) == len(starts) == len(ends)):
        return []

    words: list[TimedWord] = []
    token_chars: list[str] = []
    token_start: float | None = None
    token_end: float | None = None

    def flush_token() -> None:
        nonlocal token_chars, token_start, token_end
        if token_chars and token_start is not None and token_end is not None:
            token = "".join(token_chars)
            if words and not any(character.isalnum() for character in token):
                previous = words[-1]
                words[-1] = TimedWord(
                    previous.text + token,
                    previous.start,
                    max(previous.end, token_end),
                )
            else:
                words.append(TimedWord(token, token_start, token_end))
        token_chars = []
        token_start = None
        token_end = None

    try:
        for character, start, end in zip(characters, starts, ends):
            character = str(character)
            start_value = max(0.0, float(start))
            end_value = max(start_value, float(end))
            if character.isspace():
                flush_token()
                continue
            if token_start is None:
                token_start = start_value
            token_chars.append(character)
            token_end = end_value
    except (TypeError, ValueError):
        return []
    flush_token()
    return words


def _request_elevenlabs_audio(text: str, output_path: Path) -> list[TimedWord]:
    if not ELEVENLABS_API_KEY:
        raise RuntimeError(
            "Brak ELEVENLABS_API_KEY. Uzupełnij go w .env albo ustaw TTS_PROVIDER=gemini."
        )
    if not ELEVENLABS_VOICE_ID:
        raise RuntimeError(
            "Brak ELEVENLABS_VOICE_ID. Wklej identyfikator klonu do .env "
            "albo ustaw TTS_PROVIDER=gemini."
        )
    if not ELEVENLABS_OUTPUT_FORMAT.startswith("mp3_"):
        raise ValueError(
            "Generator obsługuje obecnie format ElevenLabs zaczynający się od 'mp3_'."
        )

    url = (
        "https://api.elevenlabs.io/v1/text-to-speech/"
        f"{quote(ELEVENLABS_VOICE_ID, safe='')}/with-timestamps"
    )
    response = requests.post(
        url,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        params={"output_format": ELEVENLABS_OUTPUT_FORMAT},
        json={
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "language_code": "pl",
            "voice_settings": {
                "stability": ELEVENLABS_STABILITY,
                "similarity_boost": ELEVENLABS_SIMILARITY_BOOST,
                "style": ELEVENLABS_STYLE,
                "use_speaker_boost": ELEVENLABS_SPEAKER_BOOST,
                "speed": ELEVENLABS_SPEED,
            },
        },
        timeout=(10, 120),
    )
    if not response.ok:
        try:
            details = response.json().get("detail", "")
            if isinstance(details, dict):
                details = details.get("message") or details.get("status") or ""
        except (ValueError, AttributeError):
            details = ""
        suffix = f": {str(details)[:240]}" if details else ""
        raise RuntimeError(f"ElevenLabs HTTP {response.status_code}{suffix}")

    try:
        payload = response.json()
        audio_bytes = base64.b64decode(payload["audio_base64"], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise RuntimeError("ElevenLabs zwrócił niepełną odpowiedź audio.") from exc
    if not audio_bytes:
        raise RuntimeError("ElevenLabs zwrócił pusty plik audio.")
    output_path.write_bytes(audio_bytes)

    alignment = payload.get("normalized_alignment") or payload.get("alignment")
    return _timed_words_from_alignment(alignment)


def _generate_elevenlabs_audio_with_timings(
    text: str,
    output_name: str = "temp_audio.mp3",
) -> AudioResult:
    ensure_output_dir()
    requested_path = resolve_output_path(output_name)
    signature_used = TTS_SIGNATURE_LAUGH_ENABLED
    output_path = requested_path.with_suffix(".wav" if signature_used else ".mp3")
    narration_path = (
        output_path.with_name(f"{output_path.stem}.narration.mp3")
        if signature_used
        else output_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    narration_path.unlink(missing_ok=True)

    retries = max(1, ELEVENLABS_MAX_RETRIES)
    errors: list[str] = []
    words: list[TimedWord] = []
    try:
        for attempt in range(1, retries + 1):
            narration_path.unlink(missing_ok=True)
            try:
                words = _request_elevenlabs_audio(text, narration_path)
                break
            except Exception as exc:
                narration_path.unlink(missing_ok=True)
                errors.append(f"próba {attempt}/{retries}: {exc}")
                LOGGER.warning(
                    "ElevenLabs nie wygenerował nagrania (próba %s/%s): %s",
                    attempt,
                    retries,
                    exc,
                )
                message = str(exc)
                if any(f"HTTP {status}" in message for status in (400, 401, 403, 404, 422)):
                    break
                if attempt < retries:
                    time.sleep(attempt)
        if not narration_path.exists() or narration_path.stat().st_size == 0:
            raise RuntimeError(
                "ElevenLabs nie wygenerował nagrania: " + " | ".join(errors)
            )

        narration_duration = _audio_duration(narration_path)
        if not words:
            words = _approximate_words(text, narration_duration)
        if signature_used:
            _append_signature_laugh(
                narration_path,
                Path(TTS_SIGNATURE_LAUGH_FILE),
                output_path,
            )
            narration_path.unlink(missing_ok=True)
    except Exception:
        output_path.unlink(missing_ok=True)
        if narration_path != output_path:
            narration_path.unlink(missing_ok=True)
        raise

    cues = limit_caption_cues(build_caption_cues(words), narration_duration)
    preset = "elevenlabs_clone_signature" if signature_used else "elevenlabs_clone"
    return AudioResult(
        str(output_path),
        tuple(words),
        tuple(cues),
        ELEVENLABS_VOICE_ID,
        preset,
        "elevenlabs",
        ELEVENLABS_MODEL,
    )


def _generate_gemini_audio_with_timings(
    text: str,
    output_name: str = "temp_audio.wav",
) -> AudioResult:
    ensure_output_dir()
    requested_path = resolve_output_path(output_name)
    output_path = requested_path.with_suffix(".wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    narration_path = output_path.with_name(f"{output_path.stem}.narration.wav")
    output_path.unlink(missing_ok=True)
    narration_path.unlink(missing_ok=True)

    try:
        selected_model = _generate_gemini_pcm(text, narration_path) or GEMINI_TTS_MODEL
        narration_duration = _audio_duration(narration_path)
        signature_used = TTS_SIGNATURE_LAUGH_ENABLED
        if signature_used:
            _append_signature_laugh(
                narration_path,
                Path(TTS_SIGNATURE_LAUGH_FILE),
                output_path,
            )
            narration_path.unlink(missing_ok=True)
        else:
            narration_path.replace(output_path)
    except Exception:
        output_path.unlink(missing_ok=True)
        narration_path.unlink(missing_ok=True)
        raise

    words = _approximate_words(text, narration_duration)
    cues = limit_caption_cues(build_caption_cues(words), narration_duration)
    preset = "achird_warm_signature" if signature_used else "achird"
    return AudioResult(
        str(output_path),
        tuple(words),
        tuple(cues),
        GEMINI_TTS_VOICE,
        preset,
        "gemini",
        selected_model,
    )


def generate_audio_with_timings(
    text: str,
    output_name: str = "temp_audio.mp3",
    preset_name: str | None = None,
    provider: str | None = None,
) -> AudioResult:
    selected_provider = (provider or TTS_PROVIDER).strip().lower()
    if selected_provider == "gemini":
        return _generate_gemini_audio_with_timings(text, output_name)
    if selected_provider == "elevenlabs":
        return _generate_elevenlabs_audio_with_timings(text, output_name)
    if selected_provider == "edge":
        return _generate_edge_audio_with_timings(text, output_name, preset_name)
    raise ValueError(
        "TTS_PROVIDER musi mieć wartość 'gemini', 'elevenlabs' albo 'edge'."
    )


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


def _segment_position(
    item_count: int,
    duration: float,
    current_time: float,
) -> tuple[int, float, float]:
    segment_duration = duration / max(1, item_count)
    index = min(item_count - 1, int(current_time / max(segment_duration, 0.001)))
    local_time = max(0.0, current_time - index * segment_duration)
    progress = min(1.0, local_time / max(segment_duration, 0.001))
    return index, local_time, progress


def _asset_motion_frame(
    asset: VisualAsset,
    local_time: float,
    progress: float,
    index: int,
    strength: float,
) -> Image.Image:
    return _ken_burns(
        asset.frame_at(local_time),
        progress,
        index,
        asset.animated,
        strength_scale=strength,
    ).convert("RGBA")


def _punchline_frame(
    assets: list[VisualAsset],
    duration: float,
    current_time: float,
) -> Image.Image:
    index, local_time, progress = _segment_position(len(assets), duration, current_time)
    # Mocniejsze, bezpośrednie przybliżenie i twarde cięcia pod krótką puentę.
    return _asset_motion_frame(assets[index], local_time, progress, index, strength=1.45)


def _crossfaded_story_frame(
    assets: list[VisualAsset],
    duration: float,
    current_time: float,
    *,
    strength: float,
    transition_fraction: float,
) -> Image.Image:
    index, local_time, progress = _segment_position(len(assets), duration, current_time)
    current = _asset_motion_frame(assets[index], local_time, progress, index, strength)
    if index >= len(assets) - 1 or progress < 1.0 - transition_fraction:
        return current

    blend = (progress - (1.0 - transition_fraction)) / max(transition_fraction, 0.001)
    next_frame = _asset_motion_frame(
        assets[index + 1],
        0.0,
        blend,
        index + 1,
        strength,
    )
    return Image.blend(current, next_frame, min(1.0, max(0.0, blend)))


def _comparison_frame(
    assets: list[VisualAsset],
    duration: float,
    current_time: float,
) -> Image.Image:
    if len(assets) < 2:
        return _punchline_frame(assets, duration, current_time)

    pair_count = len(assets) - 1
    pair_index, local_time, progress = _segment_position(pair_count, duration, current_time)
    left_index = pair_index
    right_index = pair_index + 1
    left = _asset_motion_frame(
        assets[left_index], local_time, progress, left_index, strength=0.7
    )
    right = _asset_motion_frame(
        assets[right_index], local_time, 1.0 - progress, right_index, strength=0.7
    )

    half = VIDEO_WIDTH // 2
    left_half = ImageOps.fit(left, (half, VIDEO_HEIGHT), method=Image.Resampling.LANCZOS)
    right_half = ImageOps.fit(
        right,
        (VIDEO_WIDTH - half, VIDEO_HEIGHT),
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 255))
    canvas.paste(left_half, (0, 0))
    canvas.paste(right_half, (half, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((half - 4, 0, half + 4, VIDEO_HEIGHT), fill=(46, 111, 255, 255))
    return canvas


def _diary_mood_frame(
    assets: list[VisualAsset],
    duration: float,
    current_time: float,
) -> Image.Image:
    image = _crossfaded_story_frame(
        assets,
        duration,
        current_time,
        strength=0.48,
        transition_fraction=0.24,
    )
    # Delikatna, ciepła warstwa spaja spokojniejsze wpisy bez zasłaniania zdjęcia.
    mood = Image.new("RGBA", image.size, (24, 12, 42, 20))
    return Image.alpha_composite(image, mood)


def render_frame(
    assets: list[VisualAsset],
    duration: float,
    current_time: float,
    headline: str = "",
    caption_cues: list[CaptionCue] | tuple[CaptionCue, ...] = (),
    asset_order: list[int] | None = None,
    format_id: str = "punchline",
) -> np.ndarray:
    ordered = _ordered_assets(assets, asset_order)
    selected_format = normalise_reel_format(format_id)
    if selected_format == "mini_story":
        image = _crossfaded_story_frame(
            ordered,
            duration,
            current_time,
            strength=0.9,
            transition_fraction=0.14,
        )
    elif selected_format == "comparison":
        image = _comparison_frame(ordered, duration, current_time)
    elif selected_format == "diary_mood":
        image = _diary_mood_frame(ordered, duration, current_time)
    else:
        image = _punchline_frame(ordered, duration, current_time)

    headline_share = {
        "punchline": 0.16,
        "mini_story": 0.20,
        "comparison": 0.18,
        "diary_mood": 0.24,
    }[selected_format]
    headline_duration = min(2.2, max(1.1, duration * headline_share))
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
    format_id: str = "punchline",
    logger: str | None = "bar",
) -> str:
    ensure_output_dir()
    paths = [image_paths] if isinstance(image_paths, str) else image_paths
    assets = load_visual_assets(paths)
    output_path = resolve_output_path(output_name)
    audio = AudioFileClip(audio_path)
    duration = float(audio.duration)
    if duration <= 0:
        audio.close()
        raise ValueError("Nagranie lektora ma niepoprawną długość.")

    def frame_function(t: float) -> np.ndarray:
        return render_frame(
            assets,
            duration,
            float(t),
            headline,
            caption_cues,
            asset_order,
            format_id,
        )

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
    brand_y = INSTAGRAM_COVER_BRAND_TOP
    title_bar_left = 76
    title_bar_right = 88
    title_text_x = 112
    draw.text((title_text_x, brand_y), brand, font=brand_font, fill=(215, 227, 255, 255))

    title_font_size = 78
    while title_font_size >= 56:
        title_font = _font(title_font_size)
        lines = _cover_title_lines(draw, text, title_font, max_width=COVER_TITLE_MAX_WIDTH)
        if len(lines) <= 2:
            break
        title_font_size -= 2
    else:
        title_font_size = 56
        title_font = _font(title_font_size)
        lines = _wrap_words(draw, text.split(), title_font, COVER_TITLE_MAX_WIDTH)[:2]
    start_x = title_text_x
    start_y = INSTAGRAM_COVER_TITLE_TOP
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
    output_path = resolve_output_path(output_name)
    image.convert("RGB").save(output_path, quality=93, optimize=True)
    return str(output_path)


def prepare_916_image_with_text(image_path: str, text: str, output_name: str = "processed_image.jpg") -> str:
    """Wstecznie zgodny eksport pojedynczej planszy 9:16."""
    ensure_output_dir()
    asset = load_visual_asset(image_path)
    image = asset.frames[0].convert("RGBA")
    _draw_headline(image, text)
    output_path = resolve_output_path(output_name)
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
    output_path = resolve_output_path(output_name)
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n{_srt_timestamp(cue.start)} --> {_srt_timestamp(cue.end)}\n{cue.text}\n"
        )
    output_path.write_text("\n".join(blocks), encoding="utf-8")
    return str(output_path)


def save_instagram_caption(caption_text: str, output_name: str = "post_caption.txt") -> str:
    ensure_output_dir()
    output_path = resolve_output_path(output_name)
    output_path.write_text(caption_text.strip() + "\n", encoding="utf-8")
    return str(output_path)


def save_manifest(data: dict, output_name: str) -> str:
    ensure_output_dir()
    output_path = resolve_output_path(output_name)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def audio_result_to_dict(result: AudioResult) -> dict:
    return {
        "path": result.path,
        "voice": result.voice,
        "preset": result.preset,
        "provider": result.provider,
        "model": result.model,
        "words": [asdict(word) for word in result.words],
        "cues": [
            {"start": cue.start, "end": cue.end, "text": cue.text}
            for cue in result.cues
        ],
    }
