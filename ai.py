import io
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from config import (
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODELS,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_THINKING_LEVEL,
)


LOGGER = logging.getLogger(__name__)
MAX_MEDIA_FOR_AI = 12


class ReelContent(BaseModel):
    """Ustrukturyzowany plan jednej rolki."""

    voiceover: str = Field(description="Naturalny tekst lektora z perspektywy Jogiego.")
    headline: str = Field(description="Hook na pierwsze 1-2 sekundy, 3-7 słów.")
    cover_title: str = Field(description="Tytuł osobnej okładki, maksymalnie 5 słów.")
    caption_body: str = Field(description="Opis posta bez hashtagów.")
    hashtags: list[str] = Field(description="Od 3 do 5 konkretnych hashtagów.")
    alt_text: str = Field(description="Krótki, rzeczowy tekst alternatywny mediów.")
    asset_order: list[int] = Field(description="Kolejność indeksów przesłanych mediów.")


def _media_preview(path: str | Path) -> types.Part:
    """Tworzy nieduży podgląd JPEG do analizy multimodalnej."""
    with Image.open(path) as image:
        image.seek(0)
        preview = ImageOps.exif_transpose(image.copy()).convert("RGB")
        preview.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        preview.save(buffer, format="JPEG", quality=88, optimize=True)
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def _normalise_content(content: ReelContent, media_count: int) -> dict:
    order: list[int] = []
    for index in content.asset_order:
        if 0 <= index < media_count and index not in order:
            order.append(index)
    order.extend(index for index in range(media_count) if index not in order)

    hashtags: list[str] = []
    for hashtag in content.hashtags:
        clean = "#" + hashtag.strip().lstrip("#").replace(" ", "")
        if len(clean) > 1 and clean.lower() not in {item.lower() for item in hashtags}:
            hashtags.append(clean)
    if not hashtags:
        hashtags = ["#JogiPudel", "#Pudel", "#Instapies"]

    caption_body = content.caption_body.strip()
    hashtags = hashtags[:5]
    caption = caption_body + "\n\n" + " ".join(hashtags)

    return {
        "lektor": content.voiceover.strip(),
        "naglowek": content.headline.strip().upper(),
        "cover_title": content.cover_title.strip().upper(),
        "caption_body": caption_body,
        "hashtags": hashtags,
        "caption": caption,
        "alt_text": content.alt_text.strip(),
        "asset_order": order,
    }


def _finish_reason(response: object) -> str:
    try:
        candidates = getattr(response, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        return getattr(reason, "value", None) or str(reason or "unknown")
    except Exception:
        return "unknown"


def _parse_response(response: object, media_count: int) -> dict:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        content = parsed if isinstance(parsed, ReelContent) else ReelContent.model_validate(parsed)
        return _normalise_content(content, media_count)

    response_text = getattr(response, "text", None) or ""
    if not response_text.strip():
        raise ValueError(f"Gemini zwrócił pustą odpowiedź (finish_reason={_finish_reason(response)}).")
    try:
        content = ReelContent.model_validate_json(response_text)
    except Exception as exc:
        # Nie dołączamy treści odpowiedzi do wyjątku: może zawierać dane z dziennika.
        raise ValueError(
            "Gemini zwrócił niepełny lub niepoprawny JSON "
            f"(finish_reason={_finish_reason(response)}, znaki={len(response_text)})."
        ) from exc
    return _normalise_content(content, media_count)


def _thinking_level() -> types.ThinkingLevel:
    level_name = GEMINI_THINKING_LEVEL.upper()
    try:
        return types.ThinkingLevel[level_name]
    except KeyError:
        LOGGER.warning("Nieznany GEMINI_THINKING_LEVEL=%s; używam minimal.", GEMINI_THINKING_LEVEL)
        return types.ThinkingLevel.MINIMAL


def _is_non_retryable_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("404", "not_found", "not available", "permission_denied", "invalid_argument")
    )


def generate_reel_content(
    nazwa: str,
    opis: str,
    media_paths: list[str] | None = None,
) -> dict:
    """Generuje tekst i plan rolki, analizując także faktyczne zdjęcia/GIF-y."""
    if not GEMINI_API_KEY:
        raise RuntimeError("Brak GEMINI_API_KEY w pliku .env.")

    paths = [Path(path) for path in (media_paths or [])][:MAX_MEDIA_FOR_AI]
    media_count = len(paths)
    prompt = f"""
Jesteś polskim strategiem Instagram Reels i scenarzystą konta psa Jogi —
pudla o pogodnym, inteligentnym i lekko zadziornym charakterze.

Dane z dziennika:
- nazwa wydarzenia: {nazwa or 'brak'}
- opis wydarzenia: {opis or 'brak'}
- liczba mediów: {media_count}

Do wiadomości dołączono media w kolejności indeksów 0..{max(media_count - 1, 0)}.
Traktuj obraz jako źródło prawdy. Nie wymyślaj jedzenia, miejsc, zachowań ani
rezultatu wydarzenia, jeśli nie wynika to z opisu albo obrazu. Jeśli dane są
niepełne, użyj bezpiecznej, humorystycznej obserwacji zamiast zmyślonego faktu.

Przygotuj jedną spójną rolkę:
1. voiceover: 24-45 słów, około 7-14 sekund, pierwsza osoba psa, krótkie zdania,
   naturalna polszczyzna i interpunkcja pomagająca lektorowi. Jedna mini-historia:
   hook, rozwinięcie, puenta. Bez słów „wirusowy”, „słodziak” i bez żebrania o lajki.
2. headline: 3-7 słów, zatrzymuje przewijanie, ale nie jest clickbaitem.
3. cover_title: maksymalnie 5 słów, czytelny poza kontekstem rolki.
4. caption_body: 250-600 znaków, bez hashtagów. Pierwsze zdanie ma zawierać
   naturalne słowa kluczowe (Jogi/pudel/pies + temat lub pewna lokalizacja).
   Maksymalnie 3 emoji. Na końcu jedno konkretne i łatwe pytanie.
5. hashtags: 3-5 konkretnych tagów. Jeden firmowy #JogiPudel, reszta niszowa,
   tematyczna lub lokalna. Nie używaj długiej listy ogólnych angielskich tagów.
6. alt_text: rzeczowy opis tego, co faktycznie widać.
7. asset_order: najlepsza kolejność wszystkich indeksów — najmocniejszy kadr
   jako pierwszy, potem rozwinięcie i puenta. Każdy poprawny indeks dokładnie raz.
"""

    inputs: list[object] = [prompt]
    for path in paths:
        try:
            inputs.append(_media_preview(path))
        except Exception as exc:
            LOGGER.warning("Nie udało się przygotować podglądu %s: %s", path, exc)

    requested_models = [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]
    models = list(dict.fromkeys(model for model in requested_models if model))
    errors: list[str] = []
    client = genai.Client(api_key=GEMINI_API_KEY)

    for model_name in models:
        for attempt in range(1, max(1, GEMINI_MAX_RETRIES) + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=inputs,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ReelContent,
                        temperature=0.65,
                        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=_thinking_level(),
                            include_thoughts=False,
                        ),
                    ),
                )
                return _parse_response(response, media_count)
            except Exception as exc:
                message = f"{model_name}, próba {attempt}/{max(1, GEMINI_MAX_RETRIES)}: {exc}"
                LOGGER.warning("Model %s nie wygenerował treści: %s", model_name, message)
                errors.append(message)
                if _is_non_retryable_model_error(exc):
                    break
                if attempt < max(1, GEMINI_MAX_RETRIES):
                    time.sleep(min(2 ** (attempt - 1), 4))

    raise RuntimeError("Nie udało się wygenerować treści przez Gemini. " + " | ".join(errors))
