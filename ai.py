import io
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, GEMINI_MODEL


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

    requested_models = [GEMINI_MODEL, "gemini-2.5-flash"]
    models = list(dict.fromkeys(model for model in requested_models if model))
    errors: list[str] = []
    client = genai.Client(api_key=GEMINI_API_KEY)

    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=inputs,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ReelContent,
                    temperature=0.75,
                    max_output_tokens=1400,
                ),
            )
            parsed = response.parsed
            if isinstance(parsed, ReelContent):
                return _normalise_content(parsed, media_count)
            if response.text:
                return _normalise_content(ReelContent.model_validate_json(response.text), media_count)
            raise ValueError("Gemini zwrócił pustą odpowiedź.")
        except Exception as exc:
            LOGGER.warning("Model %s nie wygenerował treści: %s", model_name, exc)
            errors.append(f"{model_name}: {exc}")
            time.sleep(1)

    raise RuntimeError("Nie udało się wygenerować treści przez Gemini. " + " | ".join(errors))
