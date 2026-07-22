import io
import logging
import re
import time
import unicodedata
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from config import (
    DOG_BREED,
    DOG_NAME,
    GEMINI_API_KEY,
    GEMINI_EDITOR_ENABLED,
    GEMINI_FALLBACK_MODELS,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_PROOFREADER_ENABLED,
    GEMINI_THINKING_LEVEL,
)


LOGGER = logging.getLogger(__name__)
MAX_MEDIA_FOR_AI = 12
BLOCKED_HASHTAGS = {
    "explore",
    "explorepage",
    "follow",
    "foryou",
    "foryoupage",
    "fyp",
    "instagood",
    "reels",
    "reelsinstagram",
    "viral",
}
WRONG_BREED_REPLACEMENTS = (
    (re.compile(r"\bpudlem\s+toy\b", flags=re.IGNORECASE), "pudlem miniaturowym"),
    (re.compile(r"\bpudla\s+toy\b", flags=re.IGNORECASE), "pudla miniaturowego"),
    (re.compile(r"\bpudlowi\s+toy\b", flags=re.IGNORECASE), "pudlowi miniaturowemu"),
    (re.compile(r"\bpudlu\s+toy\b", flags=re.IGNORECASE), "pudlu miniaturowym"),
    (re.compile(r"\bpudel(?:ek|ka)?\s*toy\b", flags=re.IGNORECASE), "pudel miniaturowy"),
    (re.compile(r"\btoy\s+poodle\b", flags=re.IGNORECASE), "pudel miniaturowy"),
)
BREED_MENTION_PATTERN = re.compile(
    r"\bpud(?:el|la|lem|lowi|lu)\s+miniaturow\w*",
    flags=re.IGNORECASE,
)


class ReelContent(BaseModel):
    """Ustrukturyzowany plan jednej rolki."""

    voiceover: str = Field(description="Naturalny tekst lektora w pierwszej osobie psa.")
    headline: str = Field(description="Hook na pierwsze 1-2 sekundy, 3-7 słów.")
    cover_title: str = Field(description="Tytuł osobnej okładki, maksymalnie 4 słowa.")
    caption_body: str = Field(description="Opis posta w pierwszej osobie psa, bez hashtagów.")
    hashtags: list[str] = Field(description="Od 3 do 5 niepersonalnych hashtagów tematycznych.")
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


def _search_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    return "".join(character for character in value.lower() if character.isascii() and character.isalnum())


def _correct_breed(text: str) -> str:
    corrected = text
    for pattern, replacement in WRONG_BREED_REPLACEMENTS:
        corrected = pattern.sub(replacement, corrected)
    return corrected


def _clean_cover_title(text: str) -> str:
    words = text.strip().split()
    if len(words) > 4 and words[0].casefold() in {"mój", "moja", "moje", "nasz", "nasza"}:
        words = words[1:]
    return " ".join(words[:4]).upper()


def _normalise_hashtags(values: list[str]) -> list[str]:
    dog_key = _search_key(DOG_NAME)
    breed_tag = "#" + _search_key(DOG_BREED)
    required = [breed_tag, "#pies"]
    candidates: list[str] = []

    for hashtag in values:
        key = _search_key(hashtag.strip().lstrip("#"))
        clean = "#" + key
        if not key or key in BLOCKED_HASHTAGS:
            continue
        if dog_key and dog_key in key:
            continue
        if "toy" in key and "pudel" in key:
            continue
        if key not in {_search_key(item) for item in candidates}:
            candidates.append(clean)

    hashtags: list[str] = []
    for hashtag in [*required, *candidates, "#psiezycie", "#pudel"]:
        key = _search_key(hashtag)
        if key and key not in {_search_key(item) for item in hashtags}:
            hashtags.append(hashtag)
        if len(hashtags) == 5:
            break
    return hashtags


def _normalise_content(content: ReelContent, media_count: int) -> dict:
    order: list[int] = []
    for index in content.asset_order:
        if 0 <= index < media_count and index not in order:
            order.append(index)
    order.extend(index for index in range(media_count) if index not in order)

    voiceover = _correct_breed(content.voiceover.strip())
    caption_body = _correct_breed(content.caption_body.strip())
    alt_text = _correct_breed(content.alt_text.strip())
    hashtags = _normalise_hashtags(content.hashtags)
    caption = caption_body + "\n\n" + " ".join(hashtags)

    return {
        "lektor": voiceover,
        "naglowek": content.headline.strip().upper(),
        "cover_title": _clean_cover_title(content.cover_title),
        "caption_body": caption_body,
        "hashtags": hashtags,
        "caption": caption,
        "alt_text": alt_text,
        "asset_order": order,
    }


def _has_first_person(text: str) -> bool:
    markers = re.compile(
        r"\b(?:ja|jestem|mam|mnie|mi|mną|mój|moja|moje|mojego|mojej|"
        r"chcę|mogę|muszę|nie\s+wiem|\w+(?:łem|łam))\b",
        flags=re.IGNORECASE,
    )
    return bool(markers.search(text))


def _content_quality_issues(
    content: ReelContent,
    *,
    allow_breed_in_narrative: bool = True,
) -> list[str]:
    issues: list[str] = []
    narrative = f"{content.voiceover}\n{content.caption_body}"
    third_person_name = re.compile(
        rf"(?:^|[.!?]\s+|\n)\s*(?:pudel\s+)?{re.escape(DOG_NAME)}\s+",
        flags=re.IGNORECASE,
    )
    if third_person_name.search(narrative):
        issues.append("narrator jest przedstawiony w trzeciej osobie")
    if not _has_first_person(content.voiceover):
        issues.append("voiceover nie zawiera wyraźnej pierwszej osoby")
    if not _has_first_person(content.caption_body):
        issues.append("caption_body nie zawiera wyraźnej pierwszej osoby")
    if re.search(r"\btoy\b|#?pudeltoy\b", narrative + " " + " ".join(content.hashtags), re.I):
        issues.append("w treści występuje błędne określenie toy")
    breed_mentions = len(BREED_MENTION_PATTERN.findall(narrative))
    if not allow_breed_in_narrative and breed_mentions:
        issues.append("rasa nie jest związana z tematem, ale pojawia się w narracji")
    elif breed_mentions > 1:
        issues.append(f"rasa jest powtórzona {breed_mentions} razy zamiast maksymalnie raz")

    voiceover_words = len(re.findall(r"\b\w+[’'-]?\w*\b", content.voiceover, flags=re.UNICODE))
    if not 24 <= voiceover_words <= 45:
        issues.append(f"voiceover ma {voiceover_words} słów zamiast 24-45")
    caption_length = len(content.caption_body.strip())
    if not 250 <= caption_length <= 600:
        issues.append(f"caption_body ma {caption_length} znaków zamiast 250-600")
    return issues


def _breed_is_relevant(nazwa: str, opis: str) -> bool:
    source = f"{nazwa} {opis}".casefold()
    return bool(
        re.search(
            r"pudel|rasa|miniaturow|sier[śs][ćc]|fryz|strzy|trym|czes|"
            r"groom|wielko[śs][ćc]|rozmiar",
            source,
        )
    )


def _finish_reason(response: object) -> str:
    try:
        candidates = getattr(response, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        return getattr(reason, "value", None) or str(reason or "unknown")
    except Exception:
        return "unknown"


def _parse_content(response: object) -> ReelContent:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed if isinstance(parsed, ReelContent) else ReelContent.model_validate(parsed)

    response_text = getattr(response, "text", None) or ""
    if not response_text.strip():
        raise ValueError(f"Gemini zwrócił pustą odpowiedź (finish_reason={_finish_reason(response)}).")
    try:
        return ReelContent.model_validate_json(response_text)
    except Exception as exc:
        # Nie dołączamy odpowiedzi do wyjątku: może zawierać dane z dziennika.
        raise ValueError(
            "Gemini zwrócił niepełny lub niepoprawny JSON "
            f"(finish_reason={_finish_reason(response)}, znaki={len(response_text)})."
        ) from exc


def _parse_response(response: object, media_count: int) -> dict:
    """Kompatybilny helper używany także przez testy."""
    return _normalise_content(_parse_content(response), media_count)


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


def _request_content(
    client: genai.Client,
    models: list[str],
    inputs: list[object],
    *,
    temperature: float,
    stage: str,
    thinking_level: types.ThinkingLevel | None = None,
) -> ReelContent:
    errors: list[str] = []
    retries = max(1, GEMINI_MAX_RETRIES)
    for model_name in models:
        for attempt in range(1, retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=inputs,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ReelContent,
                        temperature=temperature,
                        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=thinking_level or _thinking_level(),
                            include_thoughts=False,
                        ),
                    ),
                )
                return _parse_content(response)
            except Exception as exc:
                message = f"{model_name}, próba {attempt}/{retries}: {exc}"
                LOGGER.warning("Gemini (%s) nie wygenerował treści: %s", stage, message)
                errors.append(message)
                if _is_non_retryable_model_error(exc):
                    break
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"Nie udał się etap Gemini: {stage}. " + " | ".join(errors))


def _editor_prompt(
    draft: ReelContent,
    nazwa: str,
    opis: str,
    media_count: int,
    allow_breed_in_narrative: bool,
    quality_feedback: list[str] | None = None,
) -> str:
    feedback = "\n".join(f"- {issue}" for issue in (quality_feedback or []))
    feedback_section = (
        f"\nBŁĘDY WYKRYTE AUTOMATYCZNIE — wszystkie muszą zostać poprawione:\n{feedback}\n"
        if feedback
        else ""
    )
    breed_instruction = (
        f"Możesz użyć określenia „{DOG_BREED}” najwyżej raz łącznie w voiceover i caption_body."
        if allow_breed_in_narrative
        else (
            f"Temat nie dotyczy rasy. Nie używaj określenia „{DOG_BREED}” ani słowa "
            "„pudel” w voiceover lub caption_body; rasa pozostaje tylko w alt_text i hashtagu."
        )
    )
    return f"""
Jesteś bezkompromisowym polskim redaktorem treści na Instagram. Popraw poniższy
szkic, ale nie zmieniaj faktów, kolejności mediów ani sensu historii.

NIEPODWAŻALNY PROFIL BOHATERA:
- imię: {DOG_NAME}
- rasa: {DOG_BREED}; nigdy pudel toy ani toy poodle
- narrator voiceover i caption_body: wyłącznie pierwsza osoba liczby pojedynczej psa

ŹRÓDŁO:
- nazwa wydarzenia: {nazwa or 'brak'}
- opis: {opis or 'brak'}
- liczba mediów: {media_count}

SZKIC JSON:
{draft.model_dump_json(ensure_ascii=False)}
{feedback_section}

Lista kontrolna przed odpowiedzią:
1. Każde zdanie voiceover i caption_body wypowiada {DOG_NAME} jako „ja”. Nie zaczynaj
   od „{DOG_NAME}...”, „Pudel {DOG_NAME}...” ani opisu bohatera w trzeciej osobie.
   Poprawnie: „Pierwszy raz odwiedziłem babcię...”.
   Błędnie: „Pudel {DOG_NAME} pierwszy raz odwiedza babcię...”.
2. „{DOG_BREED}” to fakt profilu, nie slogan. Usuń każde „toy”. {breed_instruction}
3. Zachowaj naturalną, potoczną polszczyznę, jedną mini-historię i maksymalnie 3 emoji.
4. Zweryfikuj szkic względem opisu i dołączonych mediów. Usuń każdy fakt, emocję,
   intencję i rezultat, którego nie potwierdza opis albo obraz. Nie zakładaj, że psy
   się polubiły, „przełamały lody”, zaakceptowały lub nauczyły czegoś, jeśli źródło
   tego nie mówi. Gdy brak finału, zakończ trafną obserwacją, nie wymyślonym sukcesem.
5. caption_body ma mieć 250-600 znaków, naturalne słowa kluczowe w pierwszych dwóch
   zdaniach i na końcu jedno łatwe, konkretne pytanie.
6. Podaj 3-5 hashtagów bez imienia {DOG_NAME} i bez tagów brandingowych. Zastosuj miks:
   rasa, szersza kategoria psów i 1-3 tagi ściśle związane z tą historią. Zakazane:
   #pudeltoy, #fyp, #viral, #reels, #instagood.
7. headline i cover_title mają być konkretne i zgodne z wydarzeniem. cover_title ma
   maksymalnie 4 słowa; usuń zbędne „mój/moja”, jeśli bez niego sens się nie zmienia.
8. Zrób korektę polszczyzny: składnia, odmiana, związki frazeologiczne, interpunkcja
   i powtórzenia. Nie zostawiaj urwanych konstrukcji typu „postanowiłem nie dłużny”.

Zwróć wyłącznie kompletny obiekt zgodny ze schematem.
"""


def _proofreader_prompt(
    content: ReelContent,
    nazwa: str,
    opis: str,
    media_count: int,
    allow_breed_in_narrative: bool,
    quality_feedback: list[str] | None = None,
) -> str:
    evidence_rule = (
        "Dołączono media: wolno zachować dodatkowy szczegół tylko wtedy, gdy jest na nich jednoznacznie widoczny."
        if media_count
        else "Nie dołączono mediów: każdy konkretny szczegół musi wynikać dosłownie ze ŹRÓDŁA."
    )
    breed_rule = (
        f"Rasa „{DOG_BREED}” może wystąpić najwyżej raz w voiceover + caption_body."
        if allow_breed_in_narrative
        else f"Usuń rasę „{DOG_BREED}” i słowo „pudel” z voiceover oraz caption_body."
    )
    feedback = "\n".join(f"- {issue}" for issue in (quality_feedback or []))
    feedback_section = (
        f"\nBŁĘDY Z POPRZEDNIEJ KOREKTY — popraw wszystkie:\n{feedback}\n"
        if feedback
        else ""
    )
    return f"""
Jesteś polskim korektorem końcowym. Popraw wyłącznie błędy językowe i niezręczne
sformułowania w poniższym JSON-ie. Nie dodawaj faktów, emocji, skutków ani nowych
żartów. Zachowaj pierwszą osobę psa, strukturę JSON, asset_order i sens wypowiedzi.

Profil: {DOG_NAME}, {DOG_BREED}.
Źródło: {nazwa or 'brak'} — {opis or 'brak'}.
Zasada dowodowa: {evidence_rule}
Zasada rasy: {breed_rule}

Sprawdź szczególnie:
- poprawną odmianę i pełne konstrukcje zdaniowe;
- zgodność podmiotu i orzeczenia;
- naturalne związki frazeologiczne;
- brak powtórzeń imienia i rasy (rasa najwyżej raz w voiceover + caption_body);
- zachowanie długości: voiceover 24-45 słów, caption_body 250-600 znaków;
- literalne oparcie każdego twierdzenia na źródle lub obrazie. Usuń niewskazane
  działania i cechy, np. warczenie, szczekanie, reakcję „od progu”, „ulubiony” dywan,
  aferę, rewanż, stres albo zgodę, jeśli dokładnie tego nie potwierdza źródło;
- jedno konkretne pytanie na końcu caption_body;
- maksymalnie 4 słowa w cover_title.

TREŚĆ DO KOREKTY:
{content.model_dump_json(ensure_ascii=False)}
{feedback_section}

Zwróć wyłącznie kompletny, poprawiony obiekt zgodny ze schematem.
"""


def generate_reel_content(
    nazwa: str,
    opis: str,
    media_paths: list[str] | None = None,
) -> dict:
    """Generuje i redaguje plan rolki na podstawie wpisu oraz mediów."""
    if not GEMINI_API_KEY:
        raise RuntimeError("Brak GEMINI_API_KEY w pliku .env.")

    paths = [Path(path) for path in (media_paths or [])][:MAX_MEDIA_FOR_AI]
    media_count = len(paths)
    allow_breed_in_narrative = _breed_is_relevant(nazwa, opis)
    breed_narration_instruction = (
        f"Możesz naturalnie użyć określenia „{DOG_BREED}” najwyżej raz łącznie w voiceover i caption_body."
        if allow_breed_in_narrative
        else (
            f"Temat nie dotyczy rasy: nie używaj „{DOG_BREED}” ani słowa „pudel” "
            "w voiceover i caption_body. Rasa pojawi się w alt_text i hashtagu."
        )
    )
    prompt = f"""
Jesteś polskim strategiem Instagram Reels i scenarzystą konta psa.

NIEPODWAŻALNY PROFIL BOHATERA:
- imię: {DOG_NAME}
- rasa: {DOG_BREED}; nigdy pudel toy ani toy poodle
- charakter: pogodny, inteligentny i lekko zadziorny
- narrator voiceover i caption_body: wyłącznie pierwsza osoba liczby pojedynczej psa

Dane z dziennika:
- nazwa wydarzenia: {nazwa or 'brak'}
- opis wydarzenia: {opis or 'brak'}
- liczba mediów: {media_count}

Dołączone media mają indeksy 0..{max(media_count - 1, 0)}. Obraz i opis są źródłem
prawdy. Nie wymyślaj zachowań, miejsc, jedzenia, emocji ani rezultatu wydarzenia.
Nie zakładaj, że spotkanie zakończyło się zgodą, akceptacją lub „przełamaniem lodów”.
Jeśli czegoś nie wiadomo, użyj neutralnej obserwacji zamiast dopisywać fakt.

Przygotuj jedną spójną rolkę:
1. voiceover: 24-45 słów, około 7-14 sekund, od pierwszego do ostatniego zdania
   mówi {DOG_NAME} jako „ja”. Nie zaczynaj od „{DOG_NAME}...” ani „Pudel {DOG_NAME}...”.
   Jedna mini-historia: hook, rozwinięcie, puenta. Bez żebrania o lajki.
2. headline: 3-7 słów, konkretny hook bez clickbaitu.
3. cover_title: maksymalnie 4 słowa, czytelny poza kontekstem rolki. Bez zbędnego
   „mój/moja”, jeśli tytuł jest równie jasny bez tego słowa.
4. caption_body: 250-600 znaków, również w pierwszej osobie psa. Użyj naturalnie
   słowa „pies” i tematu historii w pierwszych dwóch zdaniach. {breed_narration_instruction}
   Maksymalnie 3 emoji. Na końcu jedno łatwe pytanie.
5. hashtags: 3-5 niepersonalnych tagów. Bez imienia {DOG_NAME}. Miks: rasa, szersza
   kategoria psów i 1-3 tagi tematyczne. Bez #pudeltoy, #fyp, #viral i #reels.
6. alt_text: rzeczowy opis wyłącznie tego, co faktycznie widać.
7. asset_order: każdy poprawny indeks dokładnie raz; najmocniejszy kadr pierwszy.
"""

    media_parts: list[types.Part] = []
    for path in paths:
        try:
            media_parts.append(_media_preview(path))
        except Exception as exc:
            LOGGER.warning("Nie udało się przygotować podglądu %s: %s", path, exc)
    inputs: list[object] = [prompt, *media_parts]

    requested_models = [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]
    models = list(dict.fromkeys(model for model in requested_models if model))
    client = genai.Client(api_key=GEMINI_API_KEY)
    draft = _request_content(client, models, inputs, temperature=0.65, stage="szkic")

    final_content = draft
    if GEMINI_EDITOR_ENABLED:
        editorial_draft = draft
        quality_feedback = _content_quality_issues(
            editorial_draft,
            allow_breed_in_narrative=allow_breed_in_narrative,
        )
        for editorial_round in range(1, 3):
            try:
                final_content = _request_content(
                    client,
                    models,
                    [
                        _editor_prompt(
                            editorial_draft,
                            nazwa,
                            opis,
                            media_count,
                            allow_breed_in_narrative,
                            quality_feedback,
                        ),
                        *media_parts,
                    ],
                    temperature=0.15,
                    stage=f"redakcja {editorial_round}/2",
                    thinking_level=types.ThinkingLevel.LOW,
                )
            except Exception as exc:
                LOGGER.warning("Redakcja Gemini %s/2 nie powiodła się: %s", editorial_round, exc)
                continue

            quality_feedback = _content_quality_issues(
                final_content,
                allow_breed_in_narrative=allow_breed_in_narrative,
            )
            if not quality_feedback:
                break
            LOGGER.warning(
                "Kontrola jakości po redakcji %s/2: %s",
                editorial_round,
                "; ".join(quality_feedback),
            )
            editorial_draft = final_content

    if GEMINI_PROOFREADER_ENABLED:
        proofread_draft = final_content
        proofread_feedback: list[str] = []
        for proofread_round in range(1, 3):
            try:
                final_content = _request_content(
                    client,
                    models,
                    [
                        _proofreader_prompt(
                            proofread_draft,
                            nazwa,
                            opis,
                            media_count,
                            allow_breed_in_narrative,
                            proofread_feedback,
                        ),
                        *media_parts,
                    ],
                    temperature=0.1,
                    stage=f"korekta językowa {proofread_round}/2",
                    thinking_level=types.ThinkingLevel.MEDIUM,
                )
            except Exception as exc:
                LOGGER.warning("Końcowa korekta językowa %s/2 nie powiodła się: %s", proofread_round, exc)
                continue

            proofread_feedback = _content_quality_issues(
                final_content,
                allow_breed_in_narrative=allow_breed_in_narrative,
            )
            if not proofread_feedback:
                break
            LOGGER.warning(
                "Kontrola jakości po korekcie %s/2: %s",
                proofread_round,
                "; ".join(proofread_feedback),
            )
            proofread_draft = final_content

    if GEMINI_EDITOR_ENABLED or GEMINI_PROOFREADER_ENABLED:
        remaining_issues = _content_quality_issues(
            final_content,
            allow_breed_in_narrative=allow_breed_in_narrative,
        )
        if remaining_issues:
            raise RuntimeError(
                "Treść nie przeszła kontroli jakości: " + "; ".join(remaining_issues)
            )

    return _normalise_content(final_content, media_count)
