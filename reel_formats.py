from typing import Literal, get_args


ReelFormatId = Literal["punchline", "mini_story", "comparison", "diary_mood"]
REEL_FORMAT_IDS: tuple[str, ...] = get_args(ReelFormatId)
DEFAULT_REEL_FORMAT: ReelFormatId = "punchline"

FORMAT_DESCRIPTIONS: dict[str, str] = {
    "punchline": "Jedno mocne zdarzenie: szybki hook, krótki kontekst i celna puenta.",
    "mini_story": "Sekwencja kilku kadrów: początek, komplikacja i reakcja psa.",
    "comparison": "Kontrast dwóch lub więcej kadrów, np. oczekiwania kontra rzeczywistość.",
    "diary_mood": "Spokojniejszy wpis z dziennika: obserwacja, nastrój i ciepłe zakończenie.",
}


def normalise_reel_format(value: object) -> ReelFormatId:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in REEL_FORMAT_IDS else DEFAULT_REEL_FORMAT  # type: ignore[return-value]


def eligible_reel_formats(media_count: int) -> tuple[ReelFormatId, ...]:
    if media_count >= 2:
        return ("punchline", "mini_story", "comparison", "diary_mood")
    return ("punchline", "diary_mood")


def choose_reel_format(
    requested: object,
    media_count: int,
    recent_formats: list[str] | tuple[str, ...] = (),
) -> ReelFormatId:
    """Wybiera format zgodny z materiałami i zapobiega trzeciej powtórce z rzędu."""
    eligible = eligible_reel_formats(media_count)
    selected = normalise_reel_format(requested)
    if selected not in eligible:
        selected = "mini_story" if media_count >= 2 else DEFAULT_REEL_FORMAT

    recent = [normalise_reel_format(item) for item in recent_formats if item]
    if len(recent) < 2 or recent[0] != selected or recent[1] != selected:
        return selected

    alternatives: dict[str, tuple[ReelFormatId, ...]] = {
        "punchline": ("mini_story", "diary_mood", "comparison"),
        "mini_story": ("comparison", "punchline", "diary_mood"),
        "comparison": ("punchline", "mini_story", "diary_mood"),
        "diary_mood": ("punchline", "mini_story", "comparison"),
    }
    return next((item for item in alternatives[selected] if item in eligible), selected)
