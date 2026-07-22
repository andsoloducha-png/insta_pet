import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Zmienna {name} musi być liczbą całkowitą, a jest: {value!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true", "tak", "yes", "on"}:
        return True
    if value in {"0", "false", "nie", "no", "off"}:
        return False
    raise ValueError(f"Zmienna {name} musi być wartością true/false, a jest: {value!r}")


# Dostawca treści AI: "gemini" lub w przyszłości "openai".
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
GEMINI_FALLBACK_MODELS = tuple(
    model.strip()
    for model in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite").split(",")
    if model.strip()
)
GEMINI_MAX_RETRIES = _env_int("GEMINI_MAX_RETRIES", 3)
GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 4096)
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "minimal").strip().lower()
GEMINI_EDITOR_ENABLED = _env_bool("GEMINI_EDITOR_ENABLED", True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()

# Stały profil bohatera zapobiega zmianie rasy i perspektywy między wpisami.
DOG_NAME = os.getenv("DOG_NAME", "Jogi").strip()
DOG_BREED = os.getenv("DOG_BREED", "pudel miniaturowy").strip()

# Google Sheets i plik konta usługi.
SHEET_URL = os.getenv("SHEET_URL", "").strip()
_service_account_value = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_FILE",
    "service_account.json",
).strip()
_service_account_path = Path(_service_account_value).expanduser()
if not _service_account_path.is_absolute():
    _service_account_path = BASE_DIR / _service_account_path
SERVICE_ACCOUNT_FILE = str(_service_account_path.resolve())

# Bezpłatna synteza mowy Microsoft Edge TTS.
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge").strip().lower()
TTS_VOICE = os.getenv("TTS_VOICE", "pl-PL-ZofiaNeural").strip()
TTS_FALLBACK_VOICES = tuple(
    voice.strip()
    for voice in os.getenv("TTS_FALLBACK_VOICES", "pl-PL-MarekNeural").split(",")
    if voice.strip()
)
TTS_MAX_RETRIES = _env_int("TTS_MAX_RETRIES", 2)
TTS_RATE = os.getenv("TTS_RATE", "+8%").strip()
TTS_PITCH = os.getenv("TTS_PITCH", "+12Hz").strip()

# Ustawienia wideo (Instagram Reels / TikTok).
VIDEO_WIDTH = _env_int("VIDEO_WIDTH", 1080)
VIDEO_HEIGHT = _env_int("VIDEO_HEIGHT", 1920)
VIDEO_FPS = _env_int("VIDEO_FPS", 30)
