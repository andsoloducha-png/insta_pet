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
GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 8192)
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "minimal").strip().lower()
GEMINI_EDITOR_ENABLED = _env_bool("GEMINI_EDITOR_ENABLED", False)
GEMINI_PROOFREADER_ENABLED = _env_bool("GEMINI_PROOFREADER_ENABLED", False)
GEMINI_TTS_MODEL = os.getenv(
    "GEMINI_TTS_MODEL",
    "gemini-2.5-flash-preview-tts",
).strip()
GEMINI_TTS_FALLBACK_MODELS = tuple(
    model.strip()
    for model in os.getenv(
        "GEMINI_TTS_FALLBACK_MODELS",
        "gemini-3.1-flash-tts-preview",
    ).split(",")
    if model.strip()
)
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Achird").strip()
GEMINI_TTS_MAX_RETRIES = _env_int("GEMINI_TTS_MAX_RETRIES", 3)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()

# Stały profil bohatera zapobiega zmianie rasy i perspektywy między wpisami.
DOG_NAME = os.getenv("DOG_NAME", "Jogi").strip()
DOG_BREED = os.getenv("DOG_BREED", "pudel miniaturowy").strip()
DOG_PERSONALITY = os.getenv(
    "DOG_PERSONALITY",
    "pewny siebie urwis: bystry, ciepły, lekko bezczelny i przekonany, że to on ustala zasady",
).strip()

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

# Gotowe podfoldery rolek mogą być automatycznie synchronizowane z Google Drive.
GOOGLE_DRIVE_UPLOAD_ENABLED = _env_bool("GOOGLE_DRIVE_UPLOAD_ENABLED", False)
GOOGLE_DRIVE_OUTPUT_FOLDER_ID = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID", "").strip()
GOOGLE_DRIVE_AUTH_MODE = os.getenv("GOOGLE_DRIVE_AUTH_MODE", "oauth").strip().lower()


def _resolve_google_secret_path(env_name: str, default_name: str) -> str:
    value = os.getenv(
        env_name,
        str(Path.home() / ".insta_jog_secrets" / default_name),
    ).strip()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path.resolve())


GOOGLE_DRIVE_OAUTH_CLIENT_FILE = _resolve_google_secret_path(
    "GOOGLE_DRIVE_OAUTH_CLIENT_FILE",
    "google_oauth_client.json",
)
GOOGLE_DRIVE_TOKEN_FILE = _resolve_google_secret_path(
    "GOOGLE_DRIVE_TOKEN_FILE",
    "google_drive_token.json",
)

# Bezpłatna synteza mowy Microsoft Edge TTS.
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "gemini").strip().lower()
TTS_VOICE = os.getenv("TTS_VOICE", "pl-PL-ZofiaNeural").strip()
TTS_FALLBACK_VOICES = tuple(
    voice.strip()
    for voice in os.getenv("TTS_FALLBACK_VOICES", "pl-PL-MarekNeural").split(",")
    if voice.strip()
)
TTS_MAX_RETRIES = _env_int("TTS_MAX_RETRIES", 2)
TTS_PRESET = os.getenv("TTS_PRESET", "jogi_playful_soft").strip().lower()
TTS_EFFECTS_ENABLED = _env_bool("TTS_EFFECTS_ENABLED", True)
# Puste wartości oznaczają tempo i ton z wybranego presetu. Ustawienie wartości
# pozwala świadomie nadpisać parametry, np. TTS_RATE=+5%.
TTS_RATE = os.getenv("TTS_RATE", "").strip()
TTS_PITCH = os.getenv("TTS_PITCH", "").strip()
TTS_SIGNATURE_LAUGH_ENABLED = _env_bool("TTS_SIGNATURE_LAUGH_ENABLED", True)
_signature_laugh_value = os.getenv(
    "TTS_SIGNATURE_LAUGH_FILE",
    "assets/jogi_signature_laugh.wav",
).strip()
_signature_laugh_path = Path(_signature_laugh_value).expanduser()
if not _signature_laugh_path.is_absolute():
    _signature_laugh_path = BASE_DIR / _signature_laugh_path
TTS_SIGNATURE_LAUGH_FILE = str(_signature_laugh_path.resolve())

# Ustawienia wideo (Instagram Reels / TikTok).
VIDEO_WIDTH = _env_int("VIDEO_WIDTH", 1080)
VIDEO_HEIGHT = _env_int("VIDEO_HEIGHT", 1920)
VIDEO_FPS = _env_int("VIDEO_FPS", 30)
