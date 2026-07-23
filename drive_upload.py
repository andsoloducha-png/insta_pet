import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as UserCredentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from config import (
    GOOGLE_DRIVE_AUTH_MODE,
    GOOGLE_DRIVE_OAUTH_CLIENT_FILE,
    GOOGLE_DRIVE_TOKEN_FILE,
    SERVICE_ACCOUNT_FILE,
)


LOGGER = logging.getLogger(__name__)
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


class DriveUploadError(RuntimeError):
    """Czytelny błąd konfiguracji lub wysyłania plików na Google Drive."""


@dataclass(frozen=True)
class DriveFile:
    file_id: str
    name: str
    url: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.file_id, "name": self.name, "url": self.url}


def _escape_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _load_oauth_credentials() -> UserCredentials:
    token_path = Path(GOOGLE_DRIVE_TOKEN_FILE)
    client_path = Path(GOOGLE_DRIVE_OAUTH_CLIENT_FILE)
    credentials = None

    if token_path.exists():
        try:
            credentials = UserCredentials.from_authorized_user_file(token_path, [DRIVE_SCOPE])
        except (OSError, ValueError) as exc:
            raise DriveUploadError(
                f"Nie można odczytać tokenu Google Drive: {token_path}. "
                "Usuń uszkodzony plik tokenu i uruchom aplikację ponownie."
            ) from exc

    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except Exception as exc:
            raise DriveUploadError(
                "Nie udało się odświeżyć logowania Google Drive. "
                f"Usuń plik {token_path} i zaloguj się ponownie."
            ) from exc

    if not credentials or not credentials.valid:
        if not client_path.exists():
            raise DriveUploadError(
                "Brakuje pliku klienta OAuth Google Drive: "
                f"{client_path}. Pobierz dane typu „Aplikacja komputerowa” "
                "z Google Cloud Console i zapisz je pod tą ścieżką."
            )
        try:
            flow = InstalledAppFlow.from_client_secrets_file(client_path, [DRIVE_SCOPE])
            credentials = flow.run_local_server(port=0, open_browser=True)
        except Exception as exc:
            raise DriveUploadError("Nie udało się zalogować do Google Drive przez OAuth.") from exc

    token_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = token_path.with_suffix(token_path.suffix + ".tmp")
    temporary.write_text(credentials.to_json(), encoding="utf-8")
    temporary.replace(token_path)
    return credentials


def _load_credentials():
    if GOOGLE_DRIVE_AUTH_MODE == "oauth":
        return _load_oauth_credentials()
    if GOOGLE_DRIVE_AUTH_MODE == "service_account":
        try:
            return ServiceAccountCredentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE,
                scopes=[DRIVE_SCOPE],
            )
        except (OSError, ValueError) as exc:
            raise DriveUploadError(
                f"Nie można odczytać konta serwisowego: {SERVICE_ACCOUNT_FILE}"
            ) from exc
    raise DriveUploadError(
        "GOOGLE_DRIVE_AUTH_MODE musi mieć wartość 'oauth' albo 'service_account'."
    )


class GoogleDriveUploader:
    def __init__(self):
        credentials = _load_credentials()
        self.service = build(
            "drive",
            "v3",
            credentials=credentials,
            cache_discovery=False,
        )

    def verify_folder(self, folder_id: str) -> dict:
        if not folder_id:
            raise DriveUploadError("Brakuje GOOGLE_DRIVE_OUTPUT_FOLDER_ID w pliku .env.")
        try:
            folder = (
                self.service.files()
                .get(
                    fileId=folder_id,
                    fields="id,name,mimeType,driveId,capabilities(canAddChildren)",
                    supportsAllDrives=True,
                )
                .execute()
            )
        except HttpError as exc:
            raise DriveUploadError(
                "Nie udało się otworzyć docelowego folderu Google Drive. "
                "Sprawdź jego identyfikator i uprawnienia."
            ) from exc
        if folder.get("mimeType") != FOLDER_MIME_TYPE:
            raise DriveUploadError("GOOGLE_DRIVE_OUTPUT_FOLDER_ID nie wskazuje folderu.")
        if not folder.get("capabilities", {}).get("canAddChildren"):
            raise DriveUploadError("Używane konto nie może dodawać plików do folderu Google Drive.")
        if GOOGLE_DRIVE_AUTH_MODE == "service_account" and not folder.get("driveId"):
            raise DriveUploadError(
                "Konto serwisowe nie może wysyłać plików do zwykłego „Mojego dysku”. "
                "Użyj GOOGLE_DRIVE_AUTH_MODE=oauth albo folderu na Dysku współdzielonym."
            )
        return folder

    def _find_child(self, parent_id: str, name: str, mime_type: str | None = None) -> dict | None:
        escaped_name = _escape_query_value(name)
        query = f"'{parent_id}' in parents and name = '{escaped_name}' and trashed = false"
        if mime_type:
            query += f" and mimeType = '{_escape_query_value(mime_type)}'"
        try:
            response = (
                self.service.files()
                .list(
                    q=query,
                    fields="files(id,name,mimeType,webViewLink)",
                    pageSize=10,
                    spaces="drive",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
        except HttpError as exc:
            raise DriveUploadError(f"Nie udało się wyszukać {name!r} na Google Drive.") from exc
        files = response.get("files", [])
        return files[0] if files else None

    def ensure_folder(self, parent_id: str, name: str) -> DriveFile:
        existing = self._find_child(parent_id, name, FOLDER_MIME_TYPE)
        if existing:
            file_id = existing["id"]
            return DriveFile(
                file_id=file_id,
                name=name,
                url=existing.get("webViewLink") or f"https://drive.google.com/drive/folders/{file_id}",
            )
        try:
            created = (
                self.service.files()
                .create(
                    body={"name": name, "mimeType": FOLDER_MIME_TYPE, "parents": [parent_id]},
                    fields="id,name,webViewLink",
                    supportsAllDrives=True,
                )
                .execute()
            )
        except HttpError as exc:
            raise DriveUploadError(f"Nie udało się utworzyć folderu {name!r} na Google Drive.") from exc
        file_id = created["id"]
        return DriveFile(
            file_id=file_id,
            name=name,
            url=created.get("webViewLink") or f"https://drive.google.com/drive/folders/{file_id}",
        )

    def sync_file(self, local_path: str | Path, folder_id: str) -> DriveFile:
        path = Path(local_path)
        if not path.is_file():
            raise DriveUploadError(f"Brakuje pliku do wysłania: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        existing = self._find_child(folder_id, path.name)
        media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)
        try:
            if existing:
                uploaded = (
                    self.service.files()
                    .update(
                        fileId=existing["id"],
                        media_body=media,
                        fields="id,name,webViewLink",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
            else:
                uploaded = (
                    self.service.files()
                    .create(
                        body={"name": path.name, "parents": [folder_id]},
                        media_body=media,
                        fields="id,name,webViewLink",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
        except HttpError as exc:
            message = str(exc)
            if "storageQuotaExceeded" in message and GOOGLE_DRIVE_AUTH_MODE == "service_account":
                raise DriveUploadError(
                    "Konto serwisowe nie ma przestrzeni na pliki. "
                    "Użyj OAuth albo Dysku współdzielonego Google Workspace."
                ) from exc
            raise DriveUploadError(f"Nie udało się wysłać pliku {path.name!r}.") from exc
        file_id = uploaded["id"]
        return DriveFile(
            file_id=file_id,
            name=uploaded.get("name", path.name),
            url=uploaded.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view",
        )

    def sync_folder(
        self,
        local_folder: str | Path,
        parent_id: str,
        exclude_names: set[str] | None = None,
    ) -> dict:
        local_path = Path(local_folder)
        if not local_path.is_dir():
            raise DriveUploadError(f"Brakuje lokalnego folderu rolki: {local_path}")
        destination = self.ensure_folder(parent_id, local_path.name)
        excluded = exclude_names or set()
        files: dict[str, dict[str, str]] = {}
        for path in sorted(local_path.iterdir(), key=lambda item: item.name.casefold()):
            if path.is_file() and path.name not in excluded:
                LOGGER.info("Google Drive: wysyłam %s", path.name)
                uploaded = self.sync_file(path, destination.file_id)
                files[path.name] = uploaded.as_dict()
        return {
            "folder_id": destination.file_id,
            "folder_url": destination.url,
            "files": files,
        }
