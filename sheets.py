import re
import gspread
from google.oauth2.service_account import Credentials
from config import SHEET_URL, SERVICE_ACCOUNT_FILE

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.readonly'
]

def get_sheets_client():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def fetch_pending_entry():
    """Pobiera pierwszy wiersz, który ma w kolumnie 'instagram' wartość 'tak'."""
    gc = get_sheets_client()
    sh = gc.open_by_url(SHEET_URL)
    
    # 1. Otwieramy konkretną zakładkę po nazwie
    # (Upewnij się, że nazwa na dole w Google Sheets to dokładnie 'Wydarzenia')
    try:
        worksheet = sh.worksheet('Wydarzenia')
    except Exception:
        # Jeśli nie znajdzie zakładki 'Wydarzenia', pobierze pierwszą z brzegu
        worksheet = sh.get_worksheet(0)
    
    records = worksheet.get_all_records()
    headers = [h.lower().strip() for h in worksheet.row_values(1)]
    
    if 'instagram' not in headers:
        raise ValueError(
            f"Brak kolumny 'instagram' w pierwszym wierszu zakładki '{worksheet.title}'! "
            f"Znalezione kolumny: {headers}"
        )
        
    col_idx = headers.index('instagram') + 1

    for row_idx, row in enumerate(records, start=2):
        status = str(row.get('instagram', '')).strip().lower()
        if status == 'tak':
            return {
                'row_idx': row_idx,
                'col_idx': col_idx,
                'data': row.get('Data'),
                'nazwa': row.get('Nazwa'),
                'opis': row.get('Opis'),
                'link': row.get('Link'),
                'worksheet': worksheet
            }
    return None

def update_status(worksheet, row_idx, col_idx, new_status="ZROBIONE"):
    """Aktualizuje status w kolumnie instagram."""
    worksheet.update_cell(row_idx, col_idx, new_status)