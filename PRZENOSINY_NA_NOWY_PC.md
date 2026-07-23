# Przeniesienie projektu i przełączanie głosu

## 1. Przygotowanie nowego komputera

```powershell
git clone https://github.com/andsoloducha-png/insta_pet.git
Set-Location insta_pet
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Jeśli PowerShell blokuje aktywację środowiska, można bez aktywowania wykonywać
polecenia przez `.\.venv\Scripts\python.exe`.

Przenieś prywatnie, poza Gitem:

- wartości z dotychczasowego `.env`;
- plik konta usługi Google używany przez arkusz;
- ewentualne pliki OAuth Google Drive, tylko jeśli kiedyś ponownie włączysz upload.

Najlepiej zapisać pliki Google w
`C:\Users\TWOJA_NAZWA\.insta_jog_secrets\` i podać ich pełne ścieżki w `.env`.
Nie wklejaj sekretów do rozmowy z Codexem i nie dodawaj `.env` ani plików JSON do
commita.

## 2. Utworzenie głosu partnerki w ElevenLabs

Utwórz Instant Voice Clone z czystego nagrania partnerki, nagranego za jej zgodą.
Dobra próbka to około 1–2 minuty spójnej, naturalnej polskiej wypowiedzi, bez muzyki,
pogłosu, innych osób i zmiany odległości od mikrofonu. Po utworzeniu głosu skopiuj:

- klucz API ElevenLabs;
- identyfikator głosu (Voice ID).

W `.env` wpisz:

```dotenv
ELEVENLABS_API_KEY=twoj_klucz
ELEVENLABS_VOICE_ID=id_glosu_partnerki
ELEVENLABS_MODEL=eleven_multilingual_v2
```

## 3. Przełączanie głosu

Gemini:

```dotenv
TTS_PROVIDER=gemini
```

ElevenLabs:

```dotenv
TTS_PROVIDER=elevenlabs
```

Śmiech jest niezależny od wybranego głosu:

```dotenv
TTS_SIGNATURE_LAUGH_ENABLED=true
```

Po każdej zmianie zapisz `.env` i uruchom proces ponownie. Najpierw zrób tani,
krótki test bez arkusza:

```powershell
.\.venv\Scripts\python.exe tts_preview.py
```

Jeżeli próbka brzmi dobrze, uruchom właściwe przetwarzanie:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Prompt dla Codexa na nowym komputerze

```text
Pracujemy w repozytorium insta_pet:
https://github.com/andsoloducha-png/insta_pet

To generator rolek dla profilu psa Jogi. Przeczytaj najpierw README.md,
PRZENOSINY_NA_NOWY_PC.md i .env.example. Projekt obsługuje jeden aktywny głos
wybierany przez TTS_PROVIDER=gemini albo TTS_PROVIDER=elevenlabs. Śmiech na końcu
kontroluje osobna flaga TTS_SIGNATURE_LAUGH_ENABLED.

Na tym komputerze pomóż mi uruchomić projekt po migracji. Najpierw sprawdź git
status, wersję Pythona i zależności. Sprawdź tylko, czy wymagane zmienne w .env
istnieją — nie wyświetlaj ich wartości ani sekretów. Nie commituj .env, kluczy,
plików konta Google ani źródłowego nagrania partnerki.

Jeśli skonfigurowałem ElevenLabs, najpierw uruchom testy jednostkowe, potem krótką
próbkę przez tts_preview.py. Zweryfikuj, czy plik audio powstał, provider w wyniku
to elevenlabs, napisy mają znaczniki czasu i opcjonalny śmiech zaczyna się dopiero
po narracji. Nie uruchamiaj main.py ani nie pobieraj wpisu z arkusza bez mojego
potwierdzenia, jeśli próbka głosu jeszcze nie została przeze mnie odsłuchana.
```
