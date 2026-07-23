# Generator rolek Jogiego

Samodzielny generator pionowych rolek 9:16 na podstawie wpisu z dziennika psa.
Gemini analizuje opis i media, wybrany dostawca tworzy polskiego lektora,
a lokalny renderer składa MP4, napisy, okładkę i opis Instagrama.

## Co obsługuje

- jedno zdjęcie, animowany GIF albo kilka linków w jednej komórce;
- zwykłe adresy HTTP(S), linki Google Drive i linki zapisane jako Markdown;
- automatyczną kolejność kadrów wskazaną przez Gemini;
- cztery dobierane do historii formaty montażu: `punchline`, `mini_story`,
  `comparison` i `diary_mood`;
- format 1080 × 1920, 30 FPS, H.264/AAC;
- delikatny ruch kadru, czytelny hook i napisy zsynchronizowane z lektorem;
- osobną okładkę JPG, opis z maksymalnie 5 hashtagami, tekst alternatywny i manifest JSON.

## Instalacja

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Uzupełnij `.env`. Wartość klucza może być bez cudzysłowów:

```dotenv
GEMINI_API_KEY=twoj_klucz
```

Cudzysłowy są potrzebne tylko wtedy, gdy wartość zawiera spacje lub znak `#`.
Plik `.env` oraz dane konta Google są wykluczone z Gita.
Kompletna checklista zmiany komputera i gotowy prompt dla nowej sesji Codexa
znajdują się w `PRZENOSINY_NA_NOWY_PC.md`.

Domyślna konfiguracja Gemini ogranicza poziom rozumowania do `minimal`, przeznacza
8192 tokeny na kompletną odpowiedź JSON i ponawia chwilowo uciętą odpowiedź do
3 razy. Modelem awaryjnym jest `gemini-3.5-flash-lite`; wszystkie te wartości
można zmienić w `.env` na podstawie `.env.example`.

Profil bohatera jest konfigurowany przez `DOG_NAME=Jogi`,
`DOG_BREED=pudel miniaturowy` i `DOG_PERSONALITY`. Domyślna osobowość to pewny
siebie, ciepły i lekko bezczelny urwis. Domyślnie jedna rolka wykonuje jedno
wywołanie API do przygotowania tekstu. `GEMINI_EDITOR_ENABLED=true` włącza osobny
etap redakcji, a `GEMINI_PROOFREADER_ENABLED=true` osobną korektę językową. Obie
flagi są domyślnie wyłączone, ponieważ tekst można poprawić ręcznie. Każda flaga
dodaje najwyżej jedno wywołanie API i nie blokuje rolki, gdy pozostaną uwagi. Lokalne
reguły nadal poprawiają nazwę rasy oraz porządkują kolejność mediów i hashtagi.

Rasa trafia zawsze do hashtagu i może trafić do alt textu. W narracji pojawia się
tylko wtedy, gdy sam temat dotyczy rasy, wielkości, sierści lub pielęgnacji. Przy
integracji z wieloma profilami Dziennik psa powinien przekazywać imię i rasę z
rekordu zwierzęcia; wartości z `.env` pozostaną ustawieniami domyślnymi.

Opis zawiera maksymalnie 5 niepersonalnych hashtagów: rasowy, szeroki `#pies` oraz
tagi związane z konkretną historią. Tagi z imieniem psa i ogólne tagi spamowe
(`fyp`, `viral`, `reels`) są automatycznie usuwane.

Gemini wybiera format w tym samym zapytaniu, które tworzy scenariusz. `punchline`
używa szybszych cięć i mocniejszego ruchu, `mini_story` prowadzi kadry sekwencyjnie
z przejściami, `comparison` zestawia materiały na podzielonym ekranie, a
`diary_mood` ma spokojniejszy ruch i miękkie przenikanie. Format wymagający kilku
materiałów nie zostanie użyty dla pojedynczego zdjęcia. Lokalna historia manifestów
zapobiega trzeciej rolce z rzędu w tym samym formacie.

Aktywny jest zawsze jeden dostawca głosu, wskazany przez `TTS_PROVIDER`. Do wyboru
są Gemini TTS `Achird`, własny głos ElevenLabs oraz awaryjny Edge TTS. Modele
zapasowe Gemini mogą zmieniać się po wyczerpaniu limitu, ale głos pozostaje ten sam.
Po narracji skrypt może dokleić autorski podpis dźwiękowy
`assets/jogi_signature_laugh.wav`, wycięty z zaakceptowanej próbki `warm`. Napisy
kończą się przed śmiechem. Podpis działa ze wszystkimi dostawcami i kontroluje go
niezależna flaga `TTS_SIGNATURE_LAUGH_ENABLED`.

### Przełączanie głosu

Dotychczasowy głos Gemini:

```dotenv
TTS_PROVIDER=gemini
TTS_SIGNATURE_LAUGH_ENABLED=true
```

Własny głos partnerki z ElevenLabs:

```dotenv
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=twoj_klucz
ELEVENLABS_VOICE_ID=id_utworzonego_glosu
ELEVENLABS_MODEL=eleven_multilingual_v2
TTS_SIGNATURE_LAUGH_ENABLED=true
```

Nie trzeba komentować ani usuwać ustawień drugiego głosu. Zmieniasz tylko jedną
wartość `TTS_PROVIDER` i uruchamiasz skrypt ponownie. Flaga śmiechu jest osobna:
`true` go dokleja, a `false` wyłącza. Ustawienia `ELEVENLABS_STABILITY`,
`ELEVENLABS_SIMILARITY_BOOST`, `ELEVENLABS_STYLE` i `ELEVENLABS_SPEED` mają bezpieczne
wartości początkowe w `.env.example`; najpierw warto sprawdzić je bez zmian.

Po przełączeniu wykonaj krótką próbę bez pobierania wiersza z arkusza:

```powershell
.\.venv\Scripts\python.exe tts_preview.py
```

Próbka trafi do `output/voice_test/`. ElevenLabs zwraca dokładne znaczniki czasu
znaków, z których generator buduje synchronizację napisów.

Po wyczerpaniu limitu modelu 2.5 skrypt przechodzi bez zbędnych ponowień na
`GEMINI_TTS_FALLBACK_MODELS` (domyślnie `gemini-3.1-flash-tts-preview`), nadal
używając głosu Achird. Tekst jest zapisywany etapowo w
`output/wiersz_<numer>/content_cache.json`,
więc awaria TTS nie powoduje ponownego naliczania generowania scenariusza.

Bezpłatny Edge TTS pozostaje dostawcą awaryjnym dostępnym po ustawieniu
`TTS_PROVIDER=edge`. Nie kopiuje ani nie wykorzystuje materiałów dźwiękowych CapCut.

Dostępne presety Edge TTS:

- `jogi_playful_soft` — domyślny i najbardziej naturalny;
- `jogi_playful` — szybszy, wyrazisty, ale nadal czytelny;
- `jogi_playful_wild` — najwyższy i najbardziej kreskówkowy.
- `jogi_urwis` — niższa, mniej kobieca barwa, wolniejsze frazy i krótkie rytmiczne pauzy.

Wariant wybiera `TTS_PRESET`. `TTS_EFFECTS_ENABLED=false` wyłącza obróbkę FFmpeg,
a `TTS_RATE` i `TTS_PITCH` pozwalają nadpisać parametry presetu. Przy chwilowym
braku audio skrypt ponawia syntezę, a następnie próbuje `pl-PL-MarekNeural`.
Wybrany głos i preset zapisują się w manifeście rolki.

Porównawcze próbki wszystkich wariantów można utworzyć poleceniem:

```powershell
.\.venv\Scripts\python.exe voice_preview.py
```

Pliki trafią do `output/voice_previews/`.

Naturalniejsze głosy Gemini TTS można porównać bez przełączania głównego generatora:

```powershell
.\.venv\Scripts\python.exe gemini_voice_preview.py
```

Skrypt używa tego samego `GEMINI_API_KEY`, modelu wskazanego przez
`GEMINI_TTS_MODEL` i zapisuje głosy Puck, Fenrir, Sadachbia oraz Achird w
`output/gemini_voice_previews/`.

Trzy warianty oryginalnego, krótkiego śmiechu na końcu próbki Achird generuje:

```powershell
.\.venv\Scripts\python.exe gemini_voice_preview.py --achird-laughs
```

Instrukcja zabrania imitowania istniejących postaci i wykonawców. Wybrany wariant
`warm` znajduje się w `assets/jogi_signature_laugh.wav` i jest dołączany do rolek.

## Konfiguracja arkusza

Skrypt oczekuje kolumn `Nazwa`, `Opis`, `Link` i `instagram`. W kolumnie
`instagram` wartość `tak` oznacza wpis oczekujący. Po poprawnym wygenerowaniu
zostanie ustawione `wygenerowane`, a przy błędzie `błąd`.

Po usunięciu przyczyny błędu ustaw w danym wierszu ponownie `tak`, aby skrypt
spróbował przetworzyć go jeszcze raz.

W jednej komórce `Link` można umieścić kilka adresów, rozdzielając je nową linią,
przecinkiem albo spacją, na przykład:

```text
https://example.com/jogi-1.jpg
https://example.com/jogi.gif, https://example.com/jogi-2.jpg
```

## Uruchamianie i testy

Testy lokalne nie wysyłają żadnych danych poza komputer:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Test całego przepływu bez Google Sheets używa `output/temp_5.jpg` albo
najnowszego medium z `output/wiersz_*/`:

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

Ten test wysyła pomniejszony podgląd zdjęcia oraz opis do Gemini, a tekst lektora
do wybranego dostawcy TTS. Nie należy używać go z materiałami, których nie wolno
przekazywać tym usługom. Wariant całkowicie lokalny wykorzystuje wcześniej utworzoną
próbkę Achird i nie wywołuje zewnętrznych usług:

```powershell
.\.venv\Scripts\python.exe smoke_test.py --offline
```

Przetworzenie pierwszego oczekującego wpisu z arkusza:

```powershell
.\.venv\Scripts\python.exe main.py
```

Każda rolka otrzymuje osobny katalog `output/wiersz_<numer>/`, zawierający
pobrane media, rolkę `reel.mp4`, okładkę `okladka.jpg`, lektora, napisy
`napisy.srt`, opis `opis.txt` i `manifest.json`. Manifest jest najlepszym,
stabilnym punktem integracji z aplikacją „Dziennik psa”.

### Automatyczny upload na Google Drive

Zwykły folder „Mój dysk” wymaga logowania OAuth. Konto serwisowe może wysyłać
pliki wyłącznie do Dysku współdzielonego Google Workspace, ponieważ nie ma
własnego limitu miejsca na pliki.

W Google Cloud Console włącz Google Drive API, skonfiguruj ekran zgody OAuth,
utwórz identyfikator klienta typu „Aplikacja komputerowa” i pobrany JSON zapisz
poza repozytorium jako:

```text
C:\Users\twoja_nazwa\.insta_jog_secrets\google_oauth_client.json
```

Następnie ustaw:

```dotenv
GOOGLE_DRIVE_UPLOAD_ENABLED=true
GOOGLE_DRIVE_OUTPUT_FOLDER_ID=ID_FOLDERU
GOOGLE_DRIVE_AUTH_MODE=oauth
GOOGLE_DRIVE_OAUTH_CLIENT_FILE=C:/Users/twoja_nazwa/.insta_jog_secrets/google_oauth_client.json
GOOGLE_DRIVE_TOKEN_FILE=C:/Users/twoja_nazwa/.insta_jog_secrets/google_drive_token.json
```

Przy pierwszym uruchomieniu `main.py` otworzy przeglądarkę i poprosi o zgodę.
Token zostanie zapisany poza repozytorium, a kolejne wysyłki będą automatyczne.
Każda rolka trafia do osobnego podfolderu `wiersz_<numer>`. Ponowne uruchomienie
aktualizuje pliki o tych samych nazwach zamiast tworzyć duplikaty.

## Integracja z Dziennikiem psa

Na obecnym etapie generator powinien pozostać osobnym projektem. Dziennik może
później wywoływać `process_entry(entry)` albo przekazywać zadanie przez kolejkę/API
i odczytywać manifest. Branch służy do rozwijania zmian w tym samym repozytorium,
nie do trwałego rozdzielania dwóch aplikacji.

Najbezpieczniejsza kolejność prac:

1. ustabilizować generator na reprezentatywnych materiałach;
2. utworzyć dla niego osobne repozytorium i wersjonować kontrakt manifestu;
3. dopiero potem dodać adapter w Dzienniku psa na osobnym branchu integracyjnym.
