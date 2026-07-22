# Generator rolek Jogiego

Samodzielny generator pionowych rolek 9:16 na podstawie wpisu z dziennika psa.
Gemini analizuje opis i media, bezpłatny Microsoft Edge TTS tworzy polski głos,
a lokalny renderer składa MP4, napisy, okładkę i opis Instagrama.

## Co obsługuje

- jedno zdjęcie, animowany GIF albo kilka linków w jednej komórce;
- zwykłe adresy HTTP(S), linki Google Drive i linki zapisane jako Markdown;
- automatyczną kolejność kadrów wskazaną przez Gemini;
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

Domyślna konfiguracja Gemini ogranicza poziom rozumowania do `minimal`, przeznacza
4096 tokenów na kompletną odpowiedź JSON i ponawia chwilowo uciętą odpowiedź do
3 razy. Modelem awaryjnym jest `gemini-3.5-flash-lite`; wszystkie te wartości
można zmienić w `.env` na podstawie `.env.example`.

Domyślnym darmowym głosem jest aktualnie `pl-PL-ZofiaNeural`. Przy chwilowym
braku audio skrypt ponawia syntezę, a następnie próbuje `pl-PL-MarekNeural`.
Wybrany faktycznie głos zapisuje się w manifeście rolki.

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

Test całego przepływu bez Google Sheets używa `output/temp_5.jpg`:

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

Ten test wysyła pomniejszony podgląd zdjęcia oraz opis do Gemini, a tekst lektora
do Microsoft Edge TTS. Nie należy używać go z materiałami, których nie wolno
przekazywać tym usługom.

Przetworzenie pierwszego oczekującego wpisu z arkusza:

```powershell
.\.venv\Scripts\python.exe main.py
```

Wyniki trafiają do `output/`: rolka MP4, okładka JPG, lektor MP3, napisy SRT,
opis TXT i manifest JSON. Manifest jest najlepszym, stabilnym punktem integracji
z aplikacją „Dziennik psa”.

## Integracja z Dziennikiem psa

Na obecnym etapie generator powinien pozostać osobnym projektem. Dziennik może
później wywoływać `process_entry(entry)` albo przekazywać zadanie przez kolejkę/API
i odczytywać manifest. Branch służy do rozwijania zmian w tym samym repozytorium,
nie do trwałego rozdzielania dwóch aplikacji.

Najbezpieczniejsza kolejność prac:

1. ustabilizować generator na reprezentatywnych materiałach;
2. utworzyć dla niego osobne repozytorium i wersjonować kontrakt manifestu;
3. dopiero potem dodać adapter w Dzienniku psa na osobnym branchu integracyjnym.
