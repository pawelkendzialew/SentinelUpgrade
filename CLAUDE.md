# CLAUDE.md — Instrukcja dla Claude Code
## Projekt: Sentinel-2 Super-Resolution Pipeline

---

## Czym jest ten projekt

Narzędzie do polepszania rozdzielczości zdjęć satelitarnych Sentinel-2.

**Pipeline:**
```
Copernicus (Sentinel-2 L2A)
    ↓  10 m/piksel  (oryginał)
SEN2SR  ×4  (neural network)
    ↓  2.5 m/piksel
Real-ESRGAN  ×2  (neural network)
    ↓  ~1.25 m/piksel  (wynik)
```

**Interfejs:** GUI w Tkinter — przycisk "Pobierz i polepszy", porównanie PRZED/PO.

**Wyniki:** folder `output/` — trzy pliki PNG z watermarkiem (oryginał, po SEN2SR, po ESRGAN).

---

## Struktura plików w tym folderze

```
projekt/
├── CLAUDE.md           ← ten plik (instrukcja dla Ciebie)
├── pipeline.py         ← cały algorytm (pobieranie + SEN2SR + ESRGAN)
├── gui.py              ← interfejs graficzny Tkinter
├── SEN2SR-main/        ← KOD ŹRÓDŁOWY SEN2SR (wgrany ręcznie)
│   ├── sen2sr/
│   │   ├── __init__.py
│   │   ├── utils.py        ← predict_large() - przetwarzanie dużych obrazów
│   │   ├── nonreference.py ← architektura modelu nonreference
│   │   ├── referencex2.py
│   │   ├── referencex4.py
│   │   └── models/
│   │       └── opensr_baseline/
│   │           ├── cnn.py   ← CNNSR / SPAB architektura
│   │           ├── swin.py
│   │           └── mamba.py
│   └── README.md           ← dokumentacja SEN2SR
├── models/             ← (tworzone automatycznie przy pierwszym uruchomieniu)
│   ├── SEN2SRLite_RGBN/    ← wagi SEN2SRLite pobierane z HuggingFace
│   └── realesr-general-x4v3.pth  ← wagi Real-ESRGAN pobierane z GitHub
└── output/             ← (tworzone automatycznie)
    ├── 1_original_10m.png
    ├── 2_sen2sr_2.5m.png
    └── 3_esrgan_1.25m.png
```

**WAŻNE:** Folder `SEN2SR-main/` zawiera kod źródłowy biblioteki sen2sr.
Przy modyfikowaniu modelu / architektury korzystaj z tych plików.
Biblioteka `sen2sr` musi być zainstalowana przez pip (patrz niżej) — używa tych samych plików.

---

## Instalacja środowiska

### Wymagania wstępne
- Python 3.10+
- conda lub venv
- Dostęp do internetu (pierwsze uruchomienie pobiera modele ~200MB)

### Kroki instalacji

```bash
# 1. Utwórz środowisko conda
conda create -n sentinel_sr python=3.11 -y
conda activate sentinel_sr

# 2. Zainstaluj PyTorch (CPU — działa na każdym komputerze)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Opcjonalnie z GPU (CUDA 12.1):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Zainstaluj sen2sr i zależności do pobierania danych
pip install sen2sr mlstac git+https://github.com/ESDS-Leipzig/cubo.git

# 4. Zainstaluj Real-ESRGAN
pip install basicsr realesrgan

# 5. Zainstaluj pozostałe
pip install Pillow numpy
```

### Jeśli masz problemy z Real-ESRGAN
Biblioteka jest opcjonalna — pipeline zadziała bez niej (krok 3 użyje interpolacji Lanczos).
```bash
# Alternatywna instalacja:
pip install realesrgan --no-deps
pip install basicsr facexlib gfpgan
```

### Weryfikacja instalacji
```bash
python -c "import sen2sr; import cubo; import mlstac; print('OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Uruchomienie

### GUI (zalecane)
```bash
python gui.py
```
Otwiera okno aplikacji. Wybierz miasto z presetu (lub wpisz własne lat/lon), kliknij **▶ POBIERZ I POLEPSZY**.

### Wiersz poleceń
```bash
python pipeline.py
```
Uruchamia pipeline dla domyślnego obszaru (Kraków) i zapisuje wyniki w `output/`.

---

## Jak działa kod — dla modyfikacji

### `pipeline.py` — algorytm

| Funkcja | Co robi |
|---|---|
| `download_sentinel2()` | Pobiera dane przez `cubo.create()` z Copernicus |
| `run_sen2sr()` | Uruchamia SEN2SRLite RGBN×4 przez `mlstac` |
| `run_realesrgan()` | Uruchamia Real-ESRGAN na wyjściu SEN2SR |
| `run_pipeline()` | Orchestracja + zapis PNG |
| `tensor_to_rgb_uint8()` | Konwersja tensora Sentinel → RGB z percentile stretch |

### Kanały Sentinel-2 używane w tym projekcie
```
B02 = Blue   (490 nm)
B03 = Green  (560 nm)
B04 = Red    (665 nm)
B08 = NIR    (842 nm)
```
Kolejność w tensorze: `[B04, B03, B02, B08]` = `[R, G, B, NIR]`.
Indeksy 0,1,2 to RGB (tak działa `tensor_to_rgb_uint8`).

### SEN2SR — ważne szczegóły
- Model: `SEN2SRLite/NonReference_RGBN_x4` — wejście 4-kanałowe (RGBN), wyjście ×4
- Wagi pobierane z HuggingFace przy pierwszym uruchomieniu przez `mlstac.download()`
- `predict_large()` (z `sen2sr/utils.py`) dzieli duże obrazy na kafelki 128×128px z overlapem
- Dla obrazów ≤128px uruchamia bezpośrednio bez kafelkowania

### `gui.py` — interfejs

Klasa `SentinelSRApp(tk.Tk)`. Pipeline uruchamia się w osobnym wątku (`threading.Thread`).
Callback `progress_cb(msg, pct)` aktualizuje pasek postępu przez `self.after(0, ...)`.

---

## Typowe problemy i rozwiązania

### "No scenes found" — brak danych
Zmień zakres dat. Dla Polski najlepsza widoczność: **maj–wrzesień**.
```python
start_date="2023-05-01", end_date="2023-08-31"
```

### Out of Memory przy dużych kafelkach
Zmniejsz `edge_size` z 512 na 256 lub 128.
Na CPU procesowanie 256×256 zajmuje ~2–5 min.

### Real-ESRGAN nie zainstalowany
Pipeline automatycznie przełącza się na Lanczos (krok 3 nadal działa, gorsza jakość).

### Pierwsze uruchomienie trwa długo
Modele są pobierane raz:
- SEN2SRLite: ~50MB (HuggingFace)
- Real-ESRGAN: ~64MB (GitHub releases)

---

## Cel projektu i kierunek rozwoju

### v1 (obecna wersja)
- ✅ GUI z preselekcją miast polskich
- ✅ Automatyczne pobieranie Sentinel-2 przez cubo
- ✅ SEN2SR: 10m → 2.5m
- ✅ Real-ESRGAN: 2.5m → 1.25m
- ✅ Porównanie PRZED/PO w GUI

### v2 — planowane rozszerzenia
- [ ] Przetwarzanie wsadowe wielu kafelków (cały region)
- [ ] Eksport GeoTIFF z zachowanymi metadanymi georeferencji
- [ ] Indeks NDVI na wyjściu (do analizy upraw)
- [ ] Wybór daty sceny (aktualnie automatyczny)
- [ ] Fine-tuning SEN2SR na polskich danych rolniczych

---

## Kontekst dziedzinowy

Projekt dotyczy analizy upraw rolnych w Polsce.
Piksel 10×10m jest za gruby żeby rozróżnić małe pola.
Po pipeline'ie: ~1.25m/piksel pozwala widzieć rzędy upraw, granice pól, infrastrukturę.

Docelowe zastosowanie: monitoring stanu upraw, wykrywanie anomalii wegetacji.

---

*Ostatnia aktualizacja: v1.0*
