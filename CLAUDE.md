# CLAUDE.md — Instrukcja dla Claude Code
## Projekt: Sentinel-2 Super-Resolution Pipeline

---

## Czym jest ten projekt

Narzędzie do polepszania rozdzielczości zdjęć satelitarnych Sentinel-2 na potrzeby
**monitoringu upraw** (rozpoznawanie pól, granic działek, kondycji wegetacji / NDVI).

**Pipeline (baseline od Fazy 0 — SEN2SR-only):**
```
Copernicus (Sentinel-2 L2A)
    ↓  10 m/piksel  (oryginał)
SEN2SR  ×4  (neural network)
    ↓  2.5 m/piksel  (WYNIK baseline)

[krok 3 EDSR — OPCJONALNY, domyślnie WYŁĄCZONY (use_second_stage=False)]
```

> **Zmiana kierunku (patrz `WDROZENIE.md`):** celem nie jest sub-metrowy detal,
> tylko **wierne ~2–3 m + nienaruszone NDVI**. EDSR (dawniej Real-ESRGAN) został
> wyłączony — dorysowywał fałszywą teksturę i psuł wierność spektralną. Dalsze
> fazy: pomiar (opensr-test + zadanie docelowe) → MISR → fine-tuning pod PL.

**Interfejs:** GUI w Tkinter — przycisk "Pobierz i polepsz", porównanie PRZED/PO.

**Wyniki:** folder `output/` — pliki PNG z watermarkiem (oryginał + SEN2SR; EDSR tylko gdy włączony).

---

## Struktura plików w tym folderze

```
projekt/
├── CLAUDE.md           ← ten plik (instrukcja dla Ciebie)
├── WDROZENIE.md        ← plan rozwoju (fazy 0–4, bramy pomiaru)
├── pipeline.py         ← algorytm (pobieranie + SEN2SR; EDSR opcjonalny)
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
│   └── SEN2SRLite_RGBN/    ← wagi SEN2SRLite pobierane z HuggingFace
│                            (EDSR cachowany przez super-image tylko gdy włączony)
└── output/             ← (tworzone automatycznie)
    ├── 1_original_10m.png
    ├── 2_sen2sr_2.5m.png
    └── 3_superimage_1.25m.png   ← tylko gdy use_second_stage=True
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
| `run_sen2sr()` | Uruchamia SEN2SRLite RGBN×4 przez `mlstac` (rdzeń baseline) |
| `run_superimage()` | Krok 3 EDSR — **opcjonalny**, odpalany tylko gdy `use_second_stage=True` |
| `run_pipeline()` | Orchestracja + zapis PNG. Flaga `use_second_stage` (domyślnie `False`) |
| `tensor_to_rgb_uint8()` | Konwersja tensora Sentinel → RGB z percentile stretch |

`run_pipeline()` zwraca słownik: `original`, `sen2sr`, `final` (= `sen2sr` w baseline,
= `superimage` gdy EDSR włączony) oraz `elapsed_s`.

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

### Krok 3 (EDSR) — domyślnie wyłączony
Baseline to SEN2SR-only. EDSR (`super-image`) odpalisz tylko ustawiając
`use_second_stage=True` w `run_pipeline()`. Bez biblioteki `super-image`
robi fallback na Lanczos. Patrz `WDROZENIE.md` (Faza 0) — dlaczego wyłączony.

### Pierwsze uruchomienie trwa długo
Modele są pobierane raz:
- SEN2SRLite: ~50MB (HuggingFace)
- EDSR (tylko gdy włączony krok 3): ~40MB (HuggingFace)

---

## Cel projektu i kierunek rozwoju

**Plan rozwoju prowadzi `WDROZENIE.md`** — fazowy, z dwiema bramami pomiaru
(wierność spektralna/opensr-test + zadanie docelowe). Poniżej skrót.

### Stan (Faza 0 — ukończona)
- ✅ GUI z preselekcją miast polskich
- ✅ Automatyczne pobieranie Sentinel-2 przez cubo
- ✅ SEN2SR: 10m → 2.5m — **oficjalny baseline**
- ✅ EDSR wyłączony (był: Real-ESRGAN → EDSR; psuje NDVI)
- ✅ Porównanie PRZED/PO w GUI (crop 1:1)

### Następne fazy (wg `WDROZENIE.md`)
- [ ] **Faza 1 — Pomiar:** `opensr-test` (wierność + NDVI) + zadanie docelowe (delineacja pól)
- [ ] **Faza 2 — MISR:** Multi-Image SR ze stosu czasowego (dane już pobierane) + fenologia
- [ ] **Faza 3 — Fine-tuning PL:** dostrojenie SEN2SR na ortofoto GUGiK (25 cm), wymaga GPU
- [ ] Eksport GeoTIFF z georeferencją + indeks NDVI na wyjściu (priorytet rolniczy)
- [ ] Faza 4 (opc.): LDSR-S2 / Swin2-MOSE / fuzja z PlanetScope

---

## Kontekst dziedzinowy

Projekt dotyczy analizy upraw rolnych w Polsce.
Piksel 10×10m jest za gruby żeby rozróżnić małe pola.
Baseline SEN2SR daje wierne ~2.5m — wystarczające do granic pól i stref wegetacji.
Cel nie jest sub-metrowy: liczy się **wierna struktura ~2–3m + nienaruszone NDVI**,
nie „ładniejszy" obraz (patrz `WDROZENIE.md`).

Docelowe zastosowanie: monitoring stanu upraw, wykrywanie anomalii wegetacji.

---

*Ostatnia aktualizacja: Faza 0 (baseline SEN2SR-only) — plan w `WDROZENIE.md`*
