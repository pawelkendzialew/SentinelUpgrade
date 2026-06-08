# CLAUDE.md — Instrukcja dla Claude Code
## Projekt: Sentinel-2 Super-Resolution Pipeline

---

## Czym jest ten projekt

Narzędzie do polepszania rozdzielczości zdjęć satelitarnych Sentinel-2 na potrzeby
**monitoringu upraw** (rozpoznawanie pól, granic działek, kondycji wegetacji / NDVI).

**Pipeline (czysty — SEN2SR + produkt geo):**
```
Copernicus (Sentinel-2 L2A)
    ↓  10 m/piksel  (oryginał)
[opc. MISR: fuzja stosu czasowego → czystsza klatka 10 m (use_misr)]
SEN2SR  ×4  (neural network; opc. wagi dostrojone do PL — use_finetuned)
    ↓  2.5 m/piksel  (WYNIK)
Eksport: GeoTIFF (RGBN) + NDVI (GeoTIFF + PNG) → QGIS
```

> **Granica wierności = 2.5 m.** Zejście niżej (1.25 m) z samego Sentinela
> przetestowane dwiema metodami (fuzja MISR ×2 + uczony model ×2) — **żadna
> nie pobiła zwykłego powiększenia względem prawdy GUGiK**. To fizyczny sufit:
> w danych 10 m nie ma realnego detalu poniżej 2.5 m. Eksperymenty usunięte.

> **Zmiana kierunku (patrz `WDROZENIE.md`):** celem nie jest sub-metrowy detal,
> tylko **wierne ~2–3 m + nienaruszone NDVI**. EDSR (dawniej Real-ESRGAN) został
> wyłączony — dorysowywał fałszywą teksturę i psuł wierność spektralną. Dalsze
> fazy: pomiar (opensr-test + zadanie docelowe) → MISR → fine-tuning pod PL.

**Interfejs:** GUI w Tkinter — przycisk "Pobierz i polepsz", porównanie PRZED/PO.

**Wyniki:** folder `output/` — PNG (oryginał + SEN2SR + NDVI) oraz **GeoTIFF RGBN + NDVI** do QGIS.

---

## Struktura plików w tym folderze

```
projekt/
├── CLAUDE.md           ← ten plik (instrukcja dla Ciebie)
├── WDROZENIE.md        ← plan rozwoju + wyniki + wnioski (sufit 2.5 m)
├── pipeline.py         ← rdzeń: pobieranie + (opc. MISR) + SEN2SR + eksport GeoTIFF/NDVI
├── gui.py              ← interfejs graficzny Tkinter
├── misr.py             ← MISR: koregistracja subpikselowa + fuzja median (lepsza jakość)
├── geoexport.py        ← eksport GeoTIFF z georeferencją + mapa NDVI (produkt QGIS)
├── measure.py          ← pomiar jakości (opensr-test + NDVI + delineacja pól)
├── finetune.py         ← fine-tuning SEN2SR (pętla CPU, spain_crops)
├── gugik.py            ← pobieranie ortofoto GUGiK RGB + degradacja SEN2NAIP
├── finetune_gugik.py   ← fine-tuning na realnych polskich polach (wagi PL)
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
├── models/             ← SEN2SRLite pobierane z HuggingFace (gitignore)
│   ├── SEN2SRLite_RGBN/             ← wagi bazowe (auto-pobranie)
│   └── sen2sr_finetuned_pl_gugik.pt ← wagi dostrojone do PL (w repo, ~2.3 MB)
└── output/             ← (tworzone automatycznie)
    ├── 1_original_10m.png
    ├── 2_sen2sr_2.5m.png
    ├── sen2sr_2.5m.tif          ← GeoTIFF RGBN z georeferencją (PRODUKT → QGIS)
    ├── ndvi_2.5m.tif            ← NDVI GeoTIFF (analiza upraw)
    └── ndvi_2.5m.png            ← NDVI kolormapa (podgląd)
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
#    (mlstac wymaga matplotlib!)
pip install sen2sr mlstac matplotlib git+https://github.com/ESDS-Leipzig/cubo.git

# 4. Pomiar jakości (Faza 1) + eksport geo + dane GUGiK (Faza 3/B)
#    rasterio/pyproj zwykle przychodzą z cubo; requests do WMS GUGiK
pip install opensr-test rasterio pyproj requests

# 5. Pozostałe
pip install Pillow numpy
```

> **Uwaga:** repo stawiane na czystym pip (Python 3.14, bez conda). Drugi etap SR
> poniżej 2.5 m (Real-ESRGAN/EDSR/MISR ×2/uczony ×2) był testowany i **usunięty** —
> żaden nie dawał wiernego detalu poniżej 2.5 m (patrz `WDROZENIE.md`).

### Weryfikacja instalacji
```bash
python -c "import sen2sr, cubo, mlstac, opensr_test; print('OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Uruchomienie pomiaru (Faza 1)
```bash
python measure.py            # baseline SEN2SR, 8 próbek
python measure.py --n 28     # pełny dataset spain_crops
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
| `download_sentinel2()` | Pobiera jedną scenę przez `cubo.create()` z Copernicus |
| `download_sentinel2_stack()` | Pobiera cały stos czasowy (dla MISR) |
| `run_sen2sr()` | SEN2SRLite RGBN×4 przez `mlstac`; `use_finetuned` → wagi PL |
| `run_pipeline()` | Orchestracja: (opc. MISR) → SEN2SR → zapis PNG + GeoTIFF + NDVI |
| `tensor_to_rgb_uint8()` | Konwersja tensora Sentinel → RGB z percentile stretch |

Flagi `run_pipeline()`: `use_misr` (fuzja stosu), `use_finetuned` (wagi PL).
Zwraca słownik: `original`, `sen2sr`, `final`, `geotiff`, `ndvi_tif`, `ndvi_png`, `elapsed_s`.

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

### Dlaczego wynik to 2.5 m, nie mniej
Zejście poniżej 2.5 m z samego Sentinela testowano i odrzucono (brak realnego
detalu w danych 10 m). Wierny produkt = 2.5 m. Patrz `WDROZENIE.md`.

### Pierwsze uruchomienie trwa długo
Model SEN2SRLite (~50 MB) pobierany raz z HuggingFace.
Wagi dostrojone do PL (`sen2sr_finetuned_pl_gugik.pt`) są już w repo.

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
- ✅ **Faza 1 — Pomiar:** `measure.py` — Brama A (opensr-test + NDVI) + Brama B (delineacja pól).
      Baseline zmierzony: improvement 0.121, halucynacje 0.085, NDVI corr 0.999, F1 +0.063 vs 10m.
- ✅ **Faza 2 — MISR:** `misr.py` + `eval_misr.py` + `highresnet.py` + `download_sentinel2_stack()`.
      Fuzja MISR (median) bije pojedynczą klatkę w bramach: **+6.5 dB PSNR, +0.09 F1, NDVI corr 0.44→0.76**.
      HighRes-net (od zera, CPU) NIE bije median (−0.47 dB) → wdrażamy median. Wpięte: `use_misr=True`
      w `run_pipeline()` + przełącznik w GUI. Dalej (przyszłość): wagi PROBA-V dla sieci, realny stos z cubo.
- ✅ **Faza 3 — Fine-tuning (CPU, polskie dane):** `finetune.py` (pętla) + `gugik.py` + `finetune_gugik.py`.
      GPU **NIE konieczne** (572k param., 0.23 s/krok). Hard-constraint zamrożony. Na realnych polskich polach
      GUGiK: **+0.93 dB PSNR** na niewidzianych lokalizacjach (vs +0.11 na hiszpańskich — gap zamknięty).
      **NIR:** GUGiK daje tylko RGB → NIR wykluczony ze straty (`NIR_TODO`: CIR archiwalny w przyszłości).
- ✅ **Eksport GeoTIFF z georeferencją + NDVI** (`geoexport.py`) — produkt rolniczy. Zweryfikowane na realnym
      polskim polu (EPSG:32633, piksel 2.5 m). **MISR na realnym stosie (15/21 klatek): NDVI 0.38 (zdrowa
      wegetacja); pojedyncza scena trafiła w chmurę: NDVI −0.02** — dowód wartości MISR na realnych danych.
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
