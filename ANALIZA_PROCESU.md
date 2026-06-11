# Analiza procesu — Sentinel-2 Super-Resolution

> Dokument techniczny: źródło danych, formaty, architektura kafelkowania, zmierzone
> czasy i skalowanie. Pomiary z `benchmark.py` (CPU, `output/benchmark_data.json`).
> Lokalizacja testowa: wschodnio-centralna PL (51.40 N, 22.00 E), lato 2023.

---

## 1. Skąd pobieramy zdjęcia satelitarne

| | |
|---|---|
| **Satelita** | Sentinel-2 (ESA / Copernicus), produkt **L2A** (odbicie powierzchni, skorygowane atmosferycznie) |
| **Pośrednik** | biblioteka `cubo` → **Microsoft Planetary Computer** (publiczny mirror STAC/COG Sentinela) |
| **Format na serwerze** | Cloud-Optimized GeoTIFF (COG) — czytamy fragment bez ściągania całej sceny |
| **Rozdzielczość źródła** | **10 m/piksel** (natywna dla pasm widzialnych + NIR) |
| **Koszt** | darmowe |

### Jakie pasma dostajemy (w tym podczerwień)

Pobieramy **4 pasma 10 m**:

| pasmo | nazwa | długość fali | rola |
|---|---|---|---|
| **B04** | Red | 665 nm | R |
| **B03** | Green | 560 nm | G |
| **B02** | Blue | 490 nm | B |
| **B08** | **NIR (bliska podczerwień)** | 842 nm | NIR — **tak, dostajemy podczerwień** |

NIR (B08) jest kluczowy: napędza **NDVI** = (NIR − Red) / (NIR + Red) — wskaźnik kondycji
wegetacji do monitoringu upraw.

### W jakiej skali są wartości
- Surowe wartości to **reflektancja powierzchni × 10000** (liczby całkowite, typ uint16,
  zwykle ~0–4000 dla lądu, ~6000 dla NIR roślinności).
- W kodzie dzielimy przez 10000 → reflektancja w zakresie **[0, 1]**.

---

## 2. Co produkujemy na wyjściu

```
Sentinel-2 10 m (4 pasma RGBN)  →  SEN2SR ×4  →  2.5 m
```

Pliki (folder `output/` lub wskazany):
- **`*.tif`** — GeoTIFF 4-pasmowy RGBN, **piksel 2.5 m**, georeferencja UTM (np. EPSG:32634)
- **`ndvi_*.tif`** — NDVI jako GeoTIFF (1-pasmowy)
- **`ndvi_*.png`** — NDVI jako kolormapa (podgląd)

Każdy plik niesie **CRS + transform** (układ współrzędnych i położenie), więc w QGIS
nakłada się dokładnie na mapę.

---

## 3. Czy cała Polska to JEDEN plik, czy tysiące kafelków?

**To ZAWSZE tysiące osobnych kafelków (TIFF-ów), nigdy jeden plik.** Zaznaczenie
większego obszaru = więcej kafelków, nie większy pojedynczy obraz.

### Dlaczego MUSI być kafelkowane (3 twarde powody)

1. **Model SEN2SR przetwarza fragmenty 128×128 px** (ma stałą maskę częstotliwościową).
   Większe obrazy tniemy wewnętrznie na kafelki 128 px i sklejamy.
2. **Planetary Computer ma limity wielkości zapytania** — przy większych kafelkach
   (256–512 px) pojawiają się **timeouty** („request exceeded maximum time"). Nie da się
   jednym zapytaniem ściągnąć dużego obszaru.
3. **Pamięć RAM** — cała Polska przy 2.5 m to **~50 miliardów pikseli × 4 pasma**.
   Jednego takiego obrazu nie da się zmieścić w pamięci ani w jednym pliku praktycznie.

### Od czego zależy liczba kafelków

```
liczba kafelków  =  pole obszaru  /  pole jednego kafelka
```

Rozmiar kafelka jest **konfigurowalny** (pole „Rozmiar kafelka" w GUI: 128 / 256 / 512 px):

| kafelek | bok terenu | pole jednego kafelka |
|---|---|---|
| 128 px | 1.28 km | 1.64 km² |
| 256 px | 2.56 km | 6.55 km² |
| 512 px | 5.12 km | 26.2 km² |

Siatka jest układana na **UTM** tak, że kafelki **przylegają bez szczelin** — w QGIS
wczytujesz wszystkie naraz i składają się w jeden ciągły obraz (lub Raster → Merge).

---

## 4. Zmierzone czasy (CPU, benchmark)

Pomiar: SEN2SR na syntetycznym wejściu (czysty compute, 3 powtórzenia),
pobieranie realnych scen (2 powtórzenia), rozmiar GeoTIFF na dysku.

| kafelek | pole | **pobieranie** (s) | **SEN2SR** (s) | **dysk** (RGBN) |
|---|---|---|---|---|
| 128 px (1.28 km) | 1.64 km² | 1.8 (0.9–2.7) | 0.38 (0.15–0.82) | 3.6 MB |
| 256 px (2.56 km) | 6.55 km² | 0.9 (0.86–0.90) | 0.45 (0.43–0.46) | 14.3 MB |
| 512 px (5.12 km) | 26.2 km² | 1.4 (1.3–1.6) | 1.72 (1.6–1.9) | 57.3 MB |

> **SEN2SR (CPU)** skaluje się z liczbą pikseli: 128→256 (×4 piksele) ≈ stały (overhead),
> 256→512 (×4 piksele) ≈ ×3.8 czasu. Przewidywalne.
>
> **Pobieranie** jest **zmienne i nieprzewidywalne** (sieć): w benchmarku 0.9–2.7 s, ale
> w realnym przebiegu regionu **trafialiśmy na timeouty** wymagające ponawiania (kilkadziesiąt
> sekund na kafelek). Do planowania dużej skali zakładaj **realnie ~5–10 s/kafelek** średnio.

### Czas całkowity na kafelek (do planowania)

| kafelek | pole | total (idealne) | total (realne, z siecią) |
|---|---|---|---|
| 128 px | 1.64 km² | ~2.3 s | ~5–8 s |
| 256 px | 6.55 km² | ~1.4 s | ~5–10 s |
| 512 px | 26.2 km² | ~3.2 s | ~8–15 s |

**Im większy kafelek, tym wydajniej na km²** (mniej osobnych zapytań sieciowych).

---

## 5. Skalowanie — ile zajmie dany obszar

Założenia: kafelek **256 px** (6.55 km²), CPU, realny czas ~6 s/kafelek, dysk 14.3 MB/kafelek.

| obszar | przykład | kafelków | czas (CPU, realnie) | dysk |
|---|---|---|---|---|
| **1 km²** | 1×1 km | 1 (min. 128 px) | ~5–8 s | ~4 MB |
| **100 km²** | 10×10 km | ~16 | ~2 min | ~230 MB |
| **2 500 km²** | 50×50 km | ~382 | ~40 min | ~5.5 GB |
| **10 000 km²** | 100×100 km | ~1 530 | ~2.5 h | ~22 GB |
| **~25 000 km²** | województwo | ~3 800 | ~6 h | ~55 GB |
| **312 696 km²** | **CAŁA POLSKA** | **~47 700** | **~2–3 doby** | **~680 GB** |

> Uwaga: dysk jest ~stały niezależnie od rozmiaru kafelka (ten sam obszar = ta sama liczba
> pikseli wyjściowych). Cała Polska przy 2.5 m to **~680 GB** danych RGBN (+ NDVI).

### GPU vs CPU
- **SEN2SR na GPU**: ~20–50× szybciej niż CPU.
- **ALE pobieranie jest takie samo** (sieć, nie compute). Przy dużej skali to **pobieranie
  staje się wąskim gardłem**, nie model. GPU skraca część SEN2SR, ale całość i tak limituje
  sieć (~5–10 s/kafelek). Cała Polska na GPU: rząd **~12–24 h** (dominuje pobieranie).

---

## 6. Wąskie gardła i zalecenia

| element | status | uwaga |
|---|---|---|
| **Pobieranie (sieć)** | 🔴 główne ryzyko | timeouty Planetary Computer, zmienne; mamy ponawianie + próby kolejnych scen |
| **SEN2SR (compute)** | 🟢 przewidywalny | CPU OK do średnich obszarów; GPU dla dużych |
| **Dysk** | 🟡 rośnie szybko | cała Polska ~680 GB; planuj miejsce |
| **Wznawialność** | 🟢 jest | batch pomija gotowe kafelki — można przerwać i wrócić |

**Rekomendacje do dużej skali:**
1. Większy kafelek (512 px) — mniej zapytań sieciowych, wydajniej na km².
2. Lecieć w tle / na noc; dzięki wznawialności rozłożyć na kilka sesji.
3. Dla całych województw/Polski — realnie potrzebny **GPU + stabilne łącze + dużo dysku**.
4. Wybierać **lato** (mniej chmur) i wąski zakres dat (mniej scen do przeszukania).

---

## 7. Podsumowanie kluczowych faktów

- **Źródło:** Sentinel-2 L2A (darmowe) przez Planetary Computer, **10 m**, 4 pasma **RGB + NIR**.
- **Podczerwień:** tak, pasmo B08 (842 nm) — do NDVI.
- **Wynik:** GeoTIFF RGBN **2.5 m** + NDVI, georeferencja UTM, gotowe do QGIS.
- **Kafelkowanie:** zawsze wiele kafelków; cała Polska = **~12 000–48 000 osobnych TIFF-ów**
  (zależnie od rozmiaru kafelka), **nie jeden plik**. Wymuszone przez model, limity serwera i RAM.
- **Czas:** ~kilka s/kafelek (CPU), ale **pobieranie zmienne** — przy tysiącach kafelków
  liczone w godzinach/dobach; GPU przyspiesza compute, sieć pozostaje wąskim gardłem.
- **Dysk:** rośnie liniowo z obszarem; cała Polska ~**680 GB**.

*Dane pomiarowe: `output/benchmark_data.json` (regeneracja: `python benchmark.py`).*
