# Instrukcja wdrożenia — upgrade pipeline'u Sentinel-2 (monitoring upraw)

> Dokument roboczy, wersja 2 — przeorientowany pod realny cel: **rozpoznawanie pól, granic działek, struktury i kondycji upraw**. Nie celujemy w sub-metrowy detal (rower pod stodołą). Celujemy w wierną strukturę na skali kilku metrów + zachowaną informację spektralną (NDVI/NIR). Przechodzisz fazami od góry; każda ma: **co / dlaczego / jak / czego potrzeba / sukces**.

---

## Cel projektu — co właściwie optymalizujemy

To zmienia definicję „lepiej". Dla monitoringu upraw zysk to:
- **Ostrzejsze i pewniejsze granice pól** — szczególnie małych, wąskich, nieregularnych działek (klasyczny polski problem: przy 10 m piksel za gruby, żeby je rozróżnić — to wprost stoi w Waszym `CLAUDE.md`).
- **Zróżnicowanie wewnątrz pola** — strefy słabszej/lepszej wegetacji, wykrywanie anomalii.
- **Wierne NDVI / NIR** — kondycja upraw to przede wszystkim sygnał spektralny, nie sam kształt. SR, który ładnie wyostrza, ale rozjeżdża relację NIR↔Red, jest dla nas *bezużyteczny albo szkodliwy*, bo zafałszuje NDVI.
- **Użyteczny szereg czasowy** — fenologia (jak pole zmienia się w sezonie), bo i tak pobieramy wiele przelotów.

Czego **nie** potrzebujemy: rozróżniania pojedynczych rzędów upraw czy małych obiektów. To skala sub-metrowa, fizycznie poza zasięgiem Sentinela — i nieistotna dla zadania.

**Konsekwencja praktyczna:** miarą sukcesu nie jest „czy ładniej wygląda", tylko (1) czy NDVI pozostaje wierne i (2) czy na obrazie SR **lepiej delineują się pola / klasyfikują uprawy** niż na natywnych 10 m. To są twarde, mierzalne kryteria — patrz Faza 1.

---

## Punkt wyjścia — co już mamy

- **Pobieranie działa** (`cubo` → Sentinel-2 L2A, kanały B04/B03/B02/B08 = R/G/B/NIR). Drobiazg: `cubo` domyślnie ciągnie z **Planetary Computer**, nie z `dataspace.copernicus.eu` — to ten sam materiał (mirror), ale dobrze wiedzieć.
- **SEN2SR działa i daje realną poprawę** (10 m → 2.5 m). Zostaje jako fundament.
- **EDSR (krok 3) nic nie wnosi** — odpada (uzasadnienie niżej).

Dwa fakty, które kształtują plan:

1. **Stacking dwóch modeli SR to ślepa uliczka.** EDSR był trenowany na zwykłych zdjęciach, nie zna satelity ani polskich pól — w najlepszym razie wygładza, w gorszym dorysowuje fałsz i psuje gwarancję spektralną SEN2SR. Zamiast doklejać drugi model: albo **dokładamy realnej informacji** (więcej klatek / ostrzejszy sensor / dostrojenie do PL), albo **wymieniamy** SEN2SR na jeden lepszy model satelitarny.

2. **Dane do najlepszej metody już pobieracie.** `cubo.create()` zwraca wszystkie sceny z zakresu dat (wymiar `time`). Wasz kod bierze jedną (`da[sample_idx]`) i resztę wyrzuca. Multi-Image SR — metoda nr 1 do *wiernego* detalu i przy okazji do szeregu czasowego upraw — potrzebuje właśnie tego stosu.

---

## Realne oczekiwania — siatka vs. wierna rozdzielczość (przeczytaj zanim zaczniesz)

To rozróżnienie jest sednem całego projektu, więc na chłodno:

- **Rozmiar piksela na siatce** ≠ **wierna rozdzielczość**. Każdy obraz da się powiększyć do siatki 1 m jedną linijką kodu — ale powiększanie *nie tworzy nowej informacji* (jak rozciągnięcie rozmytego zdjęcia: pikseli więcej, detalu nie).
- **Te 2.5 m z SEN2SR to siatka 2.5 m**, w której zgrubne struktury (pole, droga, rzeka) są wierne — SEN2SR ma wbudowaną gwarancję, że po zmniejszeniu wynik wraca do oryginału 10 m. Ale ta gwarancja działa tylko na poziomie zgrubnym; *drobna tekstura* wewnątrz jest po części wnioskowana, nie zmierzona.
- Z **jednej** klatki 10 m SEN2SR wyciska już większość tego, co da się wiernie odzyskać. Optyka Sentinela fizycznie nie widzi ostro rzeczy mniejszych niż ~10–20 m, a brakujących wysokich częstotliwości z pojedynczego zdjęcia nie przeskoczysz.
- **Zejść poniżej 2.5 m wiernie da się** — ale tylko dokładając nowej informacji (MISR, ostrzejszy sensor). Realnie schodzi się w okolice **~1–3 m wiernego**, nie 1 m.
- **Wiernych 1×1 m z samego Sentinela-2 nie da się** (~100× więcej prawdziwej informacji niż jest w danych). Siatkę 1 m owszem, ale w dużej części zmyśloną — i to wyłapie test z Fazy 1.

**Dla naszego celu (uprawy) to dobra wiadomość:** nie potrzebujemy 1 m. Potrzebujemy wiernych ~2–3 m + dobrego NDVI — a to jest osiągalne i daje realny skok w rozróżnianiu małych pól i anomalii.

---

## Faza 0 — Sprzątanie (dziś, CPU, ~1 h)

### Co
Wywalić EDSR. „SEN2SR-only" staje się oficjalnym baseline'em.

### Dlaczego
Żeby nie mierzyć się z fałszywą teksturą i mieć czysty punkt odniesienia dla każdej kolejnej metody.

### Jak
Najmniej inwazyjnie — flaga wyłączająca krok 3 (domyślnie OFF), zamiast od razu kasować kod. W `run_pipeline()`:

```python
def run_pipeline(
    lat=DEFAULT_LAT, lon=DEFAULT_LON,
    start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
    edge_size=DEFAULT_EDGE_SIZE,
    esrgan_scale=2,
    use_second_stage=False,   # <-- EDSR domyślnie wyłączony
    progress_cb=None,
) -> dict:
    ...
    tensor_sr = run_sen2sr(raw_tensor, device, progress_cb=progress_cb)
    rgb_sen2sr = tensor_to_rgb_uint8(tensor_sr)
    path_sen2sr = OUTPUT_DIR / "2_sen2sr_2.5m.png"
    save_image(rgb_sen2sr, path_sen2sr, label="SEN2SR  2.5 m/px")

    results = {
        "original": str(path_before.resolve()),
        "sen2sr":   str(path_sen2sr.resolve()),
        "final":    str(path_sen2sr.resolve()),
    }

    if use_second_stage:   # zostaje na wszelki wypadek, ale domyślnie nieaktywne
        rgb_superimage = run_superimage(tensor_sr, device, scale=esrgan_scale, progress_cb=progress_cb)
        path_superimage = OUTPUT_DIR / f"3_superimage_{2.5/esrgan_scale:.2f}m.png"
        save_image(rgb_superimage, path_superimage, label=f"EDSR  {2.5/esrgan_scale:.2f} m/px")
        results["superimage"] = str(path_superimage.resolve())
        results["final"] = results["superimage"]
    ...
    return results
```

W `gui.py`: dla zakładki „superimage" użyj `paths.get("superimage")` i pokaż „krok wyłączony", gdy `None` — albo po prostu usuń tę zakładkę.

### Sukces
Pipeline puszcza dwa pliki (oryginał + SEN2SR), bez kroku 3, bez błędów.

---

## Faza 1 — Pomiar: dwie bramy (dni, CPU)

To jest serce projektu. Bez mierzenia każda metoda to zgadywanie „na oko". Mamy **dwie** bramy — i obie są ważne dla upraw.

### Brama A — wierność (opensr-test)
**Co:** framework od twórców SEN2SR (ESA OpenSR). Mierzy halucynacje, pominięcia, realną poprawę i — kluczowe dla nas — **zgodność spektralną** (czyli czy NDVI po SR zostaje wierne).

**Jak:**
```bash
pip install opensr-test
```
```python
import opensr_test, mlstac
model = mlstac.load("models/SEN2SRLite_RGBN").compiled_model(device="cpu")
dataset = opensr_test.load("spain_crops")   # hiszpańskie pola — najbliżej naszego case'u
lr, hr = dataset["L2A"], dataset["HRharm"]
m = opensr_test.Metrics()
m.compute(lr=lr, sr=model(lr), hr=hr)
m.summary()   # improvement ↑ , hallucination ↓ , consistency stabilne
```
(Dokładne API sprawdzaj w aktualnym README: https://github.com/ESAOpenSR/opensr-test.)

**Dodatkowo policz NDVI ręcznie** na wejściu i wyjściu i porównaj: NDVI = (B08 − B04) / (B08 + B04). Po SR średnie NDVI w obrębie pola nie powinno dryfować. To Twój własny, tani check spektralny.

### Brama B — zadanie docelowe (najuczciwsza miara dla upraw)
**Co:** sprawdzasz nie „ładność", tylko czy SR **realnie pomaga w zadaniu**: lepiej delineują się granice pól / lepiej klasyfikują uprawy na obrazie SR niż na natywnych 10 m?

**Jak:** weź prostą segmentację granic pól albo klasyfikację pokrycia terenu i puść ją na (1) 10 m, (2) SEN2SR 2.5 m. Porównaj dokładność. ESA OpenSR udostępnia do tego lekki toolkit walidacyjny (segmentacja na LR/SR/HR). Jeśli SR podnosi dokładność delineacji — to jest dowód wartości, którego żadna metryka pikselowa nie da.

### Sukces
Masz liczby dla gołego SEN2SR z obu bram. To **punkt kontrolny** — każda następna faza musi go pobić, inaczej jej nie wdrażamy.

---

## 📊 WYNIKI BASELINE (punkt kontrolny) — zmierzone

Skrypt: `measure.py` (Brama A + check NDVI). Dataset `spain_crops`, **n=28**, CPU, ~18 s.
Pasma `[B04,B03,B02,B08]` w L2A opensr-test = indeksy `[3,2,1,7]` (wyznaczone empirycznie, r≥0.93).
Pełne liczby w `output/faza1_metrics.json`.

**Brama A — wierność (opensr-test):**

| metryka | średnia | std | kierunek |
|---|---|---|---|
| reflectance | **0.0014** | 0.0005 | ↓ spójność LR (hard-constraint działa) |
| spectral | **0.3162** | 0.1139 | ↓ spójność spektralna |
| spatial | **0.0000** | 0.0000 | ↓ rejestracja (idealna na syntetyku) |
| improvement | **0.1206** | 0.0666 | ↑ realna poprawa |
| omission | **0.7945** | 0.1025 | ↓ pominięcia |
| hallucination | **0.0849** | 0.0414 | ↓ halucynacje (niskie — dobrze) |

**NDVI — zgodność spektralna (dodatek do Bramy A):**

| metryka | wartość | cel |
|---|---|---|
| bias (sr−lr) | **+0.0002** | ~0 → brak dryfu ✅ |
| MAE | **0.0025** | nisko ✅ |
| korelacja | **0.9987** | ~1 → NDVI zachowane ✅ |

**Brama B — delineacja granic pól (boundary F1 vs HR, label-free):**

| metryka | wartość | znaczenie |
|---|---|---|
| F1 natywne 10 m | 0.7063 | LR bilinear → 512 (baseline odniesienia) |
| F1 SEN2SR 2.5 m | **0.7690** | nasz pipeline |
| delta (SR−LR) | **+0.0626** | >0 → SR lepiej delineuje |
| SR wygrywa | **100%** | odsetek próbek (28/28) |

> Metoda: HR (512px) = referencja. Krawędzie z gradientu Sobela (stała gęstość
> przez kwantyl), zgodność liczona jako boundary F1 z tolerancją 2 px (scipy EDT).
> Brak gotowego toolkitu w opensr-test → własny (skimage + scipy).

**Wniosek:** SEN2SR to wiarygodny baseline — realna poprawa przy niskich halucynacjach,
**zerowy dryf NDVI** (hard-constraint) i **mierzalnie lepsza delineacja pól niż natywne
10 m** (F1 +0.063, wygrywa w 100% próbek). Obie bramy zaliczone. To są liczby do pobicia
przez MISR / fine-tuning.

---

## Faza 2 — MISR: realny detal + szereg czasowy (tygodnie, CPU → opc. GPU)

> **📌 POSTĘP (w toku):**
> - ✅ **2a** — `download_sentinel2_stack()` w `pipeline.py`: zwraca cały stos `[T,4,H,W]`
>   z filtrem chmur (`eo:cloud_cover < max_cloud`). Przestaliśmy wyrzucać przeloty.
> - ✅ **2b** — `misr.py`: filtr jakości klatek (odsiew chmur/cienia bez SCL),
>   koregistracja subpikselowa (`phase_cross_correlation`, dokładność 1/20 px),
>   wybór referencji (najostrzejsza klatka).
> - ✅ **Fuzja klasyczna** (shift-and-fuse, median robust) + samotest syntetyczny:
>   fuzja bije pojedynczą klatkę o **+1.4 dB (8 kl.)**, **+1.9 dB (16 kl.)**,
>   **+3.1 dB (16 kl., szum 0.10)** — zysk rośnie z liczbą klatek i szumem (jak teoria).
> - ⏳ **Do zrobienia:** (1) ewaluacja w bramach z Fazy 1 wymaga benchmarku
>   **wieloczasowego** (PROBA-V / WorldStrat) — spain_crops jest jednoklatkowy;
>   (2) model głęboki MISR (HighRes-net / RAMS) — krok 2c; (3) wpięcie fuzji
>   jako pre-etapu przed SEN2SR w `run_pipeline()`.

### Co
Multi-Image Super-Resolution — łączenie wielu przelotów tego samego pola w jeden ostrzejszy obraz.

### Dlaczego to dla nas metoda nr 1
- To **jedyna** metoda odzyskująca *fizycznie prawdziwy* detal (nie zgadywanie): każdy przelot jest minimalnie przesunięty subpikselowo i z tych przesunięć da się zrekonstruować realną informację.
- **Sceny już pobieracie** — trzeba tylko przestać je wyrzucać.
- Bonus rolniczy: ten sam stos czasowy daje **fenologię upraw** (jak pole zmienia się w sezonie) — wartość sama w sobie dla monitoringu.
- MISR z natury **zachowuje informację spektralną** lepiej niż dorysowujący model generatywny — czyli dobre dla NDVI.

### Jak — krok po kroku

**2a. Przestań wyrzucać stos czasowy.** Funkcja obok obecnej `download_sentinel2()`:
```python
def download_sentinel2_stack(lat, lon, start_date, end_date, edge_size, max_cloud=20):
    """Zwraca CAŁY stos czasowy: tensor [T, 4, H, W] zamiast jednej sceny."""
    da = cubo.create(
        lat=lat, lon=lon,
        collection="sentinel-2-l2a",
        bands=["B04", "B03", "B02", "B08"],
        start_date=start_date, end_date=end_date,
        edge_size=edge_size, resolution=10,
        query={"eo:cloud_cover": {"lt": max_cloud}},   # odsiej mocno zachmurzone
    )
    arr = (da.compute().to_numpy() / 10_000).astype("float32")  # (T, 4, H, W)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(arr)
```
To samo źródło, ten sam koszt — tylko zachowujesz wszystkie klatki.

**2b. Filtrowanie + koregistracja (krok, którego nie wolno olać):**
- odrzuć sceny z chmurami/cieniem (maska chmur Sentinela albo prosty próg jasności),
- **subpikselowo dopasuj** klatki względem siebie — bez tego MISR rozmazuje zamiast wyostrzać. Na start w Pythonie: `scikit-image` → `phase_cross_correlation`.
- Bo to uprawy: trzymaj **krótkie okno** (2–4 tyg.) i maskuj zmiany — pole zaorane między przelotami popsuje fuzję.

**2c. Model MISR (zacznij od gotowego, wytrenowanego):**
- **HighRes-net** — łączy dowolną liczbę klatek, lekki, inferencja realna na CPU dla małych kafelków.
- **RAMS** — mocne wyniki na PROBA-V.
- Wagi z konkursu PROBA-V transferują się na Sentinela (potwierdzone w literaturze) — nie musisz trenować od zera, żeby zacząć.
- Repo do oparcia się o europejskie dane: **WorldStrat** (github.com/worldstrat/worldstrat) — pary Sentinel-2 (16 przelotów) + HR 1.5 m.

**2d. Porównaj w obu bramach z Fazy 1** (wierność + zadanie docelowe).

### Czego potrzeba
CPU wystarcza do prototypu i inferencji na małych kafelkach. GPU tylko jeśli zechcesz dotrenować sieć na własnych danych (opcjonalne). Internet — więcej transferu (pobierasz wiele scen).

### Sukces
MISR na stosie kilkunastu przelotów podnosi `improvement` i/lub dokładność delineacji pól, bez wzrostu halucynacji i bez dryfu NDVI. Wtedy MISR wchodzi jako nowy etap przed/zamiast pojedynczej sceny na wejściu SEN2SR.

---

## Faza 3 — Polskie dane HR + fine-tuning SEN2SR pod rolnictwo (tygodnie, GPU)

### Co
Dostrojenie istniejącego SEN2SR do polskich pól (nie trening od zera) + rozwiązanie problemu braku danych HR.

### Dlaczego
SEN2SR uczony był na amerykańskich zdjęciach lotniczych. Polskie pola (małe, rozdrobnione), kalendarz upraw i oświetlenie są inne. Dostrojenie na polskich parach to najczystszy sposób, żeby model był wiarygodny *na naszym terenie i naszym zadaniu*.

### Jak — krok po kroku

**3a. Darmowe polskie HR (GUGiK/Geoportal).** Ortofotomapy, w większości **25 cm/px** w całym kraju — setki razy ostrzejsze niż Sentinel, idealny „nauczyciel".
Kanały: SEN2SR potrzebuje R/G/B/NIR. GUGiK daje osobno arkusze **RGB** i arkusze **CIR** (NIR/R/G). Żeby złożyć pełne RGBN — **łączysz RGB z CIR** (niebieski tylko z RGB, NIR tylko z CIR). Dla rolnictwa NIR jest krytyczny (NDVI) — nie pomijaj CIR.
Pobieranie:
- **`rgugik`** — pakiet **R** (nie Python!), ale do jednorazowego ściągnięcia świetny: `ortho_request()` + `tile_download()`.
- **Czysto w Pythonie:** pliki leżą jako publiczne GeoTIFF-y pod `opendata.geoportal.gov.pl/ortofotomapa/...` — pobierzesz je `requests` + `rasterio`, albo przez WMS/WMTS (`owslib` + `rasterio`).
- **Gotowiec na start:** dataset **LandCover.ai** (polskie ortofoto 25/50 cm) — tylko RGB, ale dobry do pierwszych prób.

**3b. Buduj pary LR–HR (przepis SEN2NAIP):** z HR (ortofoto 25 cm) generujesz sztuczny „Sentinel":
1. rozmycie Gaussa + zmniejszenie do 10 m,
2. **harmonizacja odbicia** do skali Sentinela — **najważniejszy** krok (ortofoto nie jest kalibrowane radiometrycznie; dla rolnictwa szczególnie pilnuj kanału NIR),
3. dodanie szumu.
Kod do skopiowania: dataset **SEN2NAIPv2** + loader **`tacoreader`** (HuggingFace `tacofoundation/SEN2NAIPv2`, CC0). Wariant lepszy (trudniejszy): prawdziwe pary Sentinel↔ortofoto przez koregistrację.

**3c. Fine-tuning:** start **od wag SEN2SRLite**, niskim LR, na polskich parach rolniczych, z zachowaną warstwą hard-constraint. Kod modelu masz w `SEN2SR-main/sen2sr/` (architektura `CNNSR`). Dobierz dane tak, żeby przeważały **pola** (cel projektu), z domieszką miasta/rzeki dla różnorodności.

**3d. Walidacja:** obie bramy z Fazy 1 **przed i po** dostrojeniu, na **polskim** zestawie testowym z ortofoto (kilka kafelków odłożonych tylko do testu). Szczególnie pilnuj zgodności NDVI.

### Czego potrzeba
- **GPU** — tu zamawiacie serwer. Fine-tuning na CPU jest niepraktyczny.
- Inferencja po dostrojeniu zostaje **CPU-friendly** (`opensr-utils` robi kafelkowanie dużych scen).
- Dane: do fine-tuningu wystarczy kilka tysięcy dobrze dobranych polskich kafelków. Zacznij mało, zmierz, dokładaj.

### Sukces
Dostrojony model bije gołe SEN2SR na **polskim** zestawie: wyższy improvement, niższa halucynacja, stabilne NDVI, lepsza delineacja pól. To twardy dowód, że GPU się zwróciło.

---

## Faza 4 — Opcjonalne, gdy darmowe metody się wyczerpią

**4a. Wymiana SEN2SR na jeden lepszy model satelitarny (zamiast stackowania):**
- **LDSR-S2 / `opensr-model`** — latent diffusion ESA OpenSR dla naszych kanałów RGBN, z **mapą niepewności**. Ostrzejszy, ale jako generatywny **wymaga** kontroli halucynacji (dla NDVI ostrożnie). Są notebooki Colab.
- **Swin2-MOSE** — transformer SR pod remote sensing (github.com/IMPLabUniPr/swin2-mose).

**4b. Fuzja z PlanetScope — największy *realny* skok, ale płatny:**
- PlanetScope (~3 m, niemal codziennie) + Sentinel-2, fuzja spatiotemporalna (STARFM/ESTARFM lub uczona). To **klasyka monitoringu upraw** — daje codzienne NDVI w ~3 m. Realny detal z realnego ostrzejszego sensora. Komercyjny, ale ma programy badawcze/edukacyjne i unijne. Ścieżka, jeśli potrzebujecie ~3 m wiernego na dużą skalę z dużą częstotliwością.

**4c. Sentinel-1 (radar):** dodatkowa, fizycznie prawdziwa informacja strukturalna — najlepsza do zadań pochodnych (granice pól, klasyfikacja w pochmurne dni), nie do „ładniejszego" obrazu optycznego.

---

## Co odpada i dlaczego

- **EDSR / `super-image`** — inna domena, dorysowuje fałsz, psuje NDVI. Usunięte w Fazie 0.
- **Stackowanie dwóch modeli SR** — mnoży halucynacje i psuje gwarancję spektralną. Dokładamy informacji albo wymieniamy na jeden lepszy model.
- **Generyczne upscalery z internetu** (Real-ESRGAN itp.) — ta sama wada.
- **Ocena na samym PSNR/SSIM** — potrafi wskazać zmyślający model jako lepszy. Używamy opensr-test + zadania docelowego.

---

## Mapa decyzji — kolejność i progi

| Faza | Wysiłek | Sprzęt | Dane | Zysk dla upraw | Wierny? |
|---|---|---|---|---|---|
| 0 — Usuń EDSR | godziny | CPU | — | czysty baseline | n/d |
| 1 — Pomiar (2 bramy) | dni | CPU | wbudowane | brama jakości | n/d |
| 2 — MISR | tygodnie | CPU (+GPU opc.) | już pobierane | **wierny detal + fenologia** | ✅ |
| 3 — fine-tune PL | tygodnie | **GPU** | GUGiK ortofoto | **dostrojony do polskich pól** | ✅ |
| 4a — LDSR/Swin2 | dni–tyg. | GPU | wbudowane | ostrzej, ostrożnie z NDVI | ⚠️ |
| 4b — PlanetScope | tygodnie | GPU | płatne | **codzienne NDVI ~3 m** | ✅ |

**Ścieżka rekomendowana:** 0 → 1 → 2 (MISR, bo dane już są) → jeśli MISR przejdzie obie bramy, równolegle 3 (fine-tune, gdy będzie GPU). Faza 4 tylko gdy 1–3 nie wystarczą dla zadania.

**Progi „dalej / zawracamy":**
- MISR nie podnosi improvement ani delineacji → problem z koregistracją; oprzyj się na Fazie 3.
- Fine-tune nie bije baseline'u → domain gap degradacji za duży; przejdź na prawdziwe pary Sentinel↔ortofoto.
- NDVI dryfuje po SR → odrzuć tę metodę niezależnie od „ładności" — dla upraw to dyskwalifikacja.
- Nawet ~2–3 m za mało / potrzebna codzienna częstotliwość → eskalacja do PlanetScope (4b, płatne).

**Zasada przewodnia:** preferuj metody dokładające *informacji* (więcej klatek, ostrzejszy sensor, dostrojenie do PL) nad dokładającymi tylko *tekstury*. I trzymaj się gwarancji SEN2SR: wynik SR po zmniejszeniu musi wracać do oryginalnego odbicia Sentinela (to chroni NDVI).

---

## Porządki w projekcie przy okazji

- **Instalacja:** dodaj `opensr-test`; usuń `super-image` (+ ewentualnie `basicsr`/`realesrgan`). Później dojdą `scikit-image`, `rasterio`, `tacoreader`, `opensr-utils`, `opensr-model`.
- **`CLAUDE.md` jest nieaktualny** — opisuje krok 3 jako Real-ESRGAN, a kod używał EDSR. Po Fazie 0 zaktualizuj do „SEN2SR-only (baseline) + plan MISR/fine-tune". Przy okazji punkt „NDVI na wyjściu" i „eksport GeoTIFF z georeferencją" z Waszej listy v2 wpisuje się idealnie w cel rolniczy — NDVI to wręcz priorytet, nie dodatek.
- **Literówka w GUI:** przycisk raz „POLEPSz", raz „POLEPSZY" — ujednolić.
- **Opcjonalnie:** jeśli chcecie pobierać prosto z Copernicus Data Space zamiast Planetary Computer — `cubo.create()` przyjmuje `stac=` z innym endpointem (dane te same, niewymagane).

---

*Kolejność czytania kodu przy modyfikacjach:* `pipeline.py` (`download_sentinel2`, `run_sen2sr`, `run_pipeline`) → `SEN2SR-main/sen2sr/utils.py` (`predict_large` — kafelkowanie) → `SEN2SR-main/sen2sr/models/opensr_baseline/cnn.py` (architektura `CNNSR`, gdyby fine-tune wymagał zmian w modelu).
