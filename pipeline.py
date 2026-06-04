"""
Sentinel-2 Super-Resolution Pipeline
=====================================
10m/px  →  SEN2SR x4  →  2.5m/px  →  Real-ESRGAN x2  →  ~1.25m/px

Obszar testowy: Kraków i okolice (pola, urban, rzeka)
Lat: 50.0647, Lon: 19.9450
"""

import os
import sys
import time
import warnings
import logging
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import mlstac
import cubo
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────

OUTPUT_DIR = Path("output")
MODEL_DIR  = Path("models")

# Obszar testowy – Kraków, mieszane uprawy + miasto + Wisła
DEFAULT_LAT        = 50.0647
DEFAULT_LON        = 19.9450
DEFAULT_START_DATE = "2023-06-01"
DEFAULT_END_DATE   = "2023-09-30"
DEFAULT_EDGE_SIZE  = 256   # 256x256 px @ 10m = ~2.5km x 2.5km kafelek

SEN2SR_MODEL_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
    "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
)
SEN2SR_MODEL_DIR = MODEL_DIR / "SEN2SRLite_RGBN"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    """
    Konwertuje tensor Sentinel-2 (B, H, W) do obrazu RGB uint8.
    Używa kanałów [B04-Red, B03-Green, B02-Blue] (indeksy 0,1,2 dla RGBN).
    """
    arr = tensor.detach().cpu().numpy()  # (4, H, W)
    rgb = arr[[0, 1, 2], :, :]          # Red, Green, Blue
    # Percentile stretch dla lepszej wizualizacji
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    return np.transpose(rgb, (1, 2, 0))  # (H, W, 3)


def save_image(arr: np.ndarray, path: Path, label: str = "") -> None:
    img = Image.fromarray(arr)
    if label:
        # Małe watermark z rozdzielczością
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        margin = 8
        draw.rectangle([margin-2, margin-2, margin+len(label)*11+2, margin+22], fill=(0,0,0,160))
        draw.text((margin, margin), label, fill=(255, 255, 255), font=font)
    img.save(path)
    log.info(f"  ✓ Zapisano: {path} ({img.size[0]}x{img.size[1]} px)")


# ─────────────────────────────────────────────
# KROK 1 – Pobieranie danych Sentinel-2
# ─────────────────────────────────────────────

def download_sentinel2(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    edge_size: int = DEFAULT_EDGE_SIZE,
    progress_cb: Optional[Callable] = None,
) -> tuple[torch.Tensor, int, Optional[dict]]:
    """
    Pobiera dane Sentinel-2 L2A przez cubo.
    Zwraca (tensor [4, H, W], indeks_próbki, geo) gdzie geo = dict georeferencji
    (crs, west, north, res) lub None.
    """
    log.info(f"\n[1/3] Pobieranie danych Sentinel-2...")
    log.info(f"      Lokalizacja: lat={lat}, lon={lon}")
    log.info(f"      Okres: {start_date} → {end_date}")
    log.info(f"      Rozmiar kafelka: {edge_size}x{edge_size} px (10m/px = ~{edge_size*10/1000:.1f}km)")

    if progress_cb:
        progress_cb("Łączenie z Copernicus...", 5)

    da = cubo.create(
        lat=lat,
        lon=lon,
        collection="sentinel-2-l2a",
        bands=["B04", "B03", "B02", "B08"],  # Red, Green, Blue, NIR
        start_date=start_date,
        end_date=end_date,
        edge_size=edge_size,
        resolution=10,
    )

    n_samples = da.shape[0]
    log.info(f"      Znaleziono {n_samples} dostępnych scen")

    if n_samples == 0:
        raise ValueError("Brak scen dla podanego obszaru i okresu!")

    if progress_cb:
        progress_cb(f"Pobieranie {n_samples} scen...", 20)

    # Wybierz scenę z najmniejszym zachmurzeniem (środek przedziału)
    sample_idx = n_samples // 2
    log.info(f"      Wybrano scenę nr {sample_idx + 1}/{n_samples}")

    # Zachowaj georeferencję przed konwersją do numpy
    from geoexport import geo_from_cubo
    geo = geo_from_cubo(da)

    arr = (da[sample_idx].compute().to_numpy() / 10_000).astype("float32")
    tensor = torch.from_numpy(arr).float()
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    log.info(f"      Kształt tensora: {tensor.shape} (kanały x H x W)")
    if geo:
        log.info(f"      Georeferencja: {geo['crs']}, piksel {geo['res']} m")
    return tensor, sample_idx, geo


def download_sentinel2_stack(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    edge_size: int = DEFAULT_EDGE_SIZE,
    max_cloud: int = 20,
    progress_cb: Optional[Callable] = None,
) -> tuple[torch.Tensor, Optional[dict]]:
    """
    Faza 2 (MISR) — zwraca CAŁY stos czasowy zamiast jednej sceny.

    To te same dane co `download_sentinel2()`, tylko nie wyrzucamy pozostałych
    przelotów. Każdy przelot jest minimalnie przesunięty subpikselowo —
    z tych przesunięć MISR rekonstruuje realny detal (patrz WDROZENIE.md, Faza 2).

    Zwraca (stos [T, 4, H, W], geo) — reflektancja w [0, 1], geo = georeferencja
    (crs/west/north/res) lub None.
    """
    log.info(f"\n[MISR] Pobieranie STOSU czasowego Sentinel-2...")
    log.info(f"       Lokalizacja: lat={lat}, lon={lon}")
    log.info(f"       Okres: {start_date} → {end_date}  (max chmury < {max_cloud}%)")

    if progress_cb:
        progress_cb("Łączenie z Copernicus (stos czasowy)...", 5)

    da = cubo.create(
        lat=lat,
        lon=lon,
        collection="sentinel-2-l2a",
        bands=["B04", "B03", "B02", "B08"],  # Red, Green, Blue, NIR
        start_date=start_date,
        end_date=end_date,
        edge_size=edge_size,
        resolution=10,
        query={"eo:cloud_cover": {"lt": max_cloud}},  # odsiej mocno zachmurzone
    )

    n_scenes = da.shape[0]
    log.info(f"       Znaleziono {n_scenes} scen (chmury < {max_cloud}%)")
    if n_scenes == 0:
        raise ValueError(
            "Brak scen dla podanego obszaru/okresu/progu chmur! "
            "Zwiększ max_cloud lub poszerz zakres dat."
        )

    if progress_cb:
        progress_cb(f"Pobieranie {n_scenes} scen...", 20)

    from geoexport import geo_from_cubo
    geo = geo_from_cubo(da)

    arr = (da.compute().to_numpy() / 10_000).astype("float32")  # (T, 4, H, W)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    stack = torch.from_numpy(arr).float()

    log.info(f"       Kształt stosu: {stack.shape} (T x kanały x H x W)")
    return stack, geo


# ─────────────────────────────────────────────
# KROK 2 – SEN2SR: 10m → 2.5m
# ─────────────────────────────────────────────

def _load_sen2sr_model(device: torch.device):
    """Pobiera (jeśli trzeba) i ładuje skompilowany model SEN2SRLite."""
    if not (SEN2SR_MODEL_DIR / "mlm.json").exists():
        log.info(f"      Pobieranie modelu SEN2SRLite (pierwsze uruchomienie)...")
        mlstac.download(file=SEN2SR_MODEL_URL, output_dir=str(SEN2SR_MODEL_DIR))
    model = mlstac.load(str(SEN2SR_MODEL_DIR)).compiled_model(device=device)
    model.eval()
    return model


def run_misr_x2(
    stack: torch.Tensor,
    device: torch.device,
    progress_cb: Optional[Callable] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Zejście poniżej 2.5 m z realnej informacji wieloklatkowej (WDROZENIE.md).
    Schemat: każda klatka 10m → SEN2SR x4 → 2.5m; fuzja klatek SR:
        scale=1 → 2.5 m (czyste),   scale=2 → 1.25 m (realny detal subpikselowy).
    Zwalidowane w bramach (eval_misr_x2.py): +3.65 dB / +0.117 F1 vs naiwny upscale.

    Wejście: stos [T, 4, 128, 128]. Zwraca (sr_2.5m [4,512,512], sr_1.25m [4,1024,1024]).
    """
    from misr import misr_fuse
    log.info(f"\n[MISR x2] SEN2SR per-klatka ({stack.shape[0]} klatek) → fuzja x2...")
    if progress_cb:
        progress_cb("Ładowanie modelu SEN2SR...", 45)
    model = _load_sen2sr_model(device)

    # SEN2SR per-klatka. Model działa natywnie tylko na 128px (stała maska FFT),
    # więc dla większych kafelków kafelkujemy przez predict_large (jak run_sen2sr).
    import sen2sr as s2sr
    sr_frames = []
    with torch.no_grad():
        for t in range(stack.shape[0]):
            Xf = stack[t].to(device)
            if Xf.shape[1] <= 128 and Xf.shape[2] <= 128:
                sr = model(Xf[None]).squeeze(0)
            else:
                sr = s2sr.predict_large(model=model, X=Xf, overlap=32)
            sr_frames.append(sr.clamp(min=0).cpu())
    sr_stack = torch.stack(sr_frames)  # (T,4,H*4,W*4) @2.5m

    if progress_cb:
        progress_cb("MISR: fuzja na 2.5 m i 1.25 m...", 60)
    sr_2p5 = misr_fuse(sr_stack, scale=1, robust=True)   # (4,512,512)  2.5 m
    sr_1p25 = misr_fuse(sr_stack, scale=2, robust=True)  # (4,1024,1024) 1.25 m
    log.info(f"      MISR x2: 2.5m {tuple(sr_2p5.shape)}, 1.25m {tuple(sr_1p25.shape)}")
    return sr_2p5, sr_1p25


def run_sen2sr(
    tensor: torch.Tensor,
    device: torch.device,
    progress_cb: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Uruchamia SEN2SRLite NonReference_RGBN_x4.
    Wejście: tensor [4, H, W] (B04, B03, B02, B08) w zakresie [0, 1]
    Wyjście: tensor [4, H*4, W*4]
    """
    log.info(f"\n[2/3] SEN2SR — super-rozdzielczość x4 (10m → 2.5m)...")

    if progress_cb:
        progress_cb("Ładowanie modelu SEN2SR...", 45)

    model = _load_sen2sr_model(device)

    log.info(f"      Device: {device}")
    log.info(f"      Przetwarzanie {tensor.shape[1]}x{tensor.shape[2]}px kafelkami (128px + overlap)...")

    if progress_cb:
        progress_cb("Uruchamianie SEN2SR...", 55)

    import sen2sr as s2sr
    with torch.no_grad():
        if tensor.shape[1] <= 128 and tensor.shape[2] <= 128:
            result = model(tensor.to(device)[None]).squeeze(0)
        else:
            result = s2sr.predict_large(
                model=model,
                X=tensor.to(device),
                overlap=32,
            )

    log.info(f"      Wynik: {result.shape} → {result.shape[1]/4:.0f}m/px efektywnie")
    return result.cpu()


# ─────────────────────────────────────────────
# KROK 3 – super-image (EDSR): 2.5m → ~1.25m
# ─────────────────────────────────────────────

def run_superimage(
    tensor: torch.Tensor,
    device: torch.device,
    scale: int = 2,
    progress_cb: Optional[Callable] = None,
) -> np.ndarray:
    """
    Uruchamia super-image EDSR na obrazie RGB.
    Wejście: tensor [4, H, W] z SEN2SR
    Wyjście: ndarray [H*scale, W*scale, 3] uint8
    """
    log.info(f"\n[3/3] super-image (EDSR) — super-rozdzielczość x{scale} (2.5m → {2.5/scale:.2f}m)...")

    if progress_cb:
        progress_cb("Przygotowanie super-image (EDSR)...", 70)

    try:
        from super_image import EdsrModel, ImageLoader
    except ImportError:
        log.warning("      super-image niedostępny. Pomijam krok 3.")
        log.warning("      Zainstaluj: pip install super-image")
        rgb = tensor_to_rgb_uint8(tensor)
        img = Image.fromarray(rgb)
        w, h = img.size
        img_up = img.resize((w * scale, h * scale), Image.LANCZOS)
        return np.array(img_up)

    if progress_cb:
        progress_cb("Ładowanie modelu EDSR...", 75)

    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
    model = model.to(device)
    model.eval()

    rgb = tensor_to_rgb_uint8(tensor)
    pil_img = Image.fromarray(rgb)
    inputs = ImageLoader.load_image(pil_img).to(device)

    log.info(f"      Przetwarzanie obrazu {pil_img.width}x{pil_img.height}px...")

    if progress_cb:
        progress_cb("Uruchamianie EDSR...", 80)

    with torch.no_grad():
        preds = model(inputs)

    output = preds.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    log.info(f"      Wynik: {output.shape[1]}x{output.shape[0]}px")
    return output


# ─────────────────────────────────────────────
# GŁÓWNA FUNKCJA
# ─────────────────────────────────────────────

def run_pipeline(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    edge_size: int = DEFAULT_EDGE_SIZE,
    esrgan_scale: int = 2,
    use_second_stage: bool = False,   # Faza 0: EDSR domyślnie wyłączony (baseline = SEN2SR-only)
    use_misr: bool = False,           # Faza 2: fuzja MISR przed SEN2SR
    use_misr_x2: bool = False,        # zejście < 2.5 m: SEN2SR per-klatka → fuzja x2 → 1.25 m
    misr_max_cloud: int = 20,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Pełny pipeline. Zwraca słownik ze ścieżkami do wynikowych plików.

    Baseline (use_second_stage=False): tylko SEN2SR 10m → 2.5m.
    EDSR (krok 3) jest opcjonalny — zostawiony w kodzie, ale domyślnie OFF,
    bo dorysowuje fałszywą teksturę i psuje wierność spektralną (NDVI).

    use_misr=True (Faza 2): zamiast jednej sceny pobiera cały stos czasowy,
    koregistruje subpikselowo i robi fuzję robust → czystsza klatka 10 m na
    wejście SEN2SR. Wymaga ≥2 scen w zakresie dat (patrz WDROZENIE.md, Faza 2).
    """
    ensure_dirs()
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"\n{'='*50}")
    log.info(f"  Sentinel-2 Super-Resolution Pipeline")
    log.info(f"{'='*50}")
    log.info(f"  Device: {device}")

    # use_misr_x2 wymaga stosu czasowego (kilku klatek)
    need_stack = use_misr or use_misr_x2
    kept_frames = None  # zachowane klatki dla MISR x2

    # ── Krok 1: Pobierz dane ──
    if need_stack:
        # Faza 2: cały stos czasowy → koregistracja → fuzja robust
        from misr import misr_fuse, select_frames
        stack, geo = download_sentinel2_stack(
            lat=lat, lon=lon,
            start_date=start_date, end_date=end_date,
            edge_size=edge_size, max_cloud=misr_max_cloud,
            progress_cb=progress_cb,
        )
        if stack.shape[0] < 2:
            log.warning("      MISR: <2 scen — używam pojedynczej klatki.")
            raw_tensor = stack[0]
            kept_frames = stack
        else:
            if progress_cb:
                progress_cb("MISR: koregistracja + fuzja...", 25)
            kept, keep_idx = select_frames(stack)
            kept_frames = kept
            log.info(f"      MISR: {kept.shape[0]}/{stack.shape[0]} klatek po filtrze chmur")
            # scale=1 → czystsza klatka 10 m na wejście SEN2SR
            raw_tensor = misr_fuse(kept, scale=1, robust=True)
            log.info(f"      MISR: fuzja gotowa, kształt {tuple(raw_tensor.shape)}")
    else:
        raw_tensor, scene_idx, geo = download_sentinel2(
            lat=lat, lon=lon,
            start_date=start_date, end_date=end_date,
            edge_size=edge_size,
            progress_cb=progress_cb,
        )

    # Zapisz obraz PRZED (10m/px)
    rgb_before = tensor_to_rgb_uint8(raw_tensor)
    path_before = OUTPUT_DIR / "1_original_10m.png"
    label_before = "MISR 10 m/px" if use_misr else "ORYGINAŁ  10 m/px"
    save_image(rgb_before, path_before, label=label_before)

    if progress_cb:
        progress_cb("Zapisano oryginał...", 30)

    # ── Krok 2: SEN2SR x4 (+ opcjonalnie MISR x2 → 1.25 m) ──
    tensor_x2 = None  # wynik 1.25 m (jeśli MISR x2)
    if use_misr_x2 and kept_frames is not None and kept_frames.shape[0] >= 2:
        tensor_sr, tensor_x2 = run_misr_x2(kept_frames, device, progress_cb=progress_cb)
    else:
        if use_misr_x2:
            log.warning("      MISR x2: za mało klatek — zwykłe SEN2SR (2.5 m).")
        tensor_sr = run_sen2sr(raw_tensor, device, progress_cb=progress_cb)

    # Zapisz po SEN2SR (2.5m/px)
    rgb_sen2sr = tensor_to_rgb_uint8(tensor_sr)
    path_sen2sr = OUTPUT_DIR / "2_sen2sr_2.5m.png"
    save_image(rgb_sen2sr, path_sen2sr, label="SEN2SR  2.5 m/px")

    if progress_cb:
        progress_cb("Zapisano SEN2SR...", 65)

    # Wynik bazowy = SEN2SR (klucz "final" wskazuje ostatni etap pipeline)
    results = {
        "original":  str(path_before.resolve()),
        "sen2sr":    str(path_sen2sr.resolve()),
        "final":     str(path_sen2sr.resolve()),
    }

    # ── Eksport georeferencyjny + NDVI (Faza A — produkt rolniczy) ──
    if geo is not None:
        try:
            import geoexport as gx
            tr = gx.build_transform(geo, scale=4)   # SEN2SR x4 → piksel 2.5 m
            sr_np = tensor_sr.detach().cpu().numpy()

            path_tif = OUTPUT_DIR / "sen2sr_2.5m.tif"
            gx.save_geotiff(sr_np, geo["crs"], tr, path_tif,
                            band_names=["Red", "Green", "Blue", "NIR"])

            ndvi = gx.ndvi_array(tensor_sr)
            path_ndvi_tif = OUTPUT_DIR / "ndvi_2.5m.tif"
            path_ndvi_png = OUTPUT_DIR / "ndvi_2.5m.png"
            gx.save_ndvi_geotiff(ndvi, geo["crs"], tr, path_ndvi_tif)
            gx.save_ndvi_png(ndvi, path_ndvi_png)

            results["geotiff"] = str(path_tif.resolve())
            results["ndvi_tif"] = str(path_ndvi_tif.resolve())
            results["ndvi_png"] = str(path_ndvi_png.resolve())
            log.info(f"      Eksport: GeoTIFF + NDVI ({geo['crs']}, piksel 2.5 m)")
        except Exception as ex:
            log.warning(f"      Eksport GeoTIFF/NDVI pominięty: {ex}")
    else:
        log.info(f"      Brak georeferencji — eksport GeoTIFF pominięty.")

    # ── MISR x2 → wynik 1.25 m (zejście poniżej 2.5 m, realny detal) ──
    if tensor_x2 is not None:
        rgb_x2 = tensor_to_rgb_uint8(tensor_x2)
        path_x2 = OUTPUT_DIR / "3_misr_x2_1.25m.png"
        save_image(rgb_x2, path_x2, label="MISR x2  1.25 m/px")
        results["superimage"] = str(path_x2.resolve())   # 3. okno GUI
        results["final"] = results["superimage"]
        if geo is not None:
            try:
                import geoexport as gx
                tr8 = gx.build_transform(geo, scale=8)   # 10/8 = 1.25 m
                path_x2_tif = OUTPUT_DIR / "misr_x2_1.25m.tif"
                gx.save_geotiff(tensor_x2.detach().cpu().numpy(), geo["crs"], tr8,
                                path_x2_tif, band_names=["Red", "Green", "Blue", "NIR"])
                results["geotiff_x2"] = str(path_x2_tif.resolve())
                log.info(f"      Eksport: GeoTIFF 1.25 m ({geo['crs']})")
            except Exception as ex:
                log.warning(f"      Eksport GeoTIFF 1.25m pominięty: {ex}")

    # ── Krok 3 (OPCJONALNY): super-image EDSR x2 ──
    # Domyślnie wyłączony — patrz docstring i WDROZENIE.md (Faza 0).
    if use_second_stage:
        rgb_superimage = run_superimage(tensor_sr, device, scale=esrgan_scale, progress_cb=progress_cb)
        path_superimage = OUTPUT_DIR / f"3_superimage_{2.5/esrgan_scale:.2f}m.png"
        save_image(rgb_superimage, path_superimage, label=f"EDSR  {2.5/esrgan_scale:.2f} m/px")
        results["superimage"] = str(path_superimage.resolve())
        results["final"] = results["superimage"]
    else:
        log.info(f"\n[3/3] EDSR pominięty (baseline SEN2SR-only).")

    if progress_cb:
        progress_cb("Gotowe!", 100)

    elapsed = time.time() - t_start
    results["elapsed_s"] = elapsed

    log.info(f"\n{'='*50}")
    log.info(f"  Pipeline zakończony w {elapsed:.1f}s")
    log.info(f"  Wyniki w folderze: {OUTPUT_DIR.resolve()}")
    log.info(f"{'='*50}\n")

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    results = run_pipeline()
    print("\nWyniki:")
    for k, v in results.items():
        print(f"  {k}: {v}")
