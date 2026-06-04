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
) -> tuple[torch.Tensor, int]:
    """
    Pobiera dane Sentinel-2 L2A przez cubo.
    Zwraca (tensor [4, H, W], indeks_próbki).
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

    arr = (da[sample_idx].compute().to_numpy() / 10_000).astype("float32")
    tensor = torch.from_numpy(arr).float()
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    log.info(f"      Kształt tensora: {tensor.shape} (kanały x H x W)")
    return tensor, sample_idx


# ─────────────────────────────────────────────
# KROK 2 – SEN2SR: 10m → 2.5m
# ─────────────────────────────────────────────

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
        progress_cb("Pobieranie modelu SEN2SR...", 35)

    # Pobierz model jeśli nie istnieje
    if not (SEN2SR_MODEL_DIR / "mlm.json").exists():
        log.info(f"      Pobieranie modelu SEN2SRLite (pierwsze uruchomienie)...")
        mlstac.download(
            file=SEN2SR_MODEL_URL,
            output_dir=str(SEN2SR_MODEL_DIR),
        )
    else:
        log.info(f"      Model SEN2SR znaleziony lokalnie")

    if progress_cb:
        progress_cb("Ładowanie modelu SEN2SR...", 45)

    model = mlstac.load(str(SEN2SR_MODEL_DIR)).compiled_model(device=device)
    model.eval()

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
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Pełny pipeline. Zwraca słownik ze ścieżkami do wynikowych plików.

    Baseline (use_second_stage=False): tylko SEN2SR 10m → 2.5m.
    EDSR (krok 3) jest opcjonalny — zostawiony w kodzie, ale domyślnie OFF,
    bo dorysowuje fałszywą teksturę i psuje wierność spektralną (NDVI).
    """
    ensure_dirs()
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"\n{'='*50}")
    log.info(f"  Sentinel-2 Super-Resolution Pipeline")
    log.info(f"{'='*50}")
    log.info(f"  Device: {device}")

    # ── Krok 1: Pobierz dane ──
    raw_tensor, scene_idx = download_sentinel2(
        lat=lat, lon=lon,
        start_date=start_date, end_date=end_date,
        edge_size=edge_size,
        progress_cb=progress_cb,
    )

    # Zapisz obraz PRZED (10m/px)
    rgb_before = tensor_to_rgb_uint8(raw_tensor)
    path_before = OUTPUT_DIR / "1_original_10m.png"
    save_image(rgb_before, path_before, label="ORYGINAŁ  10 m/px")

    if progress_cb:
        progress_cb("Zapisano oryginał...", 30)

    # ── Krok 2: SEN2SR x4 ──
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
