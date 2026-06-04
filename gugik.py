"""
Faza 3 / B — Pobieranie realnego ortofoto GUGiK (polskie HR) + degradacja
==========================================================================
Pobiera DARMOWE ortofoto RGB 25 cm z GUGiK przez publiczny WMS i buduje pary
LR–HR przepisem SEN2NAIP (degradacja HR → syntetyczny Sentinel).

WAŻNE — NIR (do zapamietania na przyszlosc):
    GUGiK publicznie udostepnia tylko RGB (warstwa "Raster"). CIR/NIR jest w
    ortofoto ARCHIWALNYM (darmowe, ale czesciowe pokrycie i upierdliwy dostep —
    brak czystego WFS). Dlatego TERAZ trenujemy strukture RGB polskich pol, a
    NIR jest wykluczony ze straty (placeholder). Gdy zdobedziemy arkusze CIR,
    wystarczy podmienic kanal NIR i wlaczyc go do straty — patrz NIR_TODO nizej.

WMS (poprawka: wersja 1.1.1 omija pulapke kolejnosci osi EPSG:2180):
    https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution

Uruchomienie (test pobierania kilku kafelkow):
    python gugik.py
"""

from typing import Optional

import io
import numpy as np
import torch
import torch.nn.functional as F

WMS_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"

# NIR_TODO: gdy bedzie CIR — osobny WMS/arkusz, kanal NIR/podczerwien.
NIR_AVAILABLE = False


def fetch_ortho_rgb(lat: float, lon: float, size_px: int = 512,
                    res_m: float = 2.5, timeout: int = 60) -> Optional[torch.Tensor]:
    """
    Pobiera kafelek ortofoto RGB GUGiK wokol (lat, lon).
    size_px x size_px pikseli przy res_m metrow/piksel.
    Domyslnie 512 px @ 2.5 m = 1280 m (= cel HR dla SEN2SR x4).
    Zwraca tensor (3, H, W) w [0,1] lub None gdy blad/pustka.
    """
    import requests
    from pyproj import Transformer
    from PIL import Image

    tr = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    x, y = tr.transform(lon, lat)
    half = size_px * res_m / 2.0
    bbox = f"{x-half},{y-half},{x+half},{y+half}"

    try:
        r = requests.get(WMS_URL, params={
            "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.1.1",
            "LAYERS": "Raster", "STYLES": "", "SRS": "EPSG:2180",
            "BBOX": bbox, "WIDTH": str(size_px), "HEIGHT": str(size_px),
            "FORMAT": "image/png",
        }, timeout=timeout)
        if "image" not in r.headers.get("content-type", ""):
            return None
        im = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype="float32") / 255.0
        t = torch.from_numpy(im).permute(2, 0, 1)  # (3,H,W)
        # odrzuc puste/jednolite kafelki (poza pokryciem)
        if float(t.std()) < 0.02:
            return None
        return t
    except Exception:
        return None


def fetch_tile_crops(lat: float, lon: float, tile_px: int = 1024,
                     crop_px: int = 512, res_m: float = 2.5,
                     timeout: int = 90) -> list:
    """
    Pobiera JEDEN wiekszy kafelek i tnie go na wiele wycinkow crop_px x crop_px
    (wiecej danych przy mniejszej liczbie zapytan WMS). Odrzuca puste wycinki.
    Zwraca liste tensorow (3, crop_px, crop_px) w [0,1].
    """
    import requests
    from pyproj import Transformer
    from PIL import Image

    tr = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    x, y = tr.transform(lon, lat)
    half = tile_px * res_m / 2.0
    bbox = f"{x-half},{y-half},{x+half},{y+half}"
    try:
        r = requests.get(WMS_URL, params={
            "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.1.1",
            "LAYERS": "Raster", "STYLES": "", "SRS": "EPSG:2180",
            "BBOX": bbox, "WIDTH": str(tile_px), "HEIGHT": str(tile_px),
            "FORMAT": "image/png",
        }, timeout=timeout)
        if "image" not in r.headers.get("content-type", ""):
            return []
        im = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"),
                      dtype="float32") / 255.0
        full = torch.from_numpy(im).permute(2, 0, 1)  # (3, tile, tile)
    except Exception:
        return []

    crops = []
    n = tile_px // crop_px
    for i in range(n):
        for j in range(n):
            c = full[:, i * crop_px:(i + 1) * crop_px, j * crop_px:(j + 1) * crop_px]
            if float(c.std()) >= 0.04:   # odrzuc pustke/jednolite (las, woda, brak)
                crops.append(c)
    return crops


def harmonize_to_reflectance(rgb: torch.Tensor,
                             lo: float = 0.02, hi: float = 0.22) -> torch.Tensor:
    """
    Harmonizacja radiometryczna (przyblizona) — najwazniejszy krok SEN2NAIP.
    Ortofoto to 8-bit RGB (nie kalibrowana reflektancja). Mapujemy percentyle
    2/98 kazdego kanalu na typowy zakres reflektancji powierzchni Sentinela
    [lo, hi], zeby model dostal wejscie w spodziewanej skali.
    """
    out = torch.empty_like(rgb)
    for c in range(rgb.shape[0]):
        ch = rgb[c]
        p2, p98 = torch.quantile(ch, 0.02), torch.quantile(ch, 0.98)
        scaled = (ch - p2) / (p98 - p2 + 1e-6)
        out[c] = (scaled.clamp(0, 1) * (hi - lo) + lo)
    return out.clamp(0, 1)


def synth_lr_from_hr(hr_rgb: torch.Tensor, scale: int = 4,
                     blur_sigma: float = 1.0, noise: float = 0.01,
                     seed: Optional[int] = None) -> torch.Tensor:
    """
    Przepis SEN2NAIP: HR (2.5 m) -> syntetyczny Sentinel LR (10 m).
      1) rozmycie Gaussa (symuluje PSF sensora),
      2) downsample x scale (srednia),
      3) szum gaussowski.
    hr_rgb: (3,H,W) w skali reflektancji. Zwraca (3, H/scale, W/scale).
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = hr_rgb[None]  # (1,3,H,W)

    # 1) rozmycie Gaussa (separowalne)
    k = max(3, int(2 * round(2 * blur_sigma) + 1))
    ax = torch.arange(k) - k // 2
    kern1d = torch.exp(-(ax ** 2) / (2 * blur_sigma ** 2))
    kern1d = (kern1d / kern1d.sum()).float()
    kern = (kern1d[:, None] * kern1d[None, :])[None, None]
    kern = kern.repeat(3, 1, 1, 1)
    xb = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), kern, groups=3)

    # 2) downsample
    lr = F.avg_pool2d(xb, kernel_size=scale)

    # 3) szum
    lr = lr + noise * torch.randn(lr.shape, generator=g)
    return lr[0].clamp(0, 1)


def make_pair_from_raw(raw: torch.Tensor, scale: int = 4,
                       seed: Optional[int] = None):
    """Z surowego wycinka RGB [0,1] buduje (lr, hr) zharmonizowane."""
    hr = harmonize_to_reflectance(raw)
    lr = synth_lr_from_hr(hr, scale=scale, seed=seed)
    return lr, hr


def make_pair(lat: float, lon: float, hr_px: int = 512, scale: int = 4,
              seed: Optional[int] = None):
    """
    Zwraca (lr_rgb [3,128,128], hr_rgb [3,512,512]) zharmonizowane — albo None.
    """
    raw = fetch_ortho_rgb(lat, lon, size_px=hr_px, res_m=10.0 / scale)
    if raw is None:
        return None
    return make_pair_from_raw(raw, scale=scale, seed=seed)


# Polskie obszary rolnicze (rozne regiony — roznorodnosc pol)
PL_AGRI_COORDS = [
    (51.50, 17.50), (52.20, 17.00), (52.80, 16.60), (50.80, 18.20),
    (53.10, 18.50), (51.90, 19.10), (50.55, 22.00), (52.50, 20.50),
    (53.50, 17.80), (51.20, 22.50), (52.00, 21.20), (50.35, 19.50),
    (52.65, 19.70), (51.05, 16.80), (53.30, 19.90), (50.95, 20.60),
    (52.35, 18.30), (51.65, 18.90), (52.90, 18.10), (50.70, 17.30),
    (51.35, 19.80), (52.10, 20.10), (53.00, 17.20), (51.80, 16.40),
    (50.45, 21.10), (52.75, 21.00), (51.15, 20.30), (53.20, 16.90),
    (51.55, 20.90), (52.45, 16.80), (50.65, 18.80), (53.40, 18.70),
    (51.75, 17.90), (52.30, 19.40), (50.50, 20.10), (51.95, 22.10),
    (53.05, 20.20), (50.85, 21.50), (52.55, 18.00), (51.30, 18.40),
]


def main():
    print("=" * 56)
    print("  GUGiK — test pobierania ortofoto RGB (polskie HR)")
    print("=" * 56)
    ok = 0
    for i, (lat, lon) in enumerate(PL_AGRI_COORDS[:6]):
        t = fetch_ortho_rgb(lat, lon, size_px=512, res_m=2.5)
        if t is None:
            print(f"  [{i}] ({lat},{lon})  BRAK / pustka")
        else:
            print(f"  [{i}] ({lat},{lon})  OK  {tuple(t.shape)}  "
                  f"mean={float(t.mean()):.3f} std={float(t.std()):.3f}")
            ok += 1
    print(f"\n  Pobrano {ok}/6 kafelkow. NIR niedostepny (RGB only) — patrz NIR_TODO.")


if __name__ == "__main__":
    main()
