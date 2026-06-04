"""
Faza A — Eksport GeoTIFF z georeferencją + mapa NDVI
=====================================================
Pipeline domyślnie zapisywał tylko PNG (bez georeferencji). Dla rolnictwa wynik
musi być osadzony w przestrzeni (CRS + transform), żeby dało się go nałożyć na
mapę działek i policzyć statystyki per pole. Dodajemy też NDVI — dla monitoringu
upraw to sygnał #1 (kondycja wegetacji), a dokument stawia go jako priorytet.

Funkcje:
    geo_from_cubo(da)            -> dict(crs, west, north, res)   z DataArray cubo
    build_transform(geo, scale)  -> rasterio Affine (scale=4 dla SEN2SR x4)
    save_geotiff(arr, crs, tr)   -> wielopasmowy GeoTIFF (reflektancja float32)
    ndvi_array(tensor)           -> (H,W) NDVI z kanałów [R,...,NIR]
    save_ndvi_geotiff(...)       -> jednopasmowy NDVI GeoTIFF
    save_ndvi_png(...)           -> NDVI jako kolormapa RdYlGn (podgląd)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def geo_from_cubo(da) -> Optional[dict]:
    """
    Wyciąga georeferencję z DataArray zwróconego przez cubo.create().
    Zwraca dict(crs, west, north, res) albo None gdy brak metadanych.
    west/north = lewy-górny róg (nie środek piksela).
    """
    try:
        epsg = da.attrs.get("epsg")
        res = float(da.attrs.get("resolution"))
        xs = np.asarray(da.coords["x"].values, dtype="float64")
        ys = np.asarray(da.coords["y"].values, dtype="float64")
        # współrzędne to środki pikseli → przesuń o pół piksela do rogu
        west = float(xs.min()) - res / 2.0
        north = float(ys.max()) + res / 2.0
        return {"crs": f"EPSG:{epsg}", "west": west, "north": north, "res": res}
    except Exception:
        return None


def build_transform(geo: dict, scale: int = 1):
    """Affine transform; scale>1 zmniejsza piksel (SEN2SR x4 → scale=4)."""
    from rasterio.transform import from_origin
    res = geo["res"] / scale
    return from_origin(geo["west"], geo["north"], res, res)


def save_geotiff(arr: np.ndarray, crs: str, transform, path: Path,
                 band_names=None) -> None:
    """
    Zapisuje wielopasmowy GeoTIFF. arr: (C,H,W) float32 (reflektancja [0,1]).
    """
    import rasterio
    arr = np.asarray(arr, dtype="float32")
    if arr.ndim != 3:
        raise ValueError(f"Oczekiwano (C,H,W), dostalem {arr.shape}")
    c, h, w = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=c, dtype="float32",
        crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(arr)
        if band_names:
            for i, name in enumerate(band_names[:c], start=1):
                dst.set_band_description(i, name)


def ndvi_array(tensor: torch.Tensor, red_idx: int = 0, nir_idx: int = 3,
               eps: float = 1e-6) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). tensor (C,H,W) → (H,W) w [-1,1]."""
    arr = tensor.detach().cpu().numpy().astype("float32")
    red, nir = arr[red_idx], arr[nir_idx]
    out = (nir - red) / (nir + red + eps)
    # NDVI jest z definicji w [-1, 1] — klip chroni przed degeneracja mianownika
    return np.clip(out, -1.0, 1.0)


def save_ndvi_geotiff(ndvi: np.ndarray, crs: str, transform, path: Path) -> None:
    import rasterio
    ndvi = np.asarray(ndvi, dtype="float32")
    h, w = ndvi.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=1, dtype="float32",
        crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(ndvi[None])
        dst.set_band_description(1, "NDVI")


def save_ndvi_png(ndvi: np.ndarray, path: Path,
                  vmin: float = -0.2, vmax: float = 0.9) -> None:
    """NDVI jako kolormapa RdYlGn (czerwony=słaba wegetacja, zielony=bujna)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    from PIL import Image
    n = np.clip((ndvi - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgba = cm.get_cmap("RdYlGn")(n)
    rgb = (rgba[..., :3] * 255).astype("uint8")
    Image.fromarray(rgb).save(path)
