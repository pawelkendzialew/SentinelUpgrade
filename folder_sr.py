"""
Przetwarzanie folderu gotowych TIFF-ów przez SEN2SR
====================================================
Wskazujesz folder z TIFF-ami (+ sidecar .aux.xml), a skrypt:
  - czyta każdy TIFF (zachowuje CRS + transform + układ pasm),
  - układa pasma w kolejność wymaganą przez SEN2SR (R,G,B,NIR),
  - przepuszcza przez SEN2SR ×4,
  - zapisuje wynik w TYM SAMYM formacie (uint16 ×10000, ten sam układ pasm,
    nodata) do NOWEGO folderu — piksel /4,
  - KOPIUJE pliki towarzyszące (.aux.xml, .xml, .tfw, ...) obok nowego TIFF.

UWAGA: SEN2SR jest trenowany na Sentinel-2 10 m → 2.5 m. Na danych o innym
pikselu (np. PlanetScope 3 m) wynik jest poza-projektowy — używaj świadomie.

Funkcja `process_tiff_folder(...)` jest wołana z GUI (przycisk "Przetwórz folder").
"""
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import rasterio
from rasterio.transform import Affine

from pipeline import _load_sen2sr_model


def sen2sr_tiled(rgbn: torch.Tensor, device, use_finetuned=False, tile=128):
    """
    SEN2SR ×4 na obrazie dowolnego rozmiaru (też niekwadratowym).
    Własne kafelkowanie 128px (model wymaga 128 — stała maska FFT). Brzegi
    dopełniane refleksją. Omija buga osi w sen2sr.predict_large.
    Wejście (4,H,W) [0,1] → wyjście (4, H*4, W*4).
    """
    model = _load_sen2sr_model(device, use_finetuned=use_finetuned)
    C, H, W = rgbn.shape
    Hp = math.ceil(H / tile) * tile
    Wp = math.ceil(W / tile) * tile
    pad_mode = "reflect" if (Hp - H < H and Wp - W < W) else "replicate"
    x = F.pad(rgbn[None], (0, Wp - W, 0, Hp - H), mode=pad_mode)[0]
    out = torch.zeros((C, Hp * 4, Wp * 4), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, Hp, tile):
            for j in range(0, Wp, tile):
                chunk = x[:, i:i + tile, j:j + tile]
                sr = model(chunk[None].to(device)).squeeze(0).clamp(min=0).cpu()
                out[:, i * 4:(i + tile) * 4, j * 4:(j + tile) * 4] = sr
    return out[:, :H * 4, :W * 4]

# Nazwy pasm -> rola. SEN2SR chce kolejnosci [R, G, B, NIR].
_ROLE = {
    "red": "R", "r": "R", "b04": "R",
    "green": "G", "g": "G", "b03": "G",
    "blue": "B", "b": "B", "b02": "B",
    "nir": "N", "n": "N", "b08": "N", "near-infrared": "N",
}


def rgbn_order(descriptions, n_bands):
    """
    Zwraca (order, names) gdzie order to indeksy pasm wejściowych w kolejności
    [R,G,B,NIR]. Wykrywa po opisach pasm; fallback: zakłada [R,G,B,NIR] = [0,1,2,3].
    """
    roles = {}
    if descriptions and all(descriptions[:n_bands]):
        for i, d in enumerate(descriptions[:n_bands]):
            r = _ROLE.get(str(d).strip().lower())
            if r:
                roles[r] = i
    if all(k in roles for k in ("R", "G", "B", "N")):
        return [roles["R"], roles["G"], roles["B"], roles["N"]], descriptions
    # fallback: pierwsze 4 pasma jako R,G,B,NIR
    return [0, 1, 2, 3][:n_bands], descriptions


def process_one(tif_path: Path, out_path: Path, model_device, use_finetuned):
    with rasterio.open(tif_path) as src:
        arr = src.read().astype("float32")        # (bands, H, W)
        crs = src.crs
        transform = src.transform
        nodata = src.nodata
        descs = list(src.descriptions)
        in_dtype = src.dtypes[0]
        count = src.count

    if count < 4:
        raise ValueError(f"{tif_path.name}: ma {count} pasm, SEN2SR wymaga 4 (RGBN)")

    order, descs = rgbn_order(descs, count)

    # skala -> reflektancja [0,1]
    scale = 10000.0 if np.nanmax(arr) > 1.5 else 1.0
    rgbn = np.clip(arr[order] / scale, 0, 1)       # (4,H,W) [R,G,B,NIR]

    # maska nodata (gdzie WSZYSTKIE pasma == nodata/0)
    nd = nodata if nodata is not None else 0
    mask = np.all(arr == nd, axis=0)               # (H,W) True = nodata

    tensor = torch.from_numpy(rgbn).float()
    sr = sen2sr_tiled(tensor, model_device, use_finetuned=use_finetuned)  # (4, H*4, W*4)

    # powrot do oryginalnej kolejnosci pasm (zeby wyjscie bylo "takie samo")
    inv = [0, 0, 0, 0]
    for out_pos, in_idx in enumerate(order):       # order[out_pos]=in_idx
        inv[in_idx] = out_pos                       # gdzie w sr siedzi pasmo in_idx
    out_bands = sr[inv]                              # (bands_orig_order, H*4, W*4)

    # przywroc nodata (maska x4)
    mask_t = torch.from_numpy(mask.astype("float32"))[None, None]
    mask_up = F.interpolate(mask_t, scale_factor=4, mode="nearest")[0, 0] > 0.5
    out_bands[:, mask_up] = 0.0

    # zapis w formacie wejscia
    if "int" in in_dtype:
        out_np = np.clip(out_bands.numpy() * scale, 0, 65535).astype(in_dtype)
    else:
        out_np = out_bands.numpy().astype(in_dtype)

    out_tr = Affine(transform.a / 4, transform.b, transform.c,
                    transform.d, transform.e / 4, transform.f)
    h, w = out_np.shape[1], out_np.shape[2]
    with rasterio.open(
        out_path, "w", driver="GTiff", height=h, width=w, count=out_np.shape[0],
        dtype=in_dtype, crs=crs, transform=out_tr,
        nodata=(nd if nodata is not None else None), compress="deflate",
    ) as dst:
        dst.write(out_np)
        for i, d in enumerate(descs[:out_np.shape[0]], start=1):
            if d:
                dst.set_band_description(i, d)


def copy_sidecars(tif_path: Path, out_dir: Path):
    """Kopiuje pliki towarzyszace (np. .tif.aux.xml, .xml, .tfw) obok wyniku."""
    copied = 0
    for f in tif_path.parent.iterdir():
        if f == tif_path or not f.is_file():
            continue
        # sidecar = ten sam rdzen nazwy (np. 'X.tif.aux.xml' albo 'X.xml')
        if f.name.startswith(tif_path.name) or f.stem == tif_path.stem:
            if f.suffix.lower() in (".tif", ".tiff"):
                continue
            shutil.copy2(f, out_dir / f.name)
            copied += 1
    return copied


def process_tiff_folder(in_dir, out_dir, use_finetuned=False,
                        progress_cb=None, should_stop=None):
    """
    Przetwarza wszystkie TIFF-y z in_dir przez SEN2SR -> out_dir (ten sam format),
    kopiuje sidecary. Wznawialne (pomija juz zrobione). Zwraca liczbe przetworzonych.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tifs = sorted([p for p in in_dir.iterdir()
                   if p.suffix.lower() in (".tif", ".tiff") and p.is_file()])
    total = len(tifs)
    done = 0
    for i, tif in enumerate(tifs):
        if should_stop and should_stop():
            break
        out_tif = out_dir / tif.name
        if out_tif.exists():
            done += 1
            if progress_cb:
                progress_cb(i + 1, total, f"{tif.name}: gotowy (pomijam)")
            continue
        if progress_cb:
            progress_cb(i, total, f"{tif.name}: przetwarzam...")
        try:
            process_one(tif, out_tif, device, use_finetuned)
            n_side = copy_sidecars(tif, out_dir)
            done += 1
            if progress_cb:
                progress_cb(i + 1, total, f"{tif.name}: OK (+{n_side} sidecar)")
        except Exception as ex:
            if progress_cb:
                progress_cb(i + 1, total, f"{tif.name}: BLAD {str(ex)[:40]}")
    return done
