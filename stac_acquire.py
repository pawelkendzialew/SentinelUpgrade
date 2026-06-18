"""
Pozyskiwanie danych przez STAC (odpowiednik procesu z depictAI) + NIR
=====================================================================
depictAI szuka scen w Microsoft Planetary Computer (STAC) i bierze warstwę
"visual" (3 pasma RGB, 8-bit, bez NIR). My robimy TO SAMO szukanie scen, ale
pobieramy **4 pasma reflektancji z NIR** (B04, B03, B02, B08) — czyli wejście
gotowe pod SEN2SR (+ NDVI).

Dwie funkcje:
  search_scenes(bbox, start, end, max_cloud)  -> lista scen [{id, datetime, cloud}]
                                                  posortowana od najczystszej
  download_scene(item, bbox, out_path)         -> 4-pasmowy GeoTIFF reflektancji

Zalety vs nasze dotychczasowe: WIDZIMY listę scen z % chmur i bierzemy
NAJCZYSTSZĄ (a nie środkową na ślepo).

Uruchom test:  python stac_acquire.py
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import planetary_computer as pc
import pystac_client
import stackstac
import rasterio
from rasterio.transform import from_bounds
from pyproj import Transformer

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = ["B04", "B03", "B02", "B08"]   # R, G, B, NIR — kolejność jak chce SEN2SR


def _catalog():
    return pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)


def search_scenes(bbox, start, end, max_cloud=80, limit=100):
    """
    Szuka scen Sentinel-2 L2A w obszarze (bbox = [lon_min,lat_min,lon_max,lat_max]).
    Zwraca listę {id, datetime, cloud, item} posortowaną od NAJCZYSTSZEJ.
    """
    cat = _catalog()
    search = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=limit,
    )
    scenes = []
    for it in search.items():
        scenes.append({
            "id": it.id,
            "datetime": it.properties.get("datetime", ""),
            "cloud": it.properties.get("eo:cloud_cover"),
            "item": it,
        })
    scenes.sort(key=lambda s: s["cloud"] if s["cloud"] is not None else 999)
    return scenes


def download_scene(item, bbox, out_path, resolution=10):
    """
    Pobiera 4 pasma (RGBN) sceny przyciętej do bbox i zapisuje GeoTIFF reflektancji
    (uint16 ×10000, jak natywny Sentinel). bbox = [lon_min,lat_min,lon_max,lat_max].
    """
    # UTM EPSG ze środka bbox (stackstac wymaga jawnego CRS dla S2 L2A)
    lon_c = (bbox[0] + bbox[2]) / 2
    lat_c = (bbox[1] + bbox[3]) / 2
    epsg_in = (32600 if lat_c >= 0 else 32700) + int((lon_c + 180) // 6) + 1

    da = stackstac.stack(
        [item], assets=BANDS, resolution=resolution, epsg=epsg_in,
        bounds_latlon=bbox, dtype="float64", fill_value=float("nan"),
        rescale=False,                      # surowe DN (reflektancja ×10000)
    ).isel(time=0)
    arr = da.compute().values               # (4, H, W) w skali ×10000
    arr = np.nan_to_num(arr, nan=0.0).astype("float32")

    epsg = int(da.coords["epsg"].item())
    xs = da.coords["x"].values
    ys = da.coords["y"].values
    res = float(abs(xs[1] - xs[0]))
    west = float(xs.min()) - res / 2
    east = float(xs.max()) + res / 2
    north = float(ys.max()) + res / 2
    south = float(ys.min()) - res / 2
    H, W = arr.shape[1], arr.shape[2]
    transform = from_bounds(west, south, east, north, W, H)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out16 = np.clip(arr, 0, 65535).astype("uint16")
    with rasterio.open(
        out_path, "w", driver="GTiff", height=H, width=W, count=4,
        dtype="uint16", crs=f"EPSG:{epsg}", transform=transform,
        nodata=0, compress="deflate",
    ) as dst:
        dst.write(out16)
        for i, name in enumerate(["red", "green", "blue", "nir"], start=1):
            dst.set_band_description(i, name)
    return {"crs": f"EPSG:{epsg}", "res": res, "shape": (H, W)}


def _region_tile_bboxes(bbox, edge):
    """Dzieli region (lon/lat bbox) na siatkę kafelków edge×edge px @10m (siatka UTM).
    Zwraca (epsg, [(r, c, tile_bbox_lonlat), ...])."""
    import math
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_c, lon_c = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    epsg = (32600 if lat_c >= 0 else 32700) + int((lon_c + 180) // 6) + 1
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    xs, ys = [], []
    for la in (lat_min, lat_max):
        for lo in (lon_min, lon_max):
            x, y = to_utm.transform(lo, la); xs.append(x); ys.append(y)
    min_e, max_e, min_n, max_n = min(xs), max(xs), min(ys), max(ys)
    step = edge * 10
    nx = max(1, math.ceil((max_e - min_e) / step))
    ny = max(1, math.ceil((max_n - min_n) / step))
    tiles = []
    for r in range(ny):
        n_top = max_n - r * step
        n_bot = n_top - step
        for c in range(nx):
            e_left = min_e + c * step
            e_right = e_left + step
            lo1, la1 = to_ll.transform(e_left, n_bot)
            lo2, la2 = to_ll.transform(e_right, n_top)
            tiles.append((r, c, [min(lo1, lo2), min(la1, la2),
                                 max(lo1, lo2), max(la1, la2)]))
    return epsg, tiles


def download_region_tiles(bbox, start, end, edge, out_dir, max_cloud=80,
                          progress_cb=None, should_stop=None):
    """
    Pobiera region jako siatkę surowych 4-pasmowych TIFF-ów (10 m, RGBN) — WSZYSTKIE
    z TEJ SAMEJ, najczystszej sceny (spójne, bez rozjazdu orbit/chmur). Wznawialne.
    Zwraca (liczba_zapisanych, info_o_scenie).
    """
    scenes = search_scenes(bbox, start, end, max_cloud=max_cloud)
    if not scenes:
        raise RuntimeError("Brak scen Sentinel-2 dla tego obszaru/dat.")
    best = scenes[0]
    _, tiles = _region_tile_bboxes(bbox, edge)
    total = len(tiles)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0, total, f"scena: {best['datetime'][:10]} ({best['cloud']:.1f}% chmur)")
    done = 0
    for i, (r, c, tb) in enumerate(tiles):
        if should_stop and should_stop():
            break
        p = out / f"tile_{r:03d}_{c:03d}.tif"
        if p.exists():
            done += 1
            if progress_cb:
                progress_cb(i + 1, total, f"kafelek {r},{c}: gotowy (pomijam)")
            continue
        try:
            download_scene(best["item"], tb, p)
            done += 1
            if progress_cb:
                progress_cb(i + 1, total, f"kafelek {r},{c}: pobrany ({done}/{total})")
        except Exception as ex:
            if progress_cb:
                progress_cb(i + 1, total, f"kafelek {r},{c}: blad {str(ex)[:30]}")
    return done, best


def acquire_cleanest(bbox, start, end, out_path, max_cloud=80):
    """
    Skrót: znajdź NAJCZYSTSZĄ scenę w obszarze i pobierz 4-pasmowy GeoTIFF (RGBN).
    bbox = [lon_min, lat_min, lon_max, lat_max]. Zwraca dict z info albo None.
    """
    scenes = search_scenes(bbox, start, end, max_cloud=max_cloud)
    if not scenes:
        return None
    best = scenes[0]
    info = download_scene(best["item"], bbox, out_path)
    info.update({"datetime": best["datetime"], "cloud": best["cloud"], "id": best["id"]})
    return info


def main():
    # Test: wsch.-centralna PL, lato 2023
    bbox = [22.00, 51.39, 22.04, 51.42]    # lon_min,lat_min,lon_max,lat_max (~3x3 km)
    print("Szukam scen (Planetary Computer STAC)...")
    scenes = search_scenes(bbox, "2023-06-01", "2023-08-31", max_cloud=80)
    print(f"Znaleziono {len(scenes)} scen. Najczystsze:")
    for s in scenes[:8]:
        print(f"  {s['datetime'][:10]}  chmury={s['cloud']:.1f}%  {s['id']}")
    if not scenes:
        return
    best = scenes[0]
    print(f"\nPobieram NAJCZYSTSZĄ: {best['datetime'][:10]} ({best['cloud']:.1f}% chmur)")
    Path("output").mkdir(exist_ok=True)
    info = download_scene(best["item"], bbox, "output/_stac_test.tif")
    with rasterio.open("output/_stac_test.tif") as ds:
        print(f"Zapisano: {ds.width}x{ds.height}, pasm={ds.count}, {ds.crs}, "
              f"piksel={ds.transform.a}m, pasma={ds.descriptions}")
        for i in range(1, 5):
            b = ds.read(i)
            print(f"  pasmo {i} ({ds.descriptions[i-1]}): min={b.min()} max={b.max()}")


if __name__ == "__main__":
    main()
