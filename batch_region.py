"""
Partia GeoTIFF-ów dla fragmentu Polski (do złożenia w QGIS)
===========================================================
Kładzie siatkę nx×ny przylegających kafelków na obszarze, dla każdego:
  pobiera najmniej zachmurzoną scenę Sentinel-2 → SEN2SR ×4 → zapis GeoTIFF (2.5 m).
Kafelki są georeferencyjne (CRS UTM) — w QGIS wczytujesz wszystkie naraz i
złożą się w jeden obraz regionu (lub zrób Raster → Miscellaneous → Merge).

Wynik: output/region/tile_R_C.tif (RGBN) + ndvi_R_C.tif

Uruchom:
    python batch_region.py                          # 2x2, wsch.-centralna PL
    python batch_region.py --lat 51.4 --lon 22.0 --nx 3 --ny 3 --edge 256
"""
import sys
import time
import argparse
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import cubo
from pyproj import Transformer

import geoexport as gx
from pipeline import run_sen2sr


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def tiles_from_bbox(lat_min, lat_max, lon_min, lon_max, edge=256):
    """
    Pokrywa prostokąt (bbox WGS84) siatką przylegających kafelków na siatce UTM.
    Zwraca (epsg, [(row, col, lat, lon), ...]) — środki kafelków.
    """
    lat_c, lon_c = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    epsg = utm_epsg(lon_c, lat_c)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    # rogi bbox -> UTM (bierzemy skrajne, by objąć cały obszar)
    xs, ys = [], []
    for la in (lat_min, lat_max):
        for lo in (lon_min, lon_max):
            x, y = to_utm.transform(lo, la)
            xs.append(x); ys.append(y)
    min_e, max_e, min_n, max_n = min(xs), max(xs), min(ys), max(ys)

    step = edge * 10  # m
    import math
    nx = max(1, math.ceil((max_e - min_e) / step))
    ny = max(1, math.ceil((max_n - min_n) / step))

    tiles = []
    for r in range(ny):                       # od góry (północ) w dół
        cy = max_n - (r + 0.5) * step
        for c in range(nx):                   # od lewej (zachód) w prawo
            cx = min_e + (c + 0.5) * step
            lon, lat = to_ll.transform(cx, cy)
            tiles.append((r, c, lat, lon))
    return epsg, tiles


def estimate_region(lat_min, lat_max, lon_min, lon_max, edge=256):
    """Szacuje (liczba_kafelkow, sekundy, MB_dysku) dla obszaru."""
    _, tiles = tiles_from_bbox(lat_min, lat_max, lon_min, lon_max, edge)
    n = len(tiles)
    return n, n * 20, n * 18   # ~20 s/kafelek (CPU), ~18 MB (RGBN+NDVI)


def region_target_date(lat, lon, start, end, edge=256, max_cloud=60):
    """
    Wyznacza JEDNĄ wspólną datę (timestamp) dla całego regionu — żeby wszystkie
    kafelki były z tego samego przelotu (spójna mozaika, bez różnych orbit/chmur
    per kafelek). Bierze środkową scenę z zakresu (filtr chmur tylko zgrubny).
    Zwraca numpy datetime64 albo None.
    """
    da = cubo.create(
        lat=lat, lon=lon, collection="sentinel-2-l2a",
        bands=["B04", "B03", "B02", "B08"],
        start_date=start, end_date=end, edge_size=edge, resolution=10,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    times = da.time.values
    if len(times) == 0:
        return None
    return times[len(times) // 2]


def download_tile_at_date(lat, lon, target_date, start, end, edge, retries=4):
    """Pobiera kafelek ze sceny NAJBLIŻSZEJ target_date (ta sama data co region)."""
    for attempt in range(retries):
        try:
            da = cubo.create(
                lat=lat, lon=lon, collection="sentinel-2-l2a",
                bands=["B04", "B03", "B02", "B08"],
                start_date=start, end_date=end, edge_size=edge, resolution=10,
            )
            geo = gx.geo_from_cubo(da)
            times = da.time.values
            if len(times) == 0:
                return None, None
            idx = int(np.argmin(np.abs(times - target_date)))
            arr = (da[idx].compute().to_numpy() / 10_000.0).astype("float32")
            t = torch.nan_to_num(torch.from_numpy(arr).float()).clamp(0, 1)
            return t, geo
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None, None


def process_region(lat_min, lat_max, lon_min, lon_max, edge=256,
                   start="2023-06-01", end="2023-08-31", use_finetuned=False,
                   out_before="output/region_10m", out_after="output/region_2.5m",
                   progress_cb=None, should_stop=None):
    """
    Przetwarza obszar SPÓJNIE: jedna wspólna data dla całego regionu, każdy kafelek
    zapisany w DWÓCH wersjach:
      - out_before: oryginał 10 m (przed modelem)
      - out_after:  SEN2SR 2.5 m (po modelu) + NDVI
    Wznawialne (pomija gotowe). Zwraca liczbę przetworzonych kafelków.
    """
    out_b, out_a = Path(out_before), Path(out_after)
    out_b.mkdir(parents=True, exist_ok=True)
    out_a.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsg, tiles = tiles_from_bbox(lat_min, lat_max, lon_min, lon_max, edge)
    total = len(tiles)

    lat_c, lon_c = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    target = region_target_date(lat_c, lon_c, start, end, edge)
    if target is None:
        raise RuntimeError("Brak scen Sentinel-2 w zakresie dat dla tego regionu.")
    if progress_cb:
        progress_cb(0, total, f"wspolna data regionu: {str(target)[:10]}")

    saved = 0
    for i, (r, c, lat, lon) in enumerate(tiles):
        if should_stop and should_stop():
            break
        pa = out_a / f"tile_{r:03d}_{c:03d}.tif"
        if pa.exists():
            saved += 1
            if progress_cb:
                progress_cb(i + 1, total, f"kafelek {r},{c}: gotowy (pomijam)")
            continue
        if progress_cb:
            progress_cb(i, total, f"kafelek {r},{c}: pobieram ({str(target)[:10]})...")
        t, geo = download_tile_at_date(lat, lon, target, start, end, edge)
        if t is None or geo is None:
            if progress_cb:
                progress_cb(i + 1, total, f"kafelek {r},{c}: brak danych")
            continue

        # PRZED — oryginał 10 m
        tr10 = gx.build_transform(geo, scale=1)
        gx.save_geotiff(t.numpy(), geo["crs"], tr10,
                        out_b / f"tile_{r:03d}_{c:03d}.tif",
                        band_names=["Red", "Green", "Blue", "NIR"])

        # PO — SEN2SR 2.5 m + NDVI
        sr = run_sen2sr(t, dev, use_finetuned=use_finetuned)
        tr25 = gx.build_transform(geo, scale=4)
        gx.save_geotiff(sr.numpy(), geo["crs"], tr25, pa,
                        band_names=["Red", "Green", "Blue", "NIR"])
        gx.save_ndvi_geotiff(gx.ndvi_array(sr), geo["crs"], tr25,
                             out_a / f"ndvi_{r:03d}_{c:03d}.tif")
        saved += 1
        if progress_cb:
            progress_cb(i + 1, total, f"kafelek {r},{c}: zapisany ({saved}/{total})")
    return saved


def download_one_clear(lat, lon, start, end, edge, max_cloud=15, max_scenes=6):
    """
    Pobiera JEDNĄ czystą scenę dla kafelka. Zwraca (tensor, geo).
    Próbuje kolejnych scen (od środka zakresu) aż któraś się pobierze — odporne
    na pojedyncze problematyczne sceny / timeouty Planetary Computer.
    """
    da = cubo.create(
        lat=lat, lon=lon, collection="sentinel-2-l2a",
        bands=["B04", "B03", "B02", "B08"],
        start_date=start, end_date=end, edge_size=edge, resolution=10,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    geo = gx.geo_from_cubo(da)
    n = da.shape[0]
    if n == 0:
        return None, None
    # kolejnosc prob: srodek, potem coraz dalej (najlepsza widocznosc w srodku lata)
    order = sorted(range(n), key=lambda i: abs(i - n // 2))[:max_scenes]
    last = None
    for k, idx in enumerate(order):
        try:
            arr = (da[idx].compute().to_numpy() / 10_000.0).astype("float32")
            t = torch.nan_to_num(torch.from_numpy(arr).float())
            return t.clamp(0, 1), geo
        except Exception as ex:
            last = ex
            print(f"      scena {idx} nieudana ({str(ex)[:45]}), probuje inna...")
            time.sleep(3)
    raise last


def run_region(lat0, lon0, nx, ny, edge, start, end, use_finetuned):
    out = Path("output/region")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epsg = utm_epsg(lon0, lat0)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(lon0, lat0)
    step = edge * 10  # m (kafelek przylega do sąsiada)

    print(f"Region wokół lat={lat0} lon={lon0}  ({epsg})  siatka {nx}x{ny}  "
          f"kafelek {edge}px = {step/1000:.1f} km  -> obszar "
          f"{nx*step/1000:.1f}x{ny*step/1000:.1f} km")
    print(f"Wagi: {'dostrojone PL (GUGiK)' if use_finetuned else 'bazowe SEN2SR'}\n")

    saved = 0
    for r in range(ny):
        for c in range(nx):
            p_tif = out / f"tile_{r}_{c}.tif"
            if p_tif.exists():
                print(f"  tile {r}_{c}: juz istnieje, pomijam")
                saved += 1
                continue
            ox = cx + (c - (nx - 1) / 2) * step
            oy = cy - (r - (ny - 1) / 2) * step
            lon, lat = to_ll.transform(ox, oy)
            try:
                t, geo = download_one_clear(lat, lon, start, end, edge)
            except Exception as ex:
                print(f"  tile {r}_{c}: blad pobierania ({str(ex)[:60]})")
                continue
            if t is None or geo is None:
                print(f"  tile {r}_{c} ({lat:.3f},{lon:.3f}): brak czystych scen")
                continue

            sr = run_sen2sr(t, dev, use_finetuned=use_finetuned)
            tr = gx.build_transform(geo, scale=4)
            gx.save_geotiff(sr.numpy(), geo["crs"], tr, p_tif,
                            band_names=["Red", "Green", "Blue", "NIR"])
            gx.save_ndvi_geotiff(gx.ndvi_array(sr), geo["crs"], tr,
                                 out / f"ndvi_{r}_{c}.tif")
            saved += 1
            print(f"  tile {r}_{c} ({lat:.3f},{lon:.3f}): {geo['crs']} "
                  f"-> {p_tif.name}  ({sr.shape[1]}x{sr.shape[2]} px @2.5m)")

    print(f"\nZapisano {saved}/{nx*ny} kafelkow do {out.resolve()}")
    print("W QGIS: wczytaj wszystkie tile_*.tif (zloza sie po georeferencji),")
    print("albo Raster -> Miscellaneous -> Merge, by zrobic jeden mozaikowy GeoTIFF.")


def main():
    ap = argparse.ArgumentParser(description="Partia GeoTIFF dla regionu PL")
    ap.add_argument("--lat", type=float, default=51.40)   # wsch.-centralna PL (Lubelskie)
    ap.add_argument("--lon", type=float, default=22.00)
    ap.add_argument("--nx", type=int, default=2)
    ap.add_argument("--ny", type=int, default=2)
    ap.add_argument("--edge", type=int, default=256)
    ap.add_argument("--start", default="2023-06-01")
    ap.add_argument("--end", default="2023-08-31")
    ap.add_argument("--finetuned", action="store_true", help="wagi dostrojone do PL")
    args = ap.parse_args()
    run_region(args.lat, args.lon, args.nx, args.ny, args.edge,
               args.start, args.end, args.finetuned)


if __name__ == "__main__":
    main()
