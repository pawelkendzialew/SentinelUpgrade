"""
Benchmark procesu: pomiar czasów pobierania + SEN2SR dla różnych rozmiarów.
==========================================================================
Mierzy:
  - SEN2SR (CPU): syntetyczne tensory 128/256/512 px, kilka powtórzeń (czysty compute)
  - Pobieranie (sieć): realne sceny Sentinel-2 przez cubo, kilka powtórzeń (zmienne)
  - Rozmiar wynikowego GeoTIFF na dysku
Zapisuje wyniki do output/benchmark_data.json (potem składamy ANALIZA_PROCESU.md).

Uruchom:  python benchmark.py
"""
import sys
import time
import json
import statistics as stats
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch

OUT = Path("output")
OUT.mkdir(exist_ok=True)

EDGES = [128, 256, 512]          # px @ 10 m
PL_LAT, PL_LON = 51.40, 22.00    # wsch.-centralna PL
START, END = "2023-06-01", "2023-08-31"


def time_sen2sr(edges, reps=3):
    """Czas SEN2SR na syntetycznym wejściu (czysty CPU, bez sieci)."""
    from folder_sr import sen2sr_tiled
    dev = torch.device("cpu")
    res = {}
    for e in edges:
        ts = []
        for _ in range(reps):
            x = torch.rand(4, e, e).clamp(0, 1)
            t0 = time.time()
            _ = sen2sr_tiled(x, dev, use_finetuned=False)
            ts.append(time.time() - t0)
        res[e] = {"min": min(ts), "avg": stats.mean(ts), "max": max(ts),
                  "reps": reps}
        print(f"  SEN2SR {e}px: {res[e]['avg']:.1f}s avg "
              f"({res[e]['min']:.1f}-{res[e]['max']:.1f})")
    return res


def time_download(edges, reps=2):
    """Czas pobrania jednej sceny dla danego rozmiaru (sieć — zmienne)."""
    from batch_region import download_one_clear
    res = {}
    for e in edges:
        ts = []
        for _ in range(reps):
            try:
                t0 = time.time()
                t, geo = download_one_clear(PL_LAT, PL_LON, START, END, e)
                if t is not None:
                    ts.append(time.time() - t0)
            except Exception as ex:
                print(f"  pobieranie {e}px: blad ({str(ex)[:40]})")
        if ts:
            res[e] = {"min": min(ts), "avg": stats.mean(ts), "max": max(ts),
                      "reps": len(ts)}
            print(f"  pobieranie {e}px: {res[e]['avg']:.1f}s avg "
                  f"({res[e]['min']:.1f}-{res[e]['max']:.1f})")
        else:
            res[e] = None
            print(f"  pobieranie {e}px: brak udanych prob")
    return res


def disk_sizes(edges):
    """Rzeczywisty rozmiar GeoTIFF (RGBN) na dysku dla danego rozmiaru wejścia."""
    import geoexport as gx
    from rasterio.transform import from_origin
    res = {}
    for e in edges:
        sr = torch.rand(4, e * 4, e * 4)   # wynik SEN2SR x4
        tr = from_origin(500000, 5600000, 2.5, 2.5)
        p = OUT / f"_bench_{e}.tif"
        gx.save_geotiff(sr.numpy(), "EPSG:32634", tr, p,
                        band_names=["R", "G", "B", "NIR"])
        mb = p.stat().st_size / 1024 / 1024
        res[e] = round(mb, 1)
        p.unlink()
        print(f"  dysk {e}px -> GeoTIFF {res[e]} MB")
    return res


def main():
    print("=" * 60)
    print("  BENCHMARK procesu (CPU + siec)")
    print("=" * 60)
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"  Urzadzenie: {device}\n")

    print("[1/3] SEN2SR (syntetyczne, czysty compute):")
    sen2sr = time_sen2sr(EDGES)
    print("\n[2/3] Rozmiar GeoTIFF na dysku:")
    disk = disk_sizes(EDGES)
    print("\n[3/3] Pobieranie (realne sceny, siec):")
    download = time_download(EDGES)

    data = {
        "device": device,
        "edges": EDGES,
        "sen2sr_s": sen2sr,
        "download_s": download,
        "disk_mb": disk,
        "location": {"lat": PL_LAT, "lon": PL_LON},
        "date_range": [START, END],
    }
    p = OUT / "benchmark_data.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nZapisano: {p.resolve()}")


if __name__ == "__main__":
    main()
