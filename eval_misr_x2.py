"""
Zejscie ponizej 2.5 m — MISR x2 (walidacja w bramach)
======================================================
Pomysl: zejdz ponizej 2.5 m UCZCIWIE — z realnej informacji wieloklatkowej,
nie z halucynacji modelu. Schemat "SR-per-klatka -> fuzja x2":

    T klatek 10 m  --SEN2SR x4-->  T obrazow 2.5 m (kazdy z innej akwizycji,
                                   z innym subpikselowym przesunieciem)
                   --MISR fuzja x2--> 1.25 m

Detal ponizej 2.5 m pochodzi z roznic subpikselowych miedzy przelotami (realne),
a fuzja median TLUMI niespojna halucynacje per-klatka (zostaje wspolna struktura).

Walidacja (uczciwa, z realna prawda HR): bierzemy ortofoto GUGiK 25 cm jako
prawde przy 1.25 m, generujemy z niego stos zaszumionych/przesunietych klatek
10 m, przepuszczamy przez pipeline i porownujemy 1.25 m do prawdy.

Bramy (porownanie metod do HR 1.25 m):
    (1) SEN2SR + bicubic x2   — naiwne powiekszenie 2.5 m -> 1.25 m (baseline)
    (2) MISR x2 (SR->fuzja)   — nasz kandydat
PSNR(RGB) i delineacja F1. Jesli (2) > (1) -> realne, wierne zejscie < 2.5 m.

Uruchomienie:
    python eval_misr_x2.py            # 6 lokacji, 12 klatek
"""

import sys
import json
import time
import argparse
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import shift as ndi_shift

from measure import load_model, edge_map, boundary_f1
from misr import misr_fuse
from gugik import fetch_ortho_rgb, harmonize_to_reflectance, PL_AGRI_COORDS

OUTPUT_DIR = Path("output")


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10.0 * float(np.log10(1.0 / mse))


def make_lr_stack_from_hr(hr4: torch.Tensor, n_frames: int, total_scale: int = 8,
                          max_shift: float = 4.0, noise: float = 0.02, seed: int = 0):
    """HR (4, H, W) @1.25m -> stos T klatek LR (4, H/8, W/8) @10m."""
    g = torch.Generator().manual_seed(seed)
    frames = []
    for _ in range(n_frames):
        dy = float(torch.rand(1, generator=g) * 2 - 1) * max_shift
        dx = float(torch.rand(1, generator=g) * 2 - 1) * max_shift
        sh = ndi_shift(hr4.numpy(), shift=(0.0, dy, dx), order=1, mode="reflect")
        lr = F.avg_pool2d(torch.from_numpy(sh)[None], kernel_size=total_scale)[0]
        lr = (lr + noise * torch.randn(lr.shape, generator=g)).clamp(0, 1)
        frames.append(lr)
    return torch.stack(frames)  # (T,4,128,128)


def sen2sr_each(model, stack, device):
    """SEN2SR x4 na kazdej klatce -> (T,4,512,512)."""
    outs = []
    with torch.no_grad():
        for t in range(stack.shape[0]):
            outs.append(model(stack[t][None].to(device)).squeeze(0).clamp(min=0).cpu())
    return torch.stack(outs)


def evaluate(n_locs, n_frames, noise, device):
    print(f"\n{'='*60}")
    print(f"  MISR x2 — walidacja zejscia < 2.5 m (HR GUGiK 1.25 m)")
    print(f"  lokacji={n_locs}  klatek/stos={n_frames}  device={device}")
    print(f"{'='*60}\n")

    model = load_model(device)
    acc = {k: [] for k in ["psnr_single", "psnr_misrx2", "f1_single", "f1_misrx2"]}
    used = 0
    t0 = time.time()

    for li, (lat, lon) in enumerate(PL_AGRI_COORDS):
        if used >= n_locs:
            break
        raw = fetch_ortho_rgb(lat, lon, size_px=1024, res_m=1.25)  # HR @1.25m
        if raw is None:
            continue
        hr_rgb = harmonize_to_reflectance(raw)                     # (3,1024,1024)
        hr4 = torch.cat([hr_rgb, hr_rgb[0:1]], 0)                  # NIR placeholder

        stack = make_lr_stack_from_hr(hr4, n_frames, noise=noise, seed=300 + li)
        sr_stack = sen2sr_each(model, stack, device)              # (T,4,512,512) @2.5m

        # (1) baseline: pojedyncza klatka SEN2SR -> bicubic x2 -> 1.25m
        single = F.interpolate(sr_stack[0][None], scale_factor=2,
                               mode="bicubic", align_corners=False)[0].clamp(0, 1)
        # (2) MISR x2: fuzja SR-klatek na 2x gestsza siatke
        misrx2 = misr_fuse(sr_stack, scale=2, robust=True)         # (4,1024,1024) @1.25m

        hr_eval = hr4
        ref = edge_map(hr_eval[:3])
        for tag, img in [("single", single), ("misrx2", misrx2)]:
            acc[f"psnr_{tag}"].append(psnr(img[:3], hr_eval[:3]))
            acc[f"f1_{tag}"].append(boundary_f1(edge_map(img[:3]), ref))
        used += 1
        print(f"  [{used}/{n_locs}] ({lat},{lon})  "
              f"PSNR single={acc['psnr_single'][-1]:.2f} misrx2={acc['psnr_misrx2'][-1]:.2f}  "
              f"F1 {acc['f1_single'][-1]:.3f}->{acc['f1_misrx2'][-1]:.3f}")

    def m(x): return float(np.mean(x)) if x else float("nan")
    summary = {
        "n_locs": used, "n_frames": n_frames, "noise": noise,
        "elapsed_s": round(time.time() - t0, 1),
        "psnr": {"single": m(acc["psnr_single"]), "misrx2": m(acc["psnr_misrx2"]),
                 "delta": m(acc["psnr_misrx2"]) - m(acc["psnr_single"])},
        "f1": {"single": m(acc["f1_single"]), "misrx2": m(acc["f1_misrx2"]),
               "delta": m(acc["f1_misrx2"]) - m(acc["f1_single"])},
        "misrx2_wins_pct": 100.0 * float(np.mean(
            [b > a for a, b in zip(acc["psnr_single"], acc["psnr_misrx2"])])) if used else float("nan"),
    }
    return summary


def report(s):
    p, f = s["psnr"], s["f1"]
    print(f"\n{'='*60}")
    print(f"  WYNIK — zejscie < 2.5 m (1.25 m), vs HR GUGiK  (n={s['n_locs']}, {s['elapsed_s']}s)")
    print(f"{'='*60}")
    print(f"  {'metryka':<22}{'SEN2SR+bicubic':>16}{'MISR x2':>10}{'delta':>9}")
    print(f"  {'PSNR @1.25m (dB)':<22}{p['single']:>16.2f}{p['misrx2']:>10.2f}{p['delta']:>+9.2f}")
    print(f"  {'delineacja F1':<22}{f['single']:>16.3f}{f['misrx2']:>10.3f}{f['delta']:>+9.3f}")
    print(f"\n  MISR x2 wygrywa PSNR w {s['misrx2_wins_pct']:.0f}% lokacji")
    ok = p["delta"] > 0 and f["delta"] >= 0
    print(f"  -> {'WIERNE zejscie < 2.5 m' if ok else 'NIEJEDNOZNACZNE'}  "
          f"(PSNR {p['delta']:+.2f} dB, F1 {f['delta']:+.3f})")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.02)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = evaluate(args.n, args.frames, args.noise, device)
    report(s)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "misr_x2_eval.json").write_text(
        json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
