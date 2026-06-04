"""
Faza 2 — Ewaluacja MISR w bramach Fazy 1 (WDROZENIE.md)
========================================================
Problem: bramy z Fazy 1 (spain_crops) sa JEDNOKLATKOWE, a MISR potrzebuje
wielu przelotow tej samej sceny. Pelny benchmark wieloczasowy (WorldStrat ~setki
GB, PROBA-V) jest poza zasiegiem teraz.

Rozwiazanie (odtwarzalne, uczciwe): bierzemy REALNY target HR ze spain_crops
i generujemy z niego realistyczny STOS czasowy LR:
    HR (2.5 m, 512px)  --subpiksel--> --downsample x4--> --szum--> --chmury-->
    stos T klatek LR (10 m, 128px)
To ta sama filozofia degradacji co przepis SEN2NAIP z Fazy 3 dokumentu.

Porownujemy dwie sciezki, obie zakonczone SEN2SR x4, mierzone wzgledem HR:
    (1) BASELINE: pojedyncza klatka LR        -> SEN2SR -> SR 512
    (2) MISR:     fuzja stosu LR (koregistr.) -> SEN2SR -> SR 512

Bramy (jak w Fazie 1):
    - PSNR(SR, HR)                  ↑ lepiej
    - boundary F1 delineacji pol    ↑ lepiej (vs HR)
    - zgodnosc NDVI (bias/corr)     ~0 / ~1

Uruchomienie:
    python eval_misr.py            # 6 probek, 12 klatek
    python eval_misr.py --n 12 --frames 16 --noise 0.03
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import shift as ndi_shift

import opensr_test

# Reuse z Fazy 1 / 2
from measure import to_tensor, load_model, edge_map, boundary_f1, ndvi
from misr import misr_fuse

OUTPUT_DIR = Path("output")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10.0 * float(np.log10(1.0 / mse))


def make_lr_stack(hr: torch.Tensor, n_frames: int, scale: int = 4,
                  max_shift: float = 3.0, noise: float = 0.02,
                  cloud_prob: float = 0.15, seed: int = 0) -> torch.Tensor:
    """
    Z HR (4,512,512) generuje stos T klatek LR (4,128,128):
    losowe subpikselowe przesuniecie (zrodlo informacji dla MISR) -> downsample
    -> szum gaussowski -> sporadyczna chmura (jasna plama na czesci klatek).
    """
    g = torch.Generator().manual_seed(seed)
    C, H, W = hr.shape
    frames = []
    for t in range(n_frames):
        dy = float(torch.rand(1, generator=g) * 2 - 1) * max_shift
        dx = float(torch.rand(1, generator=g) * 2 - 1) * max_shift
        sh = ndi_shift(hr.numpy(), shift=(0.0, dy, dx), order=1, mode="reflect")
        lr = F.avg_pool2d(torch.from_numpy(sh)[None], kernel_size=scale)[0]
        lr = lr + noise * torch.randn(lr.shape, generator=g)
        # sporadyczna chmura: jasna gaussowska plama
        if float(torch.rand(1, generator=g)) < cloud_prob:
            h, w = lr.shape[-2:]
            cy = int(torch.randint(0, h, (1,), generator=g))
            cx = int(torch.randint(0, w, (1,), generator=g))
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            r2 = ((yy - cy) ** 2 + (xx - cx) ** 2).float()
            blob = torch.exp(-r2 / (2 * (h / 6) ** 2))
            lr = lr + 0.6 * blob[None]
        frames.append(lr.clamp(0, 1))
    return torch.stack(frames)  # (T,4,128,128)


def run_sen2sr_single(model, lr: torch.Tensor, device) -> torch.Tensor:
    with torch.no_grad():
        return model(lr[None].to(device)).squeeze(0).clamp(min=0.0).cpu()


def evaluate(n_samples: int, n_frames: int, noise: float,
             dataset: str, device: torch.device, use_net: bool = False) -> dict:
    print(f"\n{'='*62}")
    print(f"  Faza 2 — Ewaluacja MISR w bramach Fazy 1")
    print(f"  dataset={dataset}  probek={n_samples}  klatek/stos={n_frames}")
    print(f"  szum={noise}  device={device}")
    print(f"{'='*62}\n")

    model = load_model(device)
    data = opensr_test.load(dataset)
    hr_all = to_tensor(data["HRharm"])  # (N,4,512,512)
    n_samples = min(n_samples, hr_all.shape[0])

    net = None
    if use_net:
        from highresnet import load_trained
        net = load_trained()
        print("  + HighRes-net (uczona fuzja) zaladowany do porownania\n")

    keys = ["psnr_single", "psnr_misr",
            "f1_single", "f1_misr",
            "ndvi_bias_single", "ndvi_bias_misr",
            "ndvi_corr_single", "ndvi_corr_misr"]
    if use_net:
        keys += ["psnr_net", "f1_net", "ndvi_bias_net", "ndvi_corr_net"]
    acc = {k: [] for k in keys}

    t0 = time.time()
    for i in range(n_samples):
        hr = (hr_all[i] / 10_000.0).clamp(0, 1)            # (4,512,512) target
        stack = make_lr_stack(hr, n_frames, noise=noise, seed=100 + i)

        # (1) BASELINE: pojedyncza (pierwsza) klatka -> SEN2SR
        single_lr = stack[0]
        sr_single = run_sen2sr_single(model, single_lr, device)

        # (2) MISR: fuzja na natywnej siatce (scale=1, koregistracja+median) -> SEN2SR
        fused_lr = misr_fuse(stack, scale=1, robust=True)   # (4,128,128)
        sr_misr = run_sen2sr_single(model, fused_lr, device)

        # (3) MISR uczony (HighRes-net) -> SEN2SR  [opcjonalnie]
        methods = [("single", sr_single), ("misr", sr_misr)]
        if net is not None:
            with torch.no_grad():
                fused_net = net(stack).clamp(0, 1)
            sr_net = run_sen2sr_single(model, fused_net, device)
            methods.append(("net", sr_net))

        # ── Bramy: PSNR / delineacja F1 / NDVI vs HR ──
        ref_edge = edge_map(hr)
        ndvi_hr = ndvi(hr[0], hr[3])
        for tag, sr in methods:
            acc[f"psnr_{tag}"].append(psnr(sr, hr))
            acc[f"f1_{tag}"].append(boundary_f1(edge_map(sr), ref_edge))
            nv = ndvi(sr[0], sr[3])
            acc[f"ndvi_bias_{tag}"].append(float((nv - ndvi_hr).mean()))
            acc[f"ndvi_corr_{tag}"].append(
                float(np.corrcoef(nv.flatten().numpy(),
                                  ndvi_hr.flatten().numpy())[0, 1]))

        extra = f" net={acc['psnr_net'][-1]:5.2f}" if net is not None else ""
        print(f"  probka {i+1}/{n_samples}  "
              f"PSNR: single={acc['psnr_single'][-1]:5.2f} "
              f"misr={acc['psnr_misr'][-1]:5.2f}{extra}  "
              f"F1: single={acc['f1_single'][-1]:.3f} misr={acc['f1_misr'][-1]:.3f}")

    elapsed = time.time() - t0

    def m(xs):
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "dataset": dataset, "n_samples": n_samples,
        "n_frames": n_frames, "noise": noise,
        "elapsed_s": round(elapsed, 1),
        "psnr":  {"single": m(acc["psnr_single"]), "misr": m(acc["psnr_misr"]),
                  "delta": m(acc["psnr_misr"]) - m(acc["psnr_single"])},
        "delineacja_f1": {"single": m(acc["f1_single"]), "misr": m(acc["f1_misr"]),
                          "delta": m(acc["f1_misr"]) - m(acc["f1_single"])},
        "ndvi": {"bias_single": m(acc["ndvi_bias_single"]),
                 "bias_misr": m(acc["ndvi_bias_misr"]),
                 "corr_single": m(acc["ndvi_corr_single"]),
                 "corr_misr": m(acc["ndvi_corr_misr"])},
        "misr_wins_psnr_pct": 100.0 * float(np.mean(
            [b > a for a, b in zip(acc["psnr_single"], acc["psnr_misr"])])),
    }
    if use_net:
        summary["highresnet"] = {
            "psnr": m(acc["psnr_net"]), "f1": m(acc["f1_net"]),
            "ndvi_bias": m(acc["ndvi_bias_net"]), "ndvi_corr": m(acc["ndvi_corr_net"]),
            "psnr_vs_median": m(acc["psnr_net"]) - m(acc["psnr_misr"]),
        }
    return summary


def print_report(s: dict) -> None:
    print(f"\n{'='*62}")
    print(f"  WYNIK — MISR vs pojedyncza klatka  "
          f"({s['dataset']}, n={s['n_samples']}, {s['n_frames']} klatek, {s['elapsed_s']}s)")
    print(f"{'='*62}")
    p, fdel, nv = s["psnr"], s["delineacja_f1"], s["ndvi"]
    print(f"\n  {'metryka':<22}{'single':>10}{'MISR':>10}{'delta':>10}")
    print(f"  {'PSNR vs HR (dB)':<22}{p['single']:>10.2f}{p['misr']:>10.2f}{p['delta']:>+10.2f}")
    print(f"  {'delineacja F1':<22}{fdel['single']:>10.3f}{fdel['misr']:>10.3f}{fdel['delta']:>+10.3f}")
    print(f"  {'NDVI bias vs HR':<22}{nv['bias_single']:>10.4f}{nv['bias_misr']:>10.4f}")
    print(f"  {'NDVI corr vs HR':<22}{nv['corr_single']:>10.4f}{nv['corr_misr']:>10.4f}")
    if "highresnet" in s:
        hn = s["highresnet"]
        print(f"  {'HighRes-net (uczony)':<22}{'':>10}{hn['psnr']:>10.2f}"
              f"{hn['psnr_vs_median']:>+10.2f}  (vs median)")
        print(f"  {'  F1 / NDVIcorr':<22}{'':>10}{hn['f1']:>10.3f}  corr={hn['ndvi_corr']:.3f}")

    print(f"\n  MISR (median) wygrywa PSNR w {s['misr_wins_psnr_pct']:.0f}% probek")
    verdict = "MISR POMAGA" if p["delta"] > 0 and fdel["delta"] >= 0 else "MISR niejednoznaczny"
    print(f"  -> {verdict}  (PSNR {p['delta']:+.2f} dB, F1 {fdel['delta']:+.3f})")
    if "highresnet" in s:
        win = "bije" if s["highresnet"]["psnr_vs_median"] > 0 else "NIE bije"
        print(f"  -> HighRes-net {win} klasycznej median "
              f"({s['highresnet']['psnr_vs_median']:+.2f} dB)")
    print(f"{'='*62}\n")


def main():
    ap = argparse.ArgumentParser(description="Faza 2 — ewaluacja MISR w bramach")
    ap.add_argument("--n", type=int, default=6, help="liczba probek (domyslnie 6)")
    ap.add_argument("--frames", type=int, default=12, help="klatek na stos (domyslnie 12)")
    ap.add_argument("--noise", type=float, default=0.03, help="szum LR (domyslnie 0.03)")
    ap.add_argument("--dataset", default="spain_crops",
                    choices=["spain_crops", "spain_urban"])
    ap.add_argument("--with-net", action="store_true",
                    help="dolacz HighRes-net do porownania (wymaga wytrenowanych wag)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = evaluate(args.n, args.frames, args.noise, args.dataset, device,
                       use_net=args.with_net)
    print_report(summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "faza2_misr_eval.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Zapisano: {out.resolve()}\n")


if __name__ == "__main__":
    main()
