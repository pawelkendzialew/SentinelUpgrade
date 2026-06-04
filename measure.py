"""
Faza 1 — Pomiar baseline SEN2SR (WDROZENIE.md)
================================================
Dwie bramy jakosci:

  Brama A — wiernosc (opensr-test):
      reflectance / spectral / spatial / improvement / omission / hallucination
      na datasecie 'spain_crops' (hiszpanskie pola — najblizej naszego case'u).
      + dodatkowy, wlasny tani check spektralny: zgodnosc NDVI.
          NDVI = (B08 - B04) / (B08 + B04)
          Czy NDVI po SR nie dryfuje wzgledem wejscia 10 m?
          Dla monitoringu upraw dryf NDVI = dyskwalifikacja metody.

  Brama B — zadanie docelowe (delineacja granic pol), label-free:
      HR (512px) = referencja. Boundary F1 krawedzi: natywne 10 m (LR bilinear
      ->512) vs SEN2SR 2.5 m, oba liczone wzgledem krawedzi HR. Jesli SR ma
      wyzsze F1 -> realnie pomaga w wyznaczaniu granic pol (nie tylko "ladniej").
      Brak gotowego toolkitu w opensr-test -> wlasny (skimage Sobel + scipy EDT).

Uruchomienie:
    python measure.py            # domyslnie 8 probek
    python measure.py --n 28     # caly dataset (wolniej na CPU)
    python measure.py --n 4 --dataset spain_urban

Wynik: tabela w konsoli + output/faza1_metrics.json
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import mlstac
import opensr_test

from skimage.filters import sobel
from scipy.ndimage import distance_transform_edt

# Indeksy pasm [B04, B03, B02, B08] = [R, G, B, NIR] w 12-pasmowym L2A opensr-test.
# Wyznaczone empirycznie (korelacja z HRharm, r>=0.93) i zgodne z kanoniczna
# kolejnoscia Sentinel-2: B01,B02,B03,B04,...,B08 -> indeksy 1,2,3,7.
L2A_RGBN_IDX = [3, 2, 1, 7]

SEN2SR_MODEL_DIR = Path("models") / "SEN2SRLite_RGBN"
OUTPUT_DIR = Path("output")


def to_tensor(arr) -> torch.Tensor:
    return torch.as_tensor(np.asarray(arr)).float()


def ndvi(red: torch.Tensor, nir: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (nir - red) / (nir + red + eps)


# ─────────────────────────────────────────────
# Brama B — delineacja granic pol (label-free)
# ─────────────────────────────────────────────
# Pomysl: HR (512px) to "prawda" wysokiej rozdzielczosci. Sprawdzamy, czyje
# granice lepiej zgadzaja sie z HR — natywne 10 m (LR podbite bilinear do 512)
# czy SEN2SR 2.5 m. Jesli SR lepiej -> realnie pomaga w delineacji pol.

def edge_map(img_chw: torch.Tensor, q: float = 0.80) -> np.ndarray:
    """Mapa krawedzi z gradientu Sobela na luminancji RGB.
    Prog = kwantyl q (stala gestosc krawedzi -> uczciwe porownanie LR vs SR)."""
    gray = img_chw[:3].mean(0).cpu().numpy().astype(np.float32)
    g = sobel(gray)
    thr = np.quantile(g, q)
    return g >= thr


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> float:
    """Boundary F1: zgodnosc krawedzi pred z gt z tolerancja tol pikseli."""
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    dt_gt = distance_transform_edt(~gt)       # odleglosc do najblizszej krawedzi gt
    dt_pred = distance_transform_edt(~pred)
    precision = float((dt_gt[pred] <= tol).mean())
    recall = float((dt_pred[gt] <= tol).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_model(device: torch.device):
    if not (SEN2SR_MODEL_DIR / "mlm.json").exists():
        raise FileNotFoundError(
            f"Brak modelu w {SEN2SR_MODEL_DIR}. Uruchom najpierw pipeline "
            f"(pobierze wagi SEN2SRLite z HuggingFace)."
        )
    model = mlstac.load(str(SEN2SR_MODEL_DIR)).compiled_model(device=device)
    model.eval()
    return model


def run_baseline(n_samples: int, dataset: str, device: torch.device) -> dict:
    print(f"\n{'='*60}")
    print(f"  Faza 1 — Pomiar baseline SEN2SR")
    print(f"  dataset={dataset}  probek={n_samples}  device={device}")
    print(f"{'='*60}\n")

    model = load_model(device)
    data = opensr_test.load(dataset)
    l2a_all = to_tensor(data["L2A"])        # (N,12,128,128) skala *10000
    hr_all = to_tensor(data["HRharm"])      # (N,4,512,512)  [R,G,B,NIR]
    n_total = l2a_all.shape[0]
    n_samples = min(n_samples, n_total)

    metrics = opensr_test.Metrics()

    # Akumulatory Brama A
    keys_A = ["reflectance", "spectral", "spatial",
              "improvement", "omission", "hallucination"]
    acc_A = {k: [] for k in keys_A}
    # Akumulatory NDVI (dodatek do Bramy A)
    ndvi_bias, ndvi_mae, ndvi_corr = [], [], []
    # Akumulatory Brama B (delineacja granic pol: boundary F1 vs HR)
    f1_lr, f1_sr = [], []

    t0 = time.time()
    for i in range(n_samples):
        # Wejscie LR 4-kanalowe [R,G,B,NIR], reflektancja [0,1]
        lr = (l2a_all[i, L2A_RGBN_IDX] / 10_000.0).clamp(0, 1)   # (4,128,128)
        hr = (hr_all[i] / 10_000.0).clamp(0, 1)                  # (4,512,512)

        with torch.no_grad():
            sr = model(lr[None].to(device)).squeeze(0).clamp(min=0.0).cpu()  # (4,512,512)

        # ── Brama A ──
        try:
            metrics.compute(lr=lr, sr=sr, hr=hr)
            r = metrics.results
            vals = {
                "reflectance":   float(r.consistency.reflectance.nanmean()),
                "spectral":      float(r.consistency.spectral.nanmean()),
                "spatial":       float(np.nanmean(np.asarray(r.consistency.spatial))),
                "improvement":   float(np.nanmean(np.asarray(r.correctness.improvement))),
                "omission":      float(np.nanmean(np.asarray(r.correctness.omission))),
                "hallucination": float(np.nanmean(np.asarray(r.correctness.hallucination))),
            }
            for k in keys_A:
                acc_A[k].append(vals[k])
        except Exception as ex:
            print(f"  [probka {i}] Brama A blad: {ex}")
            vals = {k: float("nan") for k in keys_A}

        # ── NDVI (dodatek do Bramy A) ──
        # NDVI wejscia (128) vs NDVI wyjscia zmniejszone do 128 (srednia 4x4)
        ndvi_lr = ndvi(lr[0], lr[3])                              # (128,128)
        ndvi_sr_full = ndvi(sr[0], sr[3])                         # (512,512)
        ndvi_sr_ds = F.avg_pool2d(ndvi_sr_full[None, None], kernel_size=4)[0, 0]
        diff = (ndvi_sr_ds - ndvi_lr)
        ndvi_bias.append(float(diff.mean()))
        ndvi_mae.append(float(diff.abs().mean()))
        a = ndvi_lr.flatten().numpy()
        b = ndvi_sr_ds.flatten().numpy()
        ndvi_corr.append(float(np.corrcoef(a, b)[0, 1]))

        # ── Brama B: delineacja granic pol ──
        # Krawedzie HR = referencja. Porownujemy LR (bilinear->512) vs SR.
        lr_up = F.interpolate(lr[None], size=hr.shape[-2:],
                              mode="bilinear", align_corners=False)[0]
        ref_edge = edge_map(hr)
        f1_lr.append(boundary_f1(edge_map(lr_up), ref_edge))
        f1_sr.append(boundary_f1(edge_map(sr), ref_edge))

        print(f"  probka {i+1}/{n_samples}  "
              f"impr={vals['improvement']:.3f}  hall={vals['hallucination']:.3f}  "
              f"NDVI_bias={ndvi_bias[-1]:+.4f}  "
              f"F1: LR={f1_lr[-1]:.3f} SR={f1_sr[-1]:.3f}")

    elapsed = time.time() - t0

    def msummary(xs):
        xs = [x for x in xs if not np.isnan(x)]
        return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), 0.0)

    summary = {
        "dataset": dataset,
        "n_samples": n_samples,
        "elapsed_s": round(elapsed, 1),
        "brama_A": {k: {"mean": msummary(acc_A[k])[0], "std": msummary(acc_A[k])[1]}
                    for k in keys_A},
        "ndvi_spectral": {
            "bias_mean": msummary(ndvi_bias)[0],
            "mae_mean":  msummary(ndvi_mae)[0],
            "corr_mean": msummary(ndvi_corr)[0],
        },
        "brama_B_delineacja": {
            "f1_lr_mean": msummary(f1_lr)[0],     # natywne 10 m (bilinear)
            "f1_sr_mean": msummary(f1_sr)[0],     # SEN2SR 2.5 m
            "delta_mean": msummary([s - l for s, l in zip(f1_sr, f1_lr)])[0],
            "sr_wins_pct": 100.0 * float(np.mean([s > l for s, l in zip(f1_sr, f1_lr)]))
                           if f1_sr else float("nan"),
        },
        "band_idx_used": L2A_RGBN_IDX,
    }
    return summary


def print_report(s: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  WYNIKI — baseline SEN2SR  ({s['dataset']}, n={s['n_samples']}, {s['elapsed_s']}s)")
    print(f"{'='*60}")
    print(f"\n  BRAMA A — wiernosc (opensr-test):")
    print(f"  {'metryka':<16}{'srednia':>12}{'std':>10}    kierunek")
    arrows = {
        "reflectance":   "nizej lepiej (spojnosc LR)",
        "spectral":      "nizej lepiej (spojnosc spektralna)",
        "spatial":       "nizej lepiej (rejestracja)",
        "improvement":   "WYZEJ lepiej (realna poprawa)",
        "omission":      "nizej lepiej (pominiecia)",
        "hallucination": "nizej lepiej (halucynacje)",
    }
    for k in ["reflectance", "spectral", "spatial",
              "improvement", "omission", "hallucination"]:
        v = s["brama_A"][k]
        print(f"  {k:<16}{v['mean']:>12.4f}{v['std']:>10.4f}    {arrows[k]}")

    b = s["ndvi_spectral"]
    print(f"\n  NDVI — zgodnosc spektralna (dodatek do Bramy A):")
    print(f"  {'bias (sr-lr)':<16}{b['bias_mean']:>12.4f}    ~0 = brak dryfu (cel)")
    print(f"  {'MAE':<16}{b['mae_mean']:>12.4f}    nizej lepiej")
    print(f"  {'korelacja':<16}{b['corr_mean']:>12.4f}    ~1 = NDVI zachowane")

    bb = s["brama_B_delineacja"]
    print(f"\n  BRAMA B — delineacja granic pol (boundary F1 vs HR):")
    print(f"  {'F1 natywne 10m':<16}{bb['f1_lr_mean']:>12.4f}    LR bilinear -> 512")
    print(f"  {'F1 SEN2SR 2.5m':<16}{bb['f1_sr_mean']:>12.4f}    nasz pipeline")
    print(f"  {'delta (SR-LR)':<16}{bb['delta_mean']:>+12.4f}    >0 = SR lepiej delineuje")
    print(f"  {'SR wygrywa':<16}{bb['sr_wins_pct']:>11.1f}%    odsetek probek")

    verdict = "SR POMAGA" if bb["delta_mean"] > 0 else "SR NIE pomaga"
    print(f"\n  -> Brama B: {verdict} w delineacji pol "
          f"(srednia przewaga F1 = {bb['delta_mean']:+.4f})")
    print(f"\n{'='*60}")
    print(f"  To jest PUNKT KONTROLNY. Kazda kolejna metoda (MISR, fine-tune)")
    print(f"  musi pobic te liczby — inaczej jej nie wdrazamy.")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(description="Faza 1 — pomiar baseline SEN2SR")
    ap.add_argument("--n", type=int, default=8, help="liczba probek (domyslnie 8)")
    ap.add_argument("--dataset", default="spain_crops",
                    choices=["spain_crops", "spain_urban", "naip", "spot", "venus"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = run_baseline(args.n, args.dataset, device)
    print_report(summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "faza1_metrics.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Zapisano: {out.resolve()}\n")


if __name__ == "__main__":
    main()
