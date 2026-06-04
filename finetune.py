"""
Faza 3 — Fine-tuning SEN2SR (dowód pętli na CPU) (WDROZENIE.md)
===============================================================
Dostraja istniejacy SEN2SRLite (NIE trenuje od zera) na parach L2A→HR.
SEN2SRLite ma tylko ~572k parametrow — fine-tuning jest w pelni wykonalny na
CPU (ok. 0.23 s/krok). Ten skrypt DOWODZI, ze petla dziala i poprawia metryki
na zbiorze TESTOWYM, zanim wezmiemy sie za realne polskie ortofoto GUGiK.

Co robimy:
  - laduje SEN2SRLite, ODMRAZA sr_model, hard-constraint zostaje ZAMROZONY
    (zachowana gwarancja spektralna — kluczowe dla NDVI),
  - pary: realne L2A[RGBN] (128px) -> HRharm (512px) ze spain_crops,
  - rozlaczny podzial train/val/TEST (uczciwie: mierzymy na nigdy-niewidzianych),
  - fine-tuning niskim LR + augmentacja,
  - bramy z Fazy 1 (PSNR / delineacja F1 / NDVI) PRZED i PO — na zbiorze TEST.

Uwaga: spain_crops to hiszpanskie pola i tylko 28 probek — to dowod MECHANIZMU,
nie finalny model. Polskie dane (GUGiK 25 cm) to kolejny krok Fazy 3.

Uruchomienie:
    python finetune.py                  # ~1500 krokow, CPU
    python finetune.py --steps 3000 --lr 5e-5
"""

import sys
import argparse
import time
from pathlib import Path

# Konsola Windows (cp1252) nie zna czesci znakow — wymus UTF-8 na stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F

import mlstac
import opensr_test

from measure import to_tensor, edge_map, boundary_f1, ndvi, L2A_RGBN_IDX

SEN2SR_MODEL_DIR = Path("models") / "SEN2SRLite_RGBN"
WEIGHTS_OUT = Path("models") / "sen2sr_finetuned_pl.pt"


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10.0 * float(np.log10(1.0 / mse))


def load_trainable(device):
    """SEN2SR z odmrozonym sr_model; hard_constraint zostaje zamrozony."""
    model = mlstac.load(str(SEN2SR_MODEL_DIR)).compiled_model(device=device)
    # odmroz tylko siec SR
    for p in model.sr_model.parameters():
        p.requires_grad_(True)
    for p in model.hard_constraint.parameters():
        p.requires_grad_(False)
    return model


def get_pairs(device):
    """Realne pary ze spain_crops: lr=L2A[RGBN] (128), hr=HRharm (512), [0,1]."""
    data = opensr_test.load("spain_crops")
    l2a = to_tensor(data["L2A"])            # (N,12,128,128)
    hr = to_tensor(data["HRharm"])          # (N,4,512,512)
    lr = (l2a[:, L2A_RGBN_IDX] / 10_000.0).clamp(0, 1)
    hr = (hr / 10_000.0).clamp(0, 1)
    return lr.to(device), hr.to(device)


def augment(lr, hr, g):
    """Wspolna augmentacja (flip/rot90) dla pary lr/hr."""
    if torch.rand(1, generator=g) < 0.5:
        lr, hr = torch.flip(lr, [-1]), torch.flip(hr, [-1])
    if torch.rand(1, generator=g) < 0.5:
        lr, hr = torch.flip(lr, [-2]), torch.flip(hr, [-2])
    k = int(torch.randint(0, 4, (1,), generator=g))
    if k:
        lr, hr = torch.rot90(lr, k, [-2, -1]), torch.rot90(hr, k, [-2, -1])
    return lr, hr


@torch.no_grad()
def eval_gates(model, lr, hr, idx) -> dict:
    """Bramy z Fazy 1 na podanych indeksach (zbior testowy)."""
    model.eval()
    ps, f1s, nb, nc = [], [], [], []
    for i in idx:
        sr = model(lr[i][None]).squeeze(0).clamp(min=0.0).cpu()
        h = hr[i].cpu()
        ps.append(psnr(sr, h))
        ref = edge_map(h)
        f1s.append(boundary_f1(edge_map(sr), ref))
        nv, nvh = ndvi(sr[0], sr[3]), ndvi(h[0], h[3])
        nb.append(float((nv - nvh).mean()))
        nc.append(float(np.corrcoef(nv.flatten().numpy(),
                                    nvh.flatten().numpy())[0, 1]))
    return {"psnr": float(np.mean(ps)), "f1": float(np.mean(f1s)),
            "ndvi_bias": float(np.mean(nb)), "ndvi_corr": float(np.mean(nc))}


def finetune(steps, lr_rate, batch, seed, device):
    lr_all, hr_all = get_pairs(device)
    n = lr_all.shape[0]

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_test, n_val = 5, 4
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    print(f"  podzial: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} TEST")

    model = load_trainable(device)
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parametry uczace (sr_model): {n_train_p:,}")

    # Bramy PRZED
    before = eval_gates(model, lr_all, hr_all, test_idx)
    print(f"  PRZED  PSNR={before['psnr']:.2f}  F1={before['f1']:.3f}  "
          f"NDVIcorr={before['ndvi_corr']:.3f}")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr_rate)
    rng = np.random.default_rng(seed)
    best_val, best_state = -1e9, None
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        bi = rng.choice(train_idx, size=batch, replace=len(train_idx) < batch)
        lrs, hrs = [], []
        for i in bi:
            a, b = augment(lr_all[int(i)], hr_all[int(i)], g)
            lrs.append(a); hrs.append(b)
        xb, yb = torch.stack(lrs), torch.stack(hrs)

        sr = model(xb)
        loss = F.l1_loss(sr, yb)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step == 1:
            vg = eval_gates(model, lr_all, hr_all, val_idx)
            model.train()
            score = vg["psnr"]
            if score > best_val:
                best_val = score
                best_state = {k: v.detach().clone()
                              for k, v in model.sr_model.state_dict().items()}
            print(f"    step {step:4d}/{steps}  L1={float(loss):.4f}  "
                  f"val PSNR={vg['psnr']:.2f}  F1={vg['f1']:.3f}  ({time.time()-t0:.0f}s)")

    # przywroc najlepsze wg walidacji
    if best_state is not None:
        model.sr_model.load_state_dict(best_state)

    after = eval_gates(model, lr_all, hr_all, test_idx)
    print(f"\n  PO     PSNR={after['psnr']:.2f}  F1={after['f1']:.3f}  "
          f"NDVIcorr={after['ndvi_corr']:.3f}")

    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sr_model": model.sr_model.state_dict()}, WEIGHTS_OUT)
    print(f"  zapisano wagi: {WEIGHTS_OUT}")
    return before, after


def report(before, after):
    print(f"\n{'='*60}")
    print(f"  FINE-TUNING SEN2SR — bramy na zbiorze TEST (PRZED -> PO)")
    print(f"{'='*60}")
    rows = [
        ("PSNR vs HR (dB)", "psnr",      "wyzej", 2),
        ("delineacja F1",   "f1",        "wyzej", 3),
        ("NDVI bias",       "ndvi_bias", "~0",    4),
        ("NDVI corr",       "ndvi_corr", "wyzej", 3),
    ]
    print(f"  {'metryka':<18}{'PRZED':>10}{'PO':>10}{'delta':>10}   cel")
    for name, key, goal, dec in rows:
        b, a = before[key], after[key]
        d = a - b
        print(f"  {name:<18}{b:>10.{dec}f}{a:>10.{dec}f}{d:>+10.{dec}f}   {goal}")
    dp = after["psnr"] - before["psnr"]
    df = after["f1"] - before["f1"]
    verdict = "POPRAWA" if (dp > 0 or df > 0) else "brak poprawy"
    print(f"\n  -> Fine-tuning: {verdict}  (PSNR {dp:+.2f} dB, F1 {df:+.3f})")
    print(f"  Petla dziala na CPU. Polskie ortofoto GUGiK = kolejny krok Fazy 3.")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(description="Faza 3 — fine-tuning SEN2SR (CPU)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"  Fine-tuning SEN2SR  (device={device}, steps={args.steps}, lr={args.lr})")
    print("=" * 60)
    before, after = finetune(args.steps, args.lr, args.batch, args.seed, device)
    report(before, after)


if __name__ == "__main__":
    main()
