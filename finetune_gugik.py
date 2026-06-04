"""
Faza 3 / B — Fine-tuning SEN2SR na REALNYCH polskich danych GUGiK
=================================================================
Dostraja SEN2SRLite do STRUKTURY polskich pol (male, rozdrobnione dzialki)
na realnym ortofoto GUGiK 25 cm. To zamyka domain-gap, ktorego brakowalo w
Fazie 3 na spain_crops (tam pretrenowany model byl juz dopasowany).

Strategia NIR (uzgodniona — patrz WDROZENIE.md):
    GUGiK publicznie daje tylko RGB. Trenujemy STRUKTURE RGB, a NIR jest
    WYKLUCZONY ze straty (kanal placeholder = Red). NDVI nie jest tu poprawiane,
    ale tez nie psute (chroni zamrozony hard-constraint; przy inferencji NIR
    pochodzi z prawdziwego Sentinela). NIR_TODO: gdy bedzie CIR — wlaczyc do straty.

Uczciwy podzial: train/val/TEST po ROZNYCH lokalizacjach (test = polskie pola
nigdy niewidziane). Bramy strukturalne PRZED/PO: PSNR(RGB) + delineacja F1.

Uruchomienie:
    python finetune_gugik.py                 # pobiera dane + fine-tuning CPU
    python finetune_gugik.py --steps 1000 --lr 5e-5
"""

import sys
import argparse
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F

from measure import edge_map, boundary_f1
from finetune import load_trainable, psnr, augment
from gugik import make_pair, PL_AGRI_COORDS

WEIGHTS_OUT = Path("models") / "sen2sr_finetuned_pl_gugik.pt"


def build_dataset(seed: int = 0, max_locs: int = 28):
    """Pobiera pary GUGiK (RGB) dla polskich lokalizacji rolniczych."""
    print("  pobieranie ortofoto GUGiK (realne polskie HR)...")
    lrs, hrs, used = [], [], []
    t0 = time.time()
    for i, (lat, lon) in enumerate(PL_AGRI_COORDS[:max_locs]):
        pair = make_pair(lat, lon, seed=1000 + i)
        if pair is None:
            continue
        lr, hr = pair                     # lr (3,128,128), hr (3,512,512)
        lrs.append(lr); hrs.append(hr); used.append((lat, lon))
    print(f"  pobrano {len(used)} kafelkow z {min(max_locs,len(PL_AGRI_COORDS))} "
          f"({time.time()-t0:.0f}s)")
    return lrs, hrs


def to4(lr_rgb, hr_rgb):
    """Dodaj kanal NIR placeholder (= Red). NIR i tak wykluczony ze straty."""
    lr4 = torch.cat([lr_rgb, lr_rgb[0:1]], dim=0)   # (4,128,128)
    hr4 = torch.cat([hr_rgb, hr_rgb[0:1]], dim=0)   # (4,512,512)
    return lr4, hr4


@torch.no_grad()
def eval_gates(model, lrs, hrs, idx, device) -> dict:
    """Bramy strukturalne (RGB): PSNR + delineacja F1, vs HR."""
    model.eval()
    ps, f1s = [], []
    for i in idx:
        lr4, _ = to4(lrs[i], hrs[i])
        sr = model(lr4[None].to(device)).squeeze(0).clamp(min=0.0).cpu()
        sr_rgb, hr_rgb = sr[:3], hrs[i]
        ps.append(psnr(sr_rgb, hr_rgb))
        ref = edge_map(hr_rgb)            # edge_map uzywa 3 kanalow (RGB)
        f1s.append(boundary_f1(edge_map(sr_rgb), ref))
    return {"psnr": float(np.mean(ps)), "f1": float(np.mean(f1s))}


def finetune(steps, lr_rate, batch, seed, device):
    lrs, hrs = build_dataset(seed)
    n = len(lrs)
    if n < 6:
        raise RuntimeError(f"Za malo kafelkow GUGiK ({n}) — sprawdz polaczenie/WMS.")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_test = max(3, n // 6)
    n_val = max(2, n // 8)
    test_idx, val_idx, train_idx = (perm[:n_test],
                                    perm[n_test:n_test + n_val],
                                    perm[n_test + n_val:])
    print(f"  podzial po lokalizacjach: {len(train_idx)} train / "
          f"{len(val_idx)} val / {len(test_idx)} TEST")

    model = load_trainable(device)
    before = eval_gates(model, lrs, hrs, test_idx, device)
    print(f"  PRZED  PSNR={before['psnr']:.2f}  F1={before['f1']:.3f}")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr_rate)
    rng = np.random.default_rng(seed)
    best_val, best_state = -1e9, None
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        bi = rng.choice(train_idx, size=batch, replace=len(train_idx) < batch)
        xb, yb = [], []
        for i in bi:
            a, b = augment(lrs[int(i)], hrs[int(i)], g)
            lr4, hr4 = to4(a, b)
            xb.append(lr4); yb.append(hr4)
        xb, yb = torch.stack(xb).to(device), torch.stack(yb).to(device)

        sr = model(xb)
        # STRATA TYLKO NA RGB (NIR wykluczony — placeholder)
        loss = F.l1_loss(sr[:, :3], yb[:, :3])
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step == 1:
            vg = eval_gates(model, lrs, hrs, val_idx, device)
            model.train()
            if vg["psnr"] > best_val:
                best_val = vg["psnr"]
                best_state = {k: v.detach().clone()
                              for k, v in model.sr_model.state_dict().items()}
            print(f"    step {step:4d}/{steps}  L1(RGB)={float(loss):.4f}  "
                  f"val PSNR={vg['psnr']:.2f}  F1={vg['f1']:.3f}  ({time.time()-t0:.0f}s)")

    if best_state is not None:
        model.sr_model.load_state_dict(best_state)
    after = eval_gates(model, lrs, hrs, test_idx, device)
    print(f"\n  PO     PSNR={after['psnr']:.2f}  F1={after['f1']:.3f}")

    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sr_model": model.sr_model.state_dict()}, WEIGHTS_OUT)
    print(f"  zapisano wagi: {WEIGHTS_OUT}")
    return before, after


def report(before, after):
    print(f"\n{'='*60}")
    print(f"  FINE-TUNING SEN2SR na GUGiK (polskie pola) — TEST, PRZED -> PO")
    print(f"{'='*60}")
    dp = after["psnr"] - before["psnr"]
    df = after["f1"] - before["f1"]
    print(f"  {'metryka':<20}{'PRZED':>10}{'PO':>10}{'delta':>10}")
    print(f"  {'PSNR RGB vs HR':<20}{before['psnr']:>10.2f}{after['psnr']:>10.2f}{dp:>+10.2f}")
    print(f"  {'delineacja F1':<20}{before['f1']:>10.3f}{after['f1']:>10.3f}{df:>+10.3f}")
    verdict = "POPRAWA" if (dp > 0 or df > 0) else "brak poprawy"
    print(f"\n  -> {verdict}  (PSNR {dp:+.2f} dB, F1 {df:+.3f}) na NIEWIDZIANYCH polskich polach")
    print(f"  NIR wykluczony ze straty (RGB-only) — NIR_TODO: CIR w przyszlosci.")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(description="Faza 3/B — fine-tuning SEN2SR na GUGiK")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"  Fine-tuning SEN2SR na GUGiK  (device={device}, steps={args.steps})")
    print("=" * 60)
    before, after = finetune(args.steps, args.lr, args.batch, args.seed, device)
    report(before, after)


if __name__ == "__main__":
    main()
