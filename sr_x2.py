"""
Zejscie ponizej 2.5 m przez UCZENIE modelu x2 (2.5 m -> 1.25 m)
================================================================
Pomysl (zgodny z intuicja: "SEN2SR jakos to robi, douczmy go nizej"):
SEN2SR nauczyl sie 10m->2.5m z par. My uczymy DRUGI model 2.5m->1.25m z
realnych polskich par GUGiK (25 cm -> prawda przy 1.25 m). To ta sama zasada
co SEN2SR, tylko etap nizej. Architektura: CNNSR (ta sama rodzina co SEN2SR).

KLUCZOWE — wejscie modelu = PRAWDZIWY output SEN2SR (nie czyste GUGiK-2.5m),
zeby model uczyl sie poprawiac to, co realnie dostanie w pipeline:
    GUGiK 1.25m (prawda)
      -> avg_pool x8 -> 10 m  --SEN2SR x4-->  2.5 m (wejscie x2)
                                              --model x2-->  1.25 m
      cel = GUGiK 1.25 m

UCZCIWOSC: ponizej 2.5 m czesc detalu jest wnioskowana. Test halucynacji:
porownujemy do PRAWDY GUGiK 1.25 m. Jesli model bije bicubic w PSNR/F1 *wzgledem
prawdy* -> realny detal. Jesli jest ostrzejszy ale PSNR vs prawda SPADA -> halucynuje.

Uruchom:  python sr_x2.py --steps 1500
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
import torch.nn as nn
import torch.nn.functional as F

from sen2sr.models.opensr_baseline.cnn import CNNSR
from measure import load_model, edge_map, boundary_f1
from finetune import psnr, augment
from gugik import fetch_ortho_rgb, harmonize_to_reflectance, PL_AGRI_COORDS

WEIGHTS_OUT = Path("models") / "sr_x2_pl.pt"


def build_pairs(sen2sr, device, max_locs=20):
    """
    Buduje pary (wejscie_2.5m [4,512,512], cel_1.25m [4,1024,1024], grupa).
    Lancuch: GUGiK 1024@1.25m -> avg x8 -> 128@10m -> SEN2SR x4 -> 512@2.5m (wejscie).
    Wejscie = realny output SEN2SR (jak w docelowym pipeline). SEN2SR wymaga 128px.
    """
    print(f"  pobieranie GUGiK @1.25m (1024px) + precompute SEN2SR...")
    ins, tgts, groups = [], [], []
    t0 = time.time()
    n_loc = min(max_locs, len(PL_AGRI_COORDS))
    for gi, (lat, lon) in enumerate(PL_AGRI_COORDS[:n_loc]):
        raw = fetch_ortho_rgb(lat, lon, size_px=1024, res_m=1.25)  # (3,1024,1024) @1.25m
        if raw is None:
            continue
        hr_rgb = harmonize_to_reflectance(raw)
        hr4 = torch.cat([hr_rgb, hr_rgb[0:1]], 0)                  # (4,1024,1024) NIR placeholder
        lr10 = F.avg_pool2d(hr4[None], kernel_size=8)[0]          # (4,128,128) @10m
        with torch.no_grad():
            mid = sen2sr(lr10[None].to(device)).squeeze(0).clamp(min=0).cpu()  # (4,512,512) @2.5m
        ins.append(mid); tgts.append(hr4); groups.append(gi)
        if (gi + 1) % 10 == 0:
            print(f"    {gi+1}/{n_loc} lokacji, {len(ins)} par ({time.time()-t0:.0f}s)")
    print(f"  zebrano {len(ins)} par ({time.time()-t0:.0f}s)")
    return ins, tgts, groups


def split_by_group(groups, seed):
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed); rng.shuffle(uniq)
    n_test = max(2, len(uniq) // 6); n_val = max(1, len(uniq) // 8)
    test_g, val_g = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
    n = len(groups)
    tr = [i for i in range(n) if groups[i] not in test_g | val_g]
    va = [i for i in range(n) if groups[i] in val_g]
    te = [i for i in range(n) if groups[i] in test_g]
    return tr, va, te


def crop_patch(inp, tgt, psize, g):
    """Losowy wycinek psize z wejscia (2.5m) + odpowiadajacy 2*psize z celu (1.25m)."""
    H = inp.shape[1]
    y = int(torch.randint(0, H - psize + 1, (1,), generator=g))
    x = int(torch.randint(0, H - psize + 1, (1,), generator=g))
    pin = inp[:, y:y + psize, x:x + psize]
    pt = tgt[:, 2 * y:2 * (y + psize), 2 * x:2 * (x + psize)]
    return pin, pt


@torch.no_grad()
def eval_gates(model, ins, tgts, idx, device, csize=256):
    """PSNR + F1 vs PRAWDA, model x2 vs bicubic — na srodkowym wycinku (lekko)."""
    model.eval()
    res = {"model_psnr": [], "model_f1": [], "bic_psnr": [], "bic_f1": []}
    for i in idx:
        H = ins[i].shape[1]
        o = (H - csize) // 2
        pin = ins[i][:, o:o + csize, o:o + csize]                      # 2.5m wycinek
        ptg = tgts[i][:, 2 * o:2 * (o + csize), 2 * o:2 * (o + csize)]  # 1.25m cel
        out = model(pin[None].to(device)).squeeze(0).clamp(0, 1).cpu()
        bic = F.interpolate(pin[None], scale_factor=2, mode="bicubic",
                            align_corners=False)[0].clamp(0, 1)
        ref = edge_map(ptg[:3])
        res["model_psnr"].append(psnr(out[:3], ptg[:3]))
        res["model_f1"].append(boundary_f1(edge_map(out[:3]), ref))
        res["bic_psnr"].append(psnr(bic[:3], ptg[:3]))
        res["bic_f1"].append(boundary_f1(edge_map(bic[:3]), ref))
    return {k: float(np.mean(v)) for k, v in res.items()}


def train(steps, lr_rate, batch, seed, device, feat=32, blocks=4, psize=96):
    sen2sr = load_model(device)
    ins, tgts, groups = build_pairs(sen2sr, device)
    if len(ins) < 8:
        raise RuntimeError(f"Za malo par GUGiK ({len(ins)}).")
    tr, va, te = split_by_group(groups, seed)
    print(f"  pary: {len(tr)} train / {len(va)} val / {len(te)} TEST (podzial po regionach)")

    model = CNNSR(in_channels=4, out_channels=4, feature_channels=feat,
                  upscale=2, num_blocks=blocks, train_mode=True).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"  model x2 (CNNSR): {n_p:,} parametrow")

    before = eval_gates(model, ins, tgts, te, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr_rate)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    best_val, best_state = -1e9, None
    t0 = time.time(); model.train()
    for step in range(1, steps + 1):
        bi = rng.choice(tr, size=batch, replace=len(tr) < batch)
        xb, yb = [], []
        for i in bi:
            # lekki trening na wycinkach 128->256 (model konwolucyjny — uczy sie na latkach)
            pin, ptg = crop_patch(ins[int(i)], tgts[int(i)], psize, g)
            a, b = augment(pin, ptg, g)
            xb.append(a); yb.append(b)
        xb, yb = torch.stack(xb).to(device), torch.stack(yb).to(device)
        out = model(xb)
        loss = F.l1_loss(out[:, :3], yb[:, :3])      # RGB (NIR placeholder)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            vg = eval_gates(model, ins, tgts, va, device); model.train()
            if vg["model_psnr"] > best_val:
                best_val = vg["model_psnr"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"    step {step:4d}/{steps}  L1={float(loss):.4f}  "
                  f"val PSNR model={vg['model_psnr']:.2f} bic={vg['bic_psnr']:.2f}  "
                  f"({time.time()-t0:.0f}s)")
    if best_state is not None:
        model.load_state_dict(best_state)
    after = eval_gates(model, ins, tgts, te, device)
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, WEIGHTS_OUT)
    print(f"  zapisano wagi: {WEIGHTS_OUT}")
    return before, after


def report(before, after):
    print(f"\n{'='*64}")
    print(f"  MODEL x2 (2.5->1.25m) — TEST vs PRAWDA GUGiK 1.25 m")
    print(f"{'='*64}")
    print(f"  {'':<20}{'bicubic x2':>14}{'model x2 (PO)':>16}{'delta':>10}")
    print(f"  {'PSNR vs prawda':<20}{after['bic_psnr']:>14.2f}{after['model_psnr']:>16.2f}"
          f"{after['model_psnr']-after['bic_psnr']:>+10.2f}")
    print(f"  {'delineacja F1':<20}{after['bic_f1']:>14.3f}{after['model_f1']:>16.3f}"
          f"{after['model_f1']-after['bic_f1']:>+10.3f}")
    dp = after["model_psnr"] - after["bic_psnr"]
    df = after["model_f1"] - after["bic_f1"]
    if dp > 0 and df >= 0:
        verdict = "WIERNE zejscie < 2.5 m (bije bicubic vs prawda)"
    elif df > 0 and dp <= 0:
        verdict = "OSTRZEJSZE ale PSNR nizszy -> czesciowa HALUCYNACJA"
    else:
        verdict = "brak realnej poprawy nad bicubic"
    print(f"\n  -> {verdict}")
    print(f"     (model PSNR {dp:+.2f} dB, F1 {df:+.3f} vs bicubic, oba vs PRAWDA)")
    print(f"{'='*64}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--psize", type=int, default=96)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 64)
    print(f"  Uczenie modelu x2 (2.5->1.25m) na GUGiK  (device={device}, steps={args.steps})")
    print("=" * 64)
    before, after = train(args.steps, args.lr, args.batch, args.seed, device,
                          feat=args.feat, blocks=args.blocks, psize=args.psize)
    report(before, after)


if __name__ == "__main__":
    main()
