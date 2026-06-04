"""
Faza 2 (krok 2c) — HighRes-net: uczona fuzja MISR (WDROZENIE.md)
================================================================
Kompaktowa implementacja HighRes-net w PyTorch. Sygnatura tej architektury to
REKURSYWNA fuzja par klatek: kodujemy kazda klatke, potem parami redukujemy
T cech do jednej, dekodujemy do czystej klatki.

Zadanie: stos T zaszumionych/przesunietych klatek LR (128px) -> jedna czysta
klatka LR (128px), ktora idzie dalej na wejscie SEN2SR. To uczona alternatywa
dla klasycznej fuzji median z misr.py.

Cel treningu (samonadzor z HR): czysta klatka = downsample HR bez szumu/szift.
Dane: 28 obrazow HR ze spain_crops, z ktorych generujemy stosy w locie.
Trening na CPU (maly model, 128px, L1).

Uruchomienie:
    python highresnet.py --steps 300        # trening + zapis wag
    python highresnet.py --eval-only        # tylko ewaluacja zapisanych wag
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import opensr_test
from measure import to_tensor
from eval_misr import make_lr_stack

WEIGHTS_PATH = Path("models") / "highresnet_misr.pt"


# ─────────────────────────────────────────────
# Architektura
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class HighResNetMISR(nn.Module):
    """
    Encoder per-klatka (z podpowiedzia ref-diff) -> rekursywna fuzja par ->
    dekoder. Wyjscie = reszta nad srednia klatka (stabilny punkt startowy).
    """
    def __init__(self, channels: int = 4, feat: int = 24):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(channels + 1, feat, 3, padding=1), nn.ReLU(inplace=True),
            ResBlock(feat), ResBlock(feat),
        )
        # fuzja pary cech: 2*feat -> feat
        self.fuse = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 3, padding=1), nn.ReLU(inplace=True),
            ResBlock(feat),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(feat, feat, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(feat, channels, 3, padding=1),
        )

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        # stack: (T, C, H, W) lub (B, T, C, H, W)
        if stack.dim() == 4:
            stack = stack[None]
        B, T, C, H, W = stack.shape
        ref = stack.mean(dim=1)                       # (B,C,H,W) referencja = srednia

        feats = []
        for t in range(T):
            frame = stack[:, t]                       # (B,C,H,W)
            cue = (frame - ref).mean(dim=1, keepdim=True)   # podpowiedz o przesunieciu
            feats.append(self.encode(torch.cat([frame, cue], dim=1)))

        # Rekursywna fuzja par
        while len(feats) > 1:
            nxt = []
            for i in range(0, len(feats) - 1, 2):
                nxt.append(self.fuse(torch.cat([feats[i], feats[i + 1]], dim=1)))
            if len(feats) % 2 == 1:
                nxt.append(feats[-1])
            feats = nxt

        out = self.decode(feats[0]) + ref             # reszta nad srednia
        return out.clamp(0.0, 1.0).squeeze(0)


# ─────────────────────────────────────────────
# Trening (samonadzor z HR)
# ─────────────────────────────────────────────

def load_hr(dataset: str = "spain_crops") -> torch.Tensor:
    data = opensr_test.load(dataset)
    return (to_tensor(data["HRharm"]) / 10_000.0).clamp(0, 1)   # (N,4,512,512)


def train(steps: int, n_frames: int, noise: float, feat: int,
          lr: float, seed: int = 0) -> HighResNetMISR:
    torch.manual_seed(seed)
    hr_all = load_hr()
    n = hr_all.shape[0]
    n_val = max(4, n // 6)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = idx[:n_val].tolist(), idx[n_val:].tolist()

    model = HighResNetMISR(channels=4, feat=feat)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def clean_target(hr):           # idealna klatka LR (bez szumu/szift)
        return F.avg_pool2d(hr[None], kernel_size=4)[0]

    print(f"  trening: {len(train_idx)} HR train / {len(val_idx)} val, "
          f"{n_frames} klatek/stos, feat={feat}, steps={steps}")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(1, steps + 1):
        i = int(rng.choice(train_idx))
        hr = hr_all[i]
        stack = make_lr_stack(hr, n_frames, noise=noise, seed=int(rng.integers(1e9)))
        target = clean_target(hr)

        pred = model(stack)
        loss = F.l1_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 50 == 0 or step == 1:
            # szybka walidacja
            model.eval()
            with torch.no_grad():
                vl = []
                for j in val_idx:
                    hrj = hr_all[j]
                    st = make_lr_stack(hrj, n_frames, noise=noise, seed=7000 + j)
                    vl.append(float(F.l1_loss(model(st), clean_target(hrj))))
            model.train()
            print(f"    step {step:4d}/{steps}  train L1={float(loss):.4f}  "
                  f"val L1={np.mean(vl):.4f}  ({time.time()-t0:.0f}s)")

    model.eval()
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feat": feat}, WEIGHTS_PATH)
    print(f"  zapisano wagi: {WEIGHTS_PATH}")
    return model


def load_trained() -> HighResNetMISR:
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    model = HighResNetMISR(channels=4, feat=ckpt.get("feat", 24))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ─────────────────────────────────────────────
# Szybka ewaluacja fuzji: HighRes-net vs median klasyczna
# (na poziomie czystej klatki LR, bez SEN2SR — izoluje jakosc fuzji)
# ─────────────────────────────────────────────

def quick_eval(model: HighResNetMISR, n_frames: int, noise: float, n: int = 6):
    from misr import misr_fuse
    hr_all = load_hr()
    n = min(n, hr_all.shape[0])
    L1_net, L1_med = [], []
    for i in range(n):
        hr = hr_all[i]
        target = F.avg_pool2d(hr[None], kernel_size=4)[0]
        stack = make_lr_stack(hr, n_frames, noise=noise, seed=5000 + i)
        with torch.no_grad():
            pred = model(stack)
        med = misr_fuse(stack, scale=1, robust=True)
        L1_net.append(float(F.l1_loss(pred, target)))
        L1_med.append(float(F.l1_loss(med, target)))
    print(f"\n  Jakosc fuzji (L1 do czystej klatki, nizej lepiej, n={n}):")
    print(f"    median klasyczna : {np.mean(L1_med):.4f}")
    print(f"    HighRes-net      : {np.mean(L1_net):.4f}  "
          f"({(np.mean(L1_med)-np.mean(L1_net)):+.4f} vs median)")


def main():
    ap = argparse.ArgumentParser(description="HighRes-net MISR — trening/ewaluacja")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--feat", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("  HighRes-net MISR (Faza 2, krok 2c)")
    print("=" * 60)
    if args.eval_only:
        model = load_trained()
    else:
        model = train(args.steps, args.frames, args.noise, args.feat, args.lr)
    quick_eval(model, args.frames, args.noise)


if __name__ == "__main__":
    main()
