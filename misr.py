"""
Faza 2 — MISR: Multi-Image Super-Resolution (WDROZENIE.md)
==========================================================
Laczy wiele przelotow Sentinel-2 tego samego pola w jeden ostrzejszy/czystszy
obraz. Kazdy przelot jest minimalnie przesuniety subpikselowo — z tych przesuniec
da sie odzyskac realna informacje (nie zgadywanie jak generatywne SR).

Pipeline MISR (klasyczny, bez sieci — baseline przed modelem glebokim 2c):
    stos [T,4,H,W]
      -> 2b. filtr jakosci (odsiej chmury/cien)
      -> 2b. koregistracja subpikselowa (phase_cross_correlation)
      -> fuzja robust (median/mean na siatce x scale)
    => obraz [4, H*scale, W*scale]  (czystszy, z odzyskana informacja)

Ten wynik idzie potem na wejscie SEN2SR (zamiast pojedynczej sceny).

Uwaga dot. ewaluacji: bramy z Fazy 1 (spain_crops) sa jednoklatkowe, wiec
pelny test MISR wymaga benchmarku wieloczasowego (PROBA-V / WorldStrat) — to
kolejny krok. Tu walidujemy logike fuzji samotestem na danych syntetycznych.

Uruchomienie samotestu:
    python misr.py            # syntetyczny test: fuzja vs pojedyncza klatka
"""

import numpy as np
import torch
import torch.nn.functional as F

from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift


# ─────────────────────────────────────────────
# 2b. Filtr jakosci klatek (odsiew chmur/cienia)
# ─────────────────────────────────────────────

def frame_quality(stack: torch.Tensor) -> torch.Tensor:
    """
    Prosty wskaznik jakosci kazdej klatki (bez maski SCL — mamy tylko RGBN).
    Chmury = jasne i nisko-kontrastowe; cien = bardzo ciemny.
    Wyzej = lepiej. Zwraca tensor [T].
    """
    lum = stack[:, :3].mean(dim=1)               # (T,H,W) luminancja RGB
    brightness = lum.mean(dim=(1, 2))            # chmury podnosza srednia
    contrast = lum.std(dim=(1, 2))               # chmury splaszczaja kontrast
    # Kara za nadmierna jasnosc (chmura), nagroda za kontrast (struktura)
    score = contrast - 0.5 * torch.relu(brightness - 0.25)
    return score


def select_frames(stack: torch.Tensor, min_keep: int = 3,
                  drop_frac: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Odrzuca najgorsze klatki (najbardziej zachmurzone). Zachowuje co najmniej
    min_keep. Zwraca (przefiltrowany_stos, indeksy_zachowane).
    """
    T = stack.shape[0]
    scores = frame_quality(stack)
    n_keep = max(min_keep, int(round(T * (1.0 - drop_frac))))
    n_keep = min(n_keep, T)
    keep = torch.argsort(scores, descending=True)[:n_keep]
    keep, _ = torch.sort(keep)
    return stack[keep], keep


# ─────────────────────────────────────────────
# 2b. Koregistracja subpikselowa
# ─────────────────────────────────────────────

def estimate_shifts(stack: torch.Tensor, ref_idx: int,
                    upsample: int = 20) -> list[tuple[float, float]]:
    """
    Szacuje subpikselowe przesuniecie (dy, dx) kazdej klatki wzgledem referencji
    metoda korelacji fazowej. Zwraca liste przesuniec w pikselach LR.
    """
    lum = stack[:, :3].mean(dim=1).cpu().numpy()  # (T,H,W)
    ref = lum[ref_idx]
    shifts = []
    for t in range(stack.shape[0]):
        if t == ref_idx:
            shifts.append((0.0, 0.0))
            continue
        sh, _, _ = phase_cross_correlation(ref, lum[t], upsample_factor=upsample)
        shifts.append((float(sh[0]), float(sh[1])))  # (dy, dx) by przesunac t -> ref
    return shifts


def pick_reference(stack: torch.Tensor) -> int:
    """Referencja = najostrzejsza klatka (najwyzszy gradient = najwiecej detalu)."""
    lum = stack[:, :3].mean(dim=1)
    gy = (lum[:, 1:, :] - lum[:, :-1, :]).abs().mean(dim=(1, 2))
    gx = (lum[:, :, 1:] - lum[:, :, :-1]).abs().mean(dim=(1, 2))
    return int(torch.argmax(gy + gx))


# ─────────────────────────────────────────────
# Fuzja MISR (shift-and-fuse na siatce x scale)
# ─────────────────────────────────────────────

def misr_fuse(stack: torch.Tensor, scale: int = 2,
              ref_idx: int | None = None, robust: bool = True) -> torch.Tensor:
    """
    Klasyczna fuzja MISR:
      1) wybierz referencje (najostrzejsza),
      2) policz subpikselowe przesuniecia wzgledem niej,
      3) podbij kazda klatke bicubic x scale i wyrownaj na siatce HR,
      4) polacz robust (median odrzuca chmury/odstajace; inaczej srednia).

    Wejscie:  stack [T, 4, H, W] (reflektancja [0,1])
    Wyjscie:  [4, H*scale, W*scale]
    """
    T, C, H, W = stack.shape
    if ref_idx is None:
        ref_idx = pick_reference(stack)

    shifts = estimate_shifts(stack, ref_idx)
    Hs, Ws = H * scale, W * scale

    aligned = torch.empty((T, C, Hs, Ws), dtype=torch.float32)
    for t in range(T):
        up = F.interpolate(stack[t][None], scale_factor=scale,
                           mode="bicubic", align_corners=False)[0]  # (C,Hs,Ws)
        dy, dx = shifts[t]
        # przesuniecie na siatce HR = (dy,dx) * scale, w kanalach bez przesuniecia
        arr = up.cpu().numpy()
        arr = ndi_shift(arr, shift=(0.0, dy * scale, dx * scale),
                        order=1, mode="reflect")
        aligned[t] = torch.from_numpy(arr)

    if robust:
        fused = aligned.median(dim=0).values
    else:
        fused = aligned.mean(dim=0)
    return fused.clamp(0.0, 1.0)


# ─────────────────────────────────────────────
# Samotest na danych syntetycznych
# ─────────────────────────────────────────────

def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10.0 * np.log10(1.0 / mse)


def _make_synthetic_stack(hr: torch.Tensor, n_frames: int, scale: int,
                          noise: float, seed: int = 0):
    """Z obrazu HR generuje T klatek LR: losowe subpikselowe przesuniecie ->
    downsample -> szum. Zwraca (stack[T,C,h,w], hr)."""
    g = torch.Generator().manual_seed(seed)
    C, Hh, Wh = hr.shape
    frames = []
    for _ in range(n_frames):
        dy = float(torch.rand(1, generator=g) * 2 - 1) * scale  # +-scale px HR
        dx = float(torch.rand(1, generator=g) * 2 - 1) * scale
        sh = ndi_shift(hr.numpy(), shift=(0.0, dy, dx), order=1, mode="reflect")
        sh_t = torch.from_numpy(sh)
        lr = F.avg_pool2d(sh_t[None], kernel_size=scale)[0]      # downsample
        lr = lr + noise * torch.randn(lr.shape, generator=g)
        frames.append(lr.clamp(0, 1))
    return torch.stack(frames), hr


def selftest():
    print("=" * 60)
    print("  MISR — samotest na danych syntetycznych")
    print("=" * 60)
    torch.manual_seed(0)
    scale = 2

    # Syntetyczny "HR": gladkie tlo + ostre krawedzie (jak pola)
    Hh = Wh = 128
    yy, xx = torch.meshgrid(torch.linspace(0, 6, Hh),
                            torch.linspace(0, 6, Wh), indexing="ij")
    base = (torch.sin(yy) * torch.cos(xx) * 0.5 + 0.5)
    edges = ((xx.floor() + yy.floor()) % 2)                      # szachownica pol
    hr1 = (0.6 * base + 0.4 * edges).clamp(0, 1)
    hr = torch.stack([hr1, hr1 * 0.9, hr1 * 0.8, hr1 * 1.1]).clamp(0, 1)  # 4 kanaly

    for n_frames, noise in [(8, 0.05), (16, 0.05), (16, 0.10)]:
        stack, hr_gt = _make_synthetic_stack(hr, n_frames, scale, noise, seed=1)

        # Baseline: pojedyncza klatka bicubic x scale
        single = F.interpolate(stack[0][None], scale_factor=scale,
                               mode="bicubic", align_corners=False)[0].clamp(0, 1)
        # MISR
        fused = misr_fuse(stack, scale=scale, robust=True)

        p_single = _psnr(single, hr_gt)
        p_fused = _psnr(fused, hr_gt)
        print(f"\n  klatek={n_frames:2d}  szum={noise:.2f}")
        print(f"    pojedyncza (bicubic)  PSNR = {p_single:5.2f} dB")
        print(f"    MISR fuzja            PSNR = {p_fused:5.2f} dB   "
              f"({p_fused - p_single:+.2f} dB)")

    print("\n" + "=" * 60)
    print("  Dodatnia delta PSNR = fuzja odzyskuje informacje / tlumi szum.")
    print("=" * 60)


if __name__ == "__main__":
    selftest()
