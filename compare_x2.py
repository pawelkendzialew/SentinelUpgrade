"""
Uczciwe porownanie SEN2SR vs MISR x2 — W TEJ SAMEJ ROZDZIELCZOSCI
==================================================================
SEN2SR to 1024 (2.5m), MISR x2 to 2048 (1.25m) — roznych rozmiarow, nie da sie
porownac ostrosci. Rozwiazanie: powiekszamy SEN2SR do 2048 (bicubic) i stawiamy
OBOK MISR x2. Jedyna roznica = ostrosc (jesli jakas jest).

Tworzy output/porownanie_x2.png:
  - gora: cale obrazy w tym samym rozmiarze [SEN2SR->2048 | MISR x2 2048]
  - dol:  zoom na fragment z detalem [SEN2SR->2048 | MISR x2 | roznica x8]

Uzycie: automatycznie z pipeline (gdy MISR x2), albo recznie: python compare_x2.py
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

OUT = Path("output")


def find_detailed_crop(img: Image.Image, size: int) -> tuple:
    """Fragment o najwiekszej gestosci krawedzi (tam najlepiej widac roznice)."""
    g = np.asarray(img.convert("L"), dtype="float32")
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    grad = gy + gx
    W, H = img.size
    best, bx, by = -1, 0, 0
    step = max(32, size // 2)
    for y in range(0, max(1, H - size), step):
        for x in range(0, max(1, W - size), step):
            s = grad[y:y+size, x:x+size].sum()
            if s > best:
                best, bx, by = s, x, y
    return (bx, by, bx + size, by + size)


def build_comparison(sr_img: Image.Image, x2_img: Image.Image, out_path: Path) -> float:
    """
    Buduje obraz porownawczy SEN2SR(powiekszony) vs MISR x2 w tym samym rozmiarze.
    Zwraca srednia roznice (0..255) — ~0 znaczy 'taki sam, nie ostrzejszy'.
    """
    sr_img = sr_img.convert("RGB")
    x2_img = x2_img.convert("RGB")
    naive = sr_img.resize(x2_img.size, Image.BICUBIC)   # SEN2SR do rozmiaru MISR x2

    a = np.asarray(naive, dtype="float32")
    b = np.asarray(x2_img, dtype="float32")
    diff = np.abs(a - b)
    mean_diff = float(diff.mean())

    cs = min(256, x2_img.size[0] // 4)
    box = find_detailed_crop(x2_img, cs)
    zoom = 3
    crop_naive = naive.crop(box).resize((cs*zoom, cs*zoom), Image.NEAREST)
    crop_x2 = x2_img.crop(box).resize((cs*zoom, cs*zoom), Image.NEAREST)
    dcrop = diff[box[1]:box[3], box[0]:box[2]]
    crop_diff = Image.fromarray(np.clip(dcrop*8, 0, 255).astype("uint8")).resize(
        (cs*zoom, cs*zoom), Image.NEAREST)

    disp = 760
    top_sr = naive.resize((disp, disp), Image.LANCZOS)
    top_x2 = x2_img.resize((disp, disp), Image.LANCZOS)

    gap, lab_h = 12, 26
    zw = cs * zoom
    W = max(disp * 2 + gap, zw * 3 + gap * 2)
    H = lab_h + disp + 22 + lab_h + zw
    panel = Image.new("RGB", (W, H), (13, 17, 23))
    d = ImageDraw.Draw(panel)

    d.text((4, 6), "CALY OBRAZ (oba 2x) — SEN2SR powiekszony", fill=(230, 237, 243))
    d.text((disp + gap + 4, 6), "CALY OBRAZ — MISR x2", fill=(63, 185, 80))
    panel.paste(top_sr, (0, lab_h))
    panel.paste(top_x2, (disp + gap, lab_h))

    y2 = lab_h + disp + 22
    d.text((4, y2 - 20), f"ZOOM na detal  |  srednia roznica: {mean_diff:.1f}/255 "
           f"({100*mean_diff/255:.1f}%)  ~0 = tak samo, nie ostrzejszy",
           fill=(230, 237, 243))
    for i, (im, lab, col) in enumerate([
        (crop_naive, "SEN2SR powiekszony", (139, 148, 158)),
        (crop_x2, "MISR x2", (63, 185, 80)),
        (crop_diff, "roznica x8", (210, 153, 34)),
    ]):
        px = i * (zw + gap)
        panel.paste(im, (px, y2))
        d.text((px + 4, y2 + 2), lab, fill=col)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)
    return mean_diff


def main():
    p_sr, p_x2 = OUT / "2_sen2sr_2.5m.png", OUT / "3_misr_x2_1.25m.png"
    if not p_sr.exists() or not p_x2.exists():
        print("Brak plikow. Uruchom pipeline z MISR x2 (zaznacz 'Zejdz < 2.5 m').")
        return
    md = build_comparison(Image.open(p_sr), Image.open(p_x2), OUT / "porownanie_x2.png")
    print(f"Srednia roznica MISR x2 vs powiekszony SEN2SR: {md:.2f}/255 "
          f"({100*md/255:.1f}%)")
    print(f"Zapisano: {(OUT/'porownanie_x2.png').resolve()}")


if __name__ == "__main__":
    main()
