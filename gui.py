"""
Sentinel-2 Super-Resolution — GUI
===================================
Workflow:
  1. Region (mapa → STAC): zaznacz obszar, pobierz NAJCZYSTSZĄ scenę (4 pasma RGBN
     z NIR) do folderu surowych TIF-ów 10 m.
  2. Przetwórz folder TIFF: SEN2SR ×4 → 2.5 m + NDVI, zachowanie georeferencji.
Wyniki w folderach output/ — otwierasz w QGIS.

Uruchom: python gui.py
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from pathlib import Path


# ─────────────────────────────────────────────
# KOLORY / STYL
# ─────────────────────────────────────────────

THEME = {
    "bg":           "#0d1117",
    "panel":        "#161b22",
    "border":       "#30363d",
    "accent":       "#58a6ff",
    "accent2":      "#3fb950",
    "warn":         "#d29922",
    "text":         "#e6edf3",
    "text_muted":   "#8b949e",
    "btn_bg":       "#21262d",
    "btn_hover":    "#30363d",
    "btn_accent":   "#1f6feb",
    "progress_bg":  "#21262d",
    "progress_fg":  "#58a6ff",
}


class SentinelSRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentinel-2 Super-Resolution")
        self.geometry("560x640")
        self.minsize(520, 560)
        self.configure(bg=THEME["bg"])

        # styl paska postępu (używany w oknach regionu/folderu)
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("custom.Horizontal.TProgressbar",
                        troughcolor=THEME["progress_bg"], background=THEME["progress_fg"],
                        bordercolor=THEME["border"], lightcolor=THEME["progress_fg"],
                        darkcolor=THEME["progress_fg"])

        self._build_ui()

    # ─────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self, bg=THEME["bg"])
        hdr.pack(fill="x", padx=20, pady=(14, 0))
        tk.Label(hdr, text="Sentinel-2  ·  Super-Resolution", bg=THEME["bg"],
                 fg=THEME["text"], font=("Courier New", 15, "bold")).pack(side="left")
        tk.Label(self, text="10m  →  SEN2SR ×4  →  2.5m  +  NDVI   (GeoTIFF → QGIS)",
                 bg=THEME["bg"], fg=THEME["text_muted"],
                 font=("Courier New", 9)).pack(anchor="w", padx=20, pady=(2, 0))
        tk.Frame(self, height=1, bg=THEME["border"]).pack(fill="x", padx=20, pady=10)

        body = tk.Frame(self, bg=THEME["bg"])
        body.pack(fill="both", expand=True, padx=20)
        self._build_controls(body)
        self._build_statusbar()

    def _build_controls(self, ctrl):
        pad = {"padx": 0, "pady": 4}

        # ── Parametry ──
        self._section_label(ctrl, "PARAMETRY")

        tk.Label(ctrl, text="Data od", bg=THEME["bg"], fg=THEME["text_muted"],
                 font=("Courier New", 9)).pack(anchor="w", **pad)
        self.start_var = tk.StringVar(value="2023-06-01")
        self._entry(ctrl, self.start_var).pack(fill="x", **pad)

        tk.Label(ctrl, text="Data do", bg=THEME["bg"], fg=THEME["text_muted"],
                 font=("Courier New", 9)).pack(anchor="w", **pad)
        self.end_var = tk.StringVar(value="2023-09-30")
        self._entry(ctrl, self.end_var).pack(fill="x", **pad)

        tk.Label(ctrl, text="Rozmiar kafelka (px @10m)", bg=THEME["bg"], fg=THEME["text_muted"],
                 font=("Courier New", 9)).pack(anchor="w", **pad)
        self.size_var = tk.StringVar(value="256")
        sizes = tk.OptionMenu(ctrl, self.size_var, "128", "256", "512")
        sizes.config(bg=THEME["btn_bg"], fg=THEME["text"], font=("Courier New", 10),
                     relief="flat", activebackground=THEME["btn_hover"],
                     activeforeground=THEME["text"], highlightthickness=0)
        sizes["menu"].config(bg=THEME["btn_bg"], fg=THEME["text"], font=("Courier New", 10))
        sizes.pack(fill="x", **pad)

        # Dostrojenie do polskich pol (wagi GUGiK)
        self.finetuned_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ctrl, text="Model dostrojony do polskich pol (GUGiK)",
            variable=self.finetuned_var,
            bg=THEME["bg"], fg=THEME["accent2"], selectcolor=THEME["btn_bg"],
            activebackground=THEME["bg"], activeforeground=THEME["accent2"],
            font=("Courier New", 8), anchor="w",
        ).pack(fill="x", pady=(6, 2))

        # ── Akcje ──
        self._section_label(ctrl, "WORKFLOW")
        tk.Label(
            ctrl,
            text=("① Zaznacz obszar na mapie i pobierz najczystszą\n"
                  "   scenę (4 pasma z NIR) do folderu surowych TIF.\n\n"
                  "② Przetwórz ten folder przez SEN2SR → 2.5 m + NDVI.\n\n"
                  "Wyniki w output/ — otwierasz w QGIS."),
            bg=THEME["bg"], fg=THEME["text_muted"], font=("Courier New", 8),
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        tk.Button(
            ctrl, text="🗺  ①  Region (mapa → STAC → folder TIF)",
            bg=THEME["btn_accent"], fg=THEME["text"],
            font=("Courier New", 10, "bold"), relief="flat", cursor="hand2",
            activebackground="#2a7ad4", activeforeground=THEME["text"],
            pady=10, command=self._open_region_window,
        ).pack(fill="x", pady=(2, 6))

        tk.Button(
            ctrl, text="📁  ②  Przetwórz folder TIFF (SEN2SR)",
            bg=THEME["btn_accent"], fg=THEME["text"],
            font=("Courier New", 10, "bold"), relief="flat", cursor="hand2",
            activebackground="#2a7ad4", activeforeground=THEME["text"],
            pady=10, command=self._process_tiff_folder,
        ).pack(fill="x", pady=(0, 6))

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=THEME["panel"], highlightthickness=1,
                       highlightbackground=THEME["border"])
        bar.pack(fill="x", side="bottom")
        tk.Label(bar, text="Gotowy.", bg=THEME["panel"], fg=THEME["text_muted"],
                 font=("Courier New", 9), anchor="w", padx=12, pady=4).pack(side="left")
        self._device_label = tk.Label(bar, text="device: ...", bg=THEME["panel"],
                                      fg=THEME["text_muted"], font=("Courier New", 9),
                                      padx=12, pady=4)
        self._device_label.pack(side="right")
        self.after(200, self._check_device)

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _section_label(self, parent, text):
        f = tk.Frame(parent, bg=THEME["bg"])
        f.pack(fill="x", pady=(12, 2))
        tk.Label(f, text=text, bg=THEME["bg"], fg=THEME["accent"],
                 font=("Courier New", 8, "bold")).pack(side="left")
        tk.Frame(f, height=1, bg=THEME["border"]).pack(side="left", fill="x",
                                                        expand=True, padx=(6, 0), pady=6)

    def _entry(self, parent, var):
        return tk.Entry(parent, textvariable=var, bg=THEME["btn_bg"], fg=THEME["text"],
                        insertbackground=THEME["text"], font=("Courier New", 10),
                        relief="flat", highlightthickness=1,
                        highlightbackground=THEME["border"], highlightcolor=THEME["accent"])

    def _check_device(self):
        import torch
        dev = "cuda ✓" if torch.cuda.is_available() else "cpu (brak GPU)"
        self._device_label.config(text=f"device: {dev}")

    # ─────────────────────────────────────────────
    # ① TRYB REGIONU (mapa → STAC → folder surowych TIF)
    # ─────────────────────────────────────────────

    def _open_region_window(self):
        try:
            import tkintermapview
        except ImportError:
            messagebox.showerror("Brak biblioteki",
                                 "Zainstaluj: pip install tkintermapview")
            return

        win = tk.Toplevel(self)
        win.title("Region — zaznacz obszar i pobierz/przetwórz")
        win.geometry("900x720")
        win.configure(bg=THEME["bg"])

        tk.Label(win, text="Kliknij DWA przeciwległe rogi prostokąta na mapie.",
                 bg=THEME["bg"], fg=THEME["text"], font=("Courier New", 10)).pack(pady=(8, 2))
        tk.Label(win, text="⬇ Pobierz (STAC) = surowe TIF 10 m z najczystszej sceny → potem 'Przetwórz folder TIFF'",
                 bg=THEME["bg"], fg=THEME["text_muted"], font=("Courier New", 8)).pack()

        mapw = tkintermapview.TkinterMapView(win, width=860, height=440, corner_radius=0)
        mapw.pack(padx=10, pady=6)
        mapw.set_position(51.4, 22.0)   # wschodnio-centralna PL
        mapw.set_zoom(8)

        st = {"corners": [], "bbox": None, "poly": None, "marks": [],
              "thread": None, "stop": False}
        est = tk.Label(win, text="Zaznacz obszar (2 rogi)...", bg=THEME["bg"],
                       fg=THEME["accent2"], font=("Courier New", 10, "bold"))
        est.pack(pady=4)

        def clear():
            for m in st["marks"]:
                m.delete()
            st["marks"] = []
            if st["poly"]:
                st["poly"].delete(); st["poly"] = None
            st["corners"] = []; st["bbox"] = None
            est.config(text="Zaznacz obszar (2 rogi)...")

        def draw_rect():
            (la1, lo1), (la2, lo2) = st["corners"]
            latmin, latmax = sorted([la1, la2])
            lonmin, lonmax = sorted([lo1, lo2])
            st["bbox"] = (latmin, latmax, lonmin, lonmax)
            st["poly"] = mapw.set_polygon(
                [(latmax, lonmin), (latmax, lonmax), (latmin, lonmax), (latmin, lonmin)],
                outline_color="#3fb950", border_width=3, fill_color=None)
            try:
                from batch_region import estimate_region
                edge = int(self.size_var.get())
                n, sec, mb = estimate_region(latmin, latmax, lonmin, lonmax, edge)
                est.config(text=f"{n} kafelkow   ~{sec//60} min (CPU)   ~{mb/1024:.1f} GB")
            except Exception as ex:
                est.config(text=f"estymacja blad: {ex}")

        def on_click(coords):
            if len(st["corners"]) >= 2:
                clear()
            st["corners"].append((coords[0], coords[1]))
            st["marks"].append(mapw.set_marker(coords[0], coords[1]))
            if len(st["corners"]) == 2:
                draw_rect()
        mapw.add_left_click_map_command(on_click)

        prog = ttk.Progressbar(win, mode="determinate", maximum=100,
                               style="custom.Horizontal.TProgressbar")
        prog.pack(fill="x", padx=10, pady=(6, 2))
        plbl = tk.Label(win, text="", bg=THEME["bg"], fg=THEME["text_muted"],
                        font=("Courier New", 8))
        plbl.pack()

        bar = tk.Frame(win, bg=THEME["bg"])
        bar.pack(pady=8)

        def do_download_stac():
            if not st["bbox"]:
                messagebox.showinfo("Region", "Najpierw zaznacz obszar (2 rogi).")
                return
            if st["thread"] and st["thread"].is_alive():
                return
            name = simpledialog.askstring(
                "Pobierz region (STAC)",
                "Nazwa folderu na surowe TIF-y 10 m (powstanie output/<nazwa>_raw).\n"
                "Pobiera NAJCZYSTSZĄ scenę, 4 pasma z NIR. Potem użyj 'Przetwórz folder TIFF':",
                initialvalue="region", parent=win)
            if not name:
                return
            out_raw = f"output/{name}_raw"
            latmin, latmax, lonmin, lonmax = st["bbox"]
            bbox = [lonmin, latmin, lonmax, latmax]
            edge = int(self.size_var.get())
            start, end = self.start_var.get(), self.end_var.get()
            st["stop"] = False

            def cb(done, total, msg):
                win.after(0, lambda: (prog.config(value=100 * done / max(total, 1)),
                                      plbl.config(text=f"[{done}/{total}] {msg}")))

            def worker():
                from stac_acquire import download_region_tiles
                try:
                    n, best = download_region_tiles(
                        bbox, start, end, edge, out_raw,
                        progress_cb=cb, should_stop=lambda: st["stop"])
                    win.after(0, lambda: plbl.config(
                        text=f"✓ Pobrano {n} TIF do {out_raw}/ (scena {best['datetime'][:10]}, "
                             f"{best['cloud']:.1f}% chmur). Teraz: 'Przetwórz folder TIFF'."))
                except Exception as ex:
                    import traceback
                    tb = traceback.format_exc()
                    win.after(0, lambda: messagebox.showerror(
                        "Blad pobierania STAC", f"{ex}\n\n{tb[:500]}"))
            st["thread"] = threading.Thread(target=worker, daemon=True)
            st["thread"].start()

        def mkbtn(txt, cmd, fg):
            return tk.Button(bar, text=txt, command=cmd, bg=THEME["btn_bg"], fg=fg,
                             font=("Courier New", 9, "bold"), relief="flat",
                             cursor="hand2", padx=10, pady=6,
                             activebackground=THEME["btn_hover"], activeforeground=fg)
        mkbtn("Reset", clear, THEME["text_muted"]).pack(side="left", padx=4)
        mkbtn("⬇  Pobierz region (STAC)", do_download_stac, THEME["accent"]).pack(side="left", padx=4)
        mkbtn("■  Stop", lambda: st.update(stop=True), THEME["warn"]).pack(side="left", padx=4)

    # ─────────────────────────────────────────────
    # ② PRZETWARZANIE FOLDERU TIFF (SEN2SR)
    # ─────────────────────────────────────────────

    def _process_tiff_folder(self):
        in_dir = filedialog.askdirectory(title="Wybierz folder z TIFF-ami")
        if not in_dir:
            return
        name = simpledialog.askstring(
            "Folder wyjściowy",
            "Nazwa NOWEGO folderu na wyniki (przy projekcie).\n"
            "Za każdym razem inna — by nie mieszać dwóch przetworzeń:",
            initialvalue="wynik_sr")
        if not name:
            return
        out_dir = Path(name)   # przy projekcie (katalog roboczy)
        if out_dir.exists() and any(out_dir.iterdir()):
            if not messagebox.askyesno(
                "Folder istnieje",
                f"'{name}' już istnieje i nie jest pusty.\n"
                "Kontynuować? (gotowe pliki zostaną pominięte)"):
                return

        win = tk.Toplevel(self)
        win.title(f"Przetwarzanie folderu → {name}")
        win.geometry("560x180")
        win.configure(bg=THEME["bg"])
        tk.Label(win, text=f"Wejście: {in_dir}", bg=THEME["bg"], fg=THEME["text_muted"],
                 font=("Courier New", 8), wraplength=540, justify="left").pack(pady=(10, 2), padx=10, anchor="w")
        tk.Label(win, text=f"Wyjście: {out_dir.resolve()}", bg=THEME["bg"], fg=THEME["text_muted"],
                 font=("Courier New", 8), wraplength=540, justify="left").pack(padx=10, anchor="w")
        ft_note = "model: dostrojony PL" if self.finetuned_var.get() else "model: bazowy SEN2SR"
        tk.Label(win, text=ft_note, bg=THEME["bg"], fg=THEME["accent2"],
                 font=("Courier New", 8)).pack(padx=10, anchor="w")
        prog = ttk.Progressbar(win, mode="determinate", maximum=100,
                               style="custom.Horizontal.TProgressbar")
        prog.pack(fill="x", padx=10, pady=(8, 2))
        plbl = tk.Label(win, text="Start...", bg=THEME["bg"], fg=THEME["text"],
                        font=("Courier New", 8))
        plbl.pack(padx=10, anchor="w")

        ft = self.finetuned_var.get()

        def cb(done, total, msg):
            win.after(0, lambda: (prog.config(value=100 * done / max(total, 1)),
                                  plbl.config(text=f"[{done}/{total}] {msg}")))

        def worker():
            from folder_sr import process_tiff_folder
            try:
                n = process_tiff_folder(in_dir, str(out_dir), use_finetuned=ft, progress_cb=cb)
                win.after(0, lambda: plbl.config(text=f"✓ Gotowe: {n} plikow w {out_dir.resolve()}"))
            except Exception as ex:
                import traceback
                tb = traceback.format_exc()
                win.after(0, lambda: messagebox.showerror("Blad", f"{ex}\n\n{tb[:500]}"))
        threading.Thread(target=worker, daemon=True).start()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = SentinelSRApp()
    app.mainloop()
