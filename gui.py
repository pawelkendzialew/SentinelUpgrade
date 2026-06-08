"""
Sentinel-2 Super-Resolution — GUI
===================================
Uruchom: python gui.py
"""

import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk


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
        self.geometry("1100x760")
        self.minsize(900, 640)
        self.configure(bg=THEME["bg"])
        self.resizable(True, True)

        self._pipeline_thread = None
        self._result_paths = {}
        self._images = {}          # PhotoImage cache (anti-GC)
        self._current_view = "all"
        self._zoom_compare = True  # domyślnie widok crop 1:1

        self._build_ui()
        self._bind_resize()

    # ─────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self, bg=THEME["bg"], pady=0)
        hdr.pack(fill="x", padx=20, pady=(14, 0))

        tk.Label(
            hdr,
            text="Sentinel-2  ·  Super-Resolution",
            bg=THEME["bg"],
            fg=THEME["text"],
            font=("Courier New", 16, "bold"),
        ).pack(side="left")

        tk.Label(
            hdr,
            text="10m/px  →  SEN2SR  →  2.5m/px  →  GeoTIFF + NDVI",
            bg=THEME["bg"],
            fg=THEME["text_muted"],
            font=("Courier New", 9),
        ).pack(side="right", padx=4)

        # separator
        tk.Frame(self, height=1, bg=THEME["border"]).pack(fill="x", padx=20, pady=8)

        # ── Main layout ──
        main = tk.Frame(self, bg=THEME["bg"])
        main.pack(fill="both", expand=True, padx=20, pady=0)

        # Left panel: controls
        self._build_controls(main)

        # Right panel: image viewer
        self._build_viewer(main)

        # ── Status bar ──
        self._build_statusbar()

    def _build_controls(self, parent):
        outer = tk.Frame(parent, bg=THEME["panel"], bd=0,
                         highlightthickness=1, highlightbackground=THEME["border"],
                         width=296)
        outer.pack(side="left", fill="y", padx=(0, 12), pady=0)
        outer.pack_propagate(False)

        # Dół przypięty na stałe: przycisk RUN + progres (zawsze widoczne)
        self._runbar = tk.Frame(outer, bg=THEME["panel"])
        self._runbar.pack(side="bottom", fill="x")

        # Środek przewijalny: Canvas + scrollbar + wewnętrzna ramka 'body'
        canvas = tk.Canvas(outer, bg=THEME["panel"], highlightthickness=0,
                           width=280)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        body = tk.Frame(canvas, bg=THEME["panel"])
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))

        # Przewijanie kółkiem myszy (gdy kursor nad panelem)
        def _wheel(e):
            canvas.yview_scroll(int(-e.delta / 120), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        ctrl = body          # całą zawartość pakujemy do przewijalnego body
        pad = {"padx": 16, "pady": 4}

        # ── Section: Lokalizacja ──
        self._section_label(ctrl, "LOKALIZACJA")

        tk.Label(ctrl, text="Latitude", bg=THEME["panel"], fg=THEME["text_muted"],
                 font=("Courier New", 9)).pack(anchor="w", **pad)
        self.lat_var = tk.StringVar(value="50.0647")
        self._entry(ctrl, self.lat_var).pack(fill="x", **pad)

        tk.Label(ctrl, text="Longitude", bg=THEME["panel"], fg=THEME["text_muted"],
                 font=("Courier New", 9)).pack(anchor="w", **pad)
        self.lon_var = tk.StringVar(value="19.9450")
        self._entry(ctrl, self.lon_var).pack(fill="x", **pad)

        # Preset buttons
        presets_frame = tk.Frame(ctrl, bg=THEME["panel"])
        presets_frame.pack(fill="x", padx=16, pady=(2, 8))
        presets = [
            ("Kraków",    50.0647,  19.9450),
            ("Warszawa",  52.2297,  21.0122),
            ("Gdańsk",    54.3521,  18.6462),
            ("Wrocław",   51.1079,  17.0385),
        ]
        for i, (name, lat, lon) in enumerate(presets):
            b = tk.Button(
                presets_frame,
                text=name,
                bg=THEME["btn_bg"],
                fg=THEME["text"],
                font=("Courier New", 8),
                relief="flat",
                activebackground=THEME["btn_hover"],
                activeforeground=THEME["text"],
                cursor="hand2",
                command=lambda la=lat, lo=lon: self._set_preset(la, lo),
            )
            b.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky="ew")
        presets_frame.columnconfigure(0, weight=1)
        presets_frame.columnconfigure(1, weight=1)

        # ── Section: Parametry ──
        self._section_label(ctrl, "PARAMETRY")

        tk.Label(ctrl, text="Data od", bg=THEME["panel"], fg=THEME["text_muted"],
                 font=("Courier New", 9)).pack(anchor="w", **pad)
        self.start_var = tk.StringVar(value="2023-06-01")
        self._entry(ctrl, self.start_var).pack(fill="x", **pad)

        tk.Label(ctrl, text="Data do", bg=THEME["panel"], fg=THEME["text_muted"],
                 font=("Courier New", 9)).pack(anchor="w", **pad)
        self.end_var = tk.StringVar(value="2023-09-30")
        self._entry(ctrl, self.end_var).pack(fill="x", **pad)

        tk.Label(ctrl, text="Rozmiar kafelka (px)", bg=THEME["panel"], fg=THEME["text_muted"],
                 font=("Courier New", 9)).pack(anchor="w", **pad)
        self.size_var = tk.StringVar(value="256")
        sizes = tk.OptionMenu(ctrl, self.size_var, "128", "256", "512")
        sizes.config(bg=THEME["btn_bg"], fg=THEME["text"], font=("Courier New", 10),
                     relief="flat", activebackground=THEME["btn_hover"],
                     activeforeground=THEME["text"], highlightthickness=0)
        sizes["menu"].config(bg=THEME["btn_bg"], fg=THEME["text"], font=("Courier New", 10))
        sizes.pack(fill="x", **pad)

        # MISR — fuzja wielu przelotow (czystsza klatka 10 m, mniej chmur/szumu)
        self.misr_var = tk.BooleanVar(value=False)
        misr_cb = tk.Checkbutton(
            ctrl, text="MISR: fuzja stosu czasowego (lepsza jakosc)",
            variable=self.misr_var,
            bg=THEME["panel"], fg=THEME["text"], selectcolor=THEME["btn_bg"],
            activebackground=THEME["panel"], activeforeground=THEME["accent"],
            font=("Courier New", 8), anchor="w",
        )
        misr_cb.pack(fill="x", padx=16, pady=(6, 2))

        # Dostrojenie do polskich pol (wagi GUGiK)
        self.finetuned_var = tk.BooleanVar(value=False)
        ft_cb = tk.Checkbutton(
            ctrl, text="Model dostrojony do polskich pol (GUGiK)",
            variable=self.finetuned_var,
            bg=THEME["panel"], fg=THEME["accent2"], selectcolor=THEME["btn_bg"],
            activebackground=THEME["panel"], activeforeground=THEME["accent2"],
            font=("Courier New", 8), anchor="w",
        )
        ft_cb.pack(fill="x", padx=16, pady=(0, 2))

        # ── Pipeline steps info ──
        self._section_label(ctrl, "PIPELINE")
        steps_text = (
            "① Pobierz Sentinel-2 L2A\n"
            "   (B04 B03 B02 B08 — RGB+NIR)\n\n"
            "② SEN2SR  →  x4  =  2.5 m/px\n"
            "   (SEN2SRLite NonRef RGBN)\n\n"
            "③ Eksport GeoTIFF + NDVI\n"
            "   (produkt do QGIS)"
        )
        tk.Label(
            ctrl, text=steps_text,
            bg=THEME["panel"], fg=THEME["text_muted"],
            font=("Courier New", 8),
            justify="left", wraplength=240
        ).pack(anchor="w", padx=16, pady=(4, 10))

        # ── RUN button + progres (przypięte do dołu, zawsze widoczne) ──
        runbar = self._runbar
        tk.Frame(runbar, height=1, bg=THEME["border"]).pack(fill="x")
        self.run_btn = tk.Button(
            runbar,
            text="▶  POBIERZ I POLEPSZ",
            bg=THEME["btn_accent"],
            fg=THEME["text"],
            font=("Courier New", 11, "bold"),
            relief="flat",
            activebackground="#2a7ad4",
            activeforeground=THEME["text"],
            cursor="hand2",
            pady=10,
            command=self._run,
        )
        self.run_btn.pack(fill="x", padx=16, pady=(8, 4))

        # ── Progress ──
        self.progress_label = tk.Label(
            runbar, text="", bg=THEME["panel"],
            fg=THEME["text_muted"], font=("Courier New", 8)
        )
        self.progress_label.pack(padx=16, pady=(0, 4))

        self.progress_bar = ttk.Progressbar(runbar, mode="determinate", maximum=100)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "custom.Horizontal.TProgressbar",
            troughcolor=THEME["progress_bg"],
            background=THEME["progress_fg"],
            bordercolor=THEME["border"],
            lightcolor=THEME["progress_fg"],
            darkcolor=THEME["progress_fg"],
        )
        self.progress_bar.configure(style="custom.Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 10))

    def _build_viewer(self, parent):
        viewer = tk.Frame(parent, bg=THEME["bg"])
        viewer.pack(side="right", fill="both", expand=True)

        # Tab bar
        tabs_frame = tk.Frame(viewer, bg=THEME["panel"],
                              highlightthickness=1, highlightbackground=THEME["border"])
        tabs_frame.pack(fill="x", pady=(0, 8))

        self._tab_btns = {}
        tabs = [
            ("all",     "Porównanie"),
            ("before",  "Oryginał 10m/px"),
            ("sen2sr",  "SEN2SR 2.5m/px"),
            ("ndvi",    "NDVI"),
        ]
        for key, label in tabs:
            b = tk.Button(
                tabs_frame, text=label,
                bg=THEME["btn_accent"] if key == "all" else THEME["panel"],
                fg=THEME["text"],
                font=("Courier New", 9, "bold" if key == "all" else "normal"),
                relief="flat",
                activebackground=THEME["btn_hover"],
                activeforeground=THEME["text"],
                cursor="hand2",
                padx=12, pady=6,
                command=lambda k=key: self._switch_tab(k),
            )
            b.pack(side="left")
            self._tab_btns[key] = b

        # Canvas area
        self.canvas_frame = tk.Frame(viewer, bg=THEME["bg"])
        self.canvas_frame.pack(fill="both", expand=True)

        # Placeholder
        self._placeholder = tk.Label(
            self.canvas_frame,
            text=(
                "Wyniki pojawią się tutaj po zakończeniu pipeline'u.\n\n"
                "← Ustaw parametry i kliknij ▶ POBIERZ I POLEPSZ"
            ),
            bg=THEME["bg"],
            fg=THEME["text_muted"],
            font=("Courier New", 11),
            justify="center",
        )
        self._placeholder.place(relx=0.5, rely=0.5, anchor="center")

        # Image labels (hidden until data arrives)
        self._img_frames = {}
        self._img_labels = {}
        self._img_captions = {}

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=THEME["panel"],
                       highlightthickness=1, highlightbackground=THEME["border"])
        bar.pack(fill="x", padx=0, pady=0, side="bottom")

        self.status_var = tk.StringVar(value="Gotowy do pracy.")
        tk.Label(
            bar, textvariable=self.status_var,
            bg=THEME["panel"], fg=THEME["text_muted"],
            font=("Courier New", 9),
            anchor="w", padx=12, pady=4,
        ).pack(side="left")

        self._device_label = tk.Label(
            bar,
            text=f"device: checking...",
            bg=THEME["panel"], fg=THEME["text_muted"],
            font=("Courier New", 9),
            padx=12, pady=4,
        )
        self._device_label.pack(side="right")
        self.after(200, self._check_device)

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _section_label(self, parent, text):
        f = tk.Frame(parent, bg=THEME["panel"])
        f.pack(fill="x", padx=16, pady=(12, 2))
        tk.Label(f, text=text, bg=THEME["panel"], fg=THEME["accent"],
                 font=("Courier New", 8, "bold")).pack(side="left")
        tk.Frame(f, height=1, bg=THEME["border"]).pack(side="left", fill="x",
                                                        expand=True, padx=(6, 0), pady=6)

    def _entry(self, parent, var):
        e = tk.Entry(
            parent, textvariable=var,
            bg=THEME["btn_bg"], fg=THEME["text"],
            insertbackground=THEME["text"],
            font=("Courier New", 10),
            relief="flat",
            highlightthickness=1,
            highlightbackground=THEME["border"],
            highlightcolor=THEME["accent"],
        )
        return e

    def _set_preset(self, lat, lon):
        self.lat_var.set(str(lat))
        self.lon_var.set(str(lon))

    def _check_device(self):
        import torch
        dev = "cuda ✓" if torch.cuda.is_available() else "cpu (brak GPU)"
        self._device_label.config(text=f"device: {dev}")

    def _switch_tab(self, key):
        self._current_view = key
        for k, b in self._tab_btns.items():
            if k == key:
                b.config(bg=THEME["btn_accent"], font=("Courier New", 9, "bold"))
            else:
                b.config(bg=THEME["panel"], font=("Courier New", 9))
        if self._result_paths:
            self._display_results(self._result_paths)

    def _bind_resize(self):
        self.bind("<Configure>", lambda e: self.after(50, self._on_resize))

    def _on_resize(self):
        if self._result_paths:
            self._display_results(self._result_paths)

    # ─────────────────────────────────────────────
    # PIPELINE RUNNER
    # ─────────────────────────────────────────────

    def _run(self):
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            return

        # Validate inputs
        try:
            lat   = float(self.lat_var.get())
            lon   = float(self.lon_var.get())
            edge  = int(self.size_var.get())
            assert 48 <= lat <= 55, "Latitude poza Polską"
            assert 14 <= lon <= 25, "Longitude poza Polską"
        except Exception as ex:
            messagebox.showerror("Błąd parametrów", str(ex))
            return

        self.run_btn.config(state="disabled", text="⏳  Przetwarzanie...")
        self.progress_bar["value"] = 0
        self.status_var.set("Uruchamianie pipeline'u...")

        def worker():
            try:
                from pipeline import run_pipeline

                def cb(msg, pct):
                    self.after(0, lambda: self._update_progress(msg, pct))

                results = run_pipeline(
                    lat=lat, lon=lon,
                    start_date=self.start_var.get(),
                    end_date=self.end_var.get(),
                    edge_size=edge,
                    use_misr=self.misr_var.get(),         # fuzja stosu czasowego
                    use_finetuned=self.finetuned_var.get(),  # wagi dostrojone do PL
                    progress_cb=cb,
                )
                self.after(0, lambda: self._on_pipeline_done(results))
            except Exception as ex:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_pipeline_error(str(ex), tb))

        self._pipeline_thread = threading.Thread(target=worker, daemon=True)
        self._pipeline_thread.start()

    def _update_progress(self, msg, pct):
        self.progress_label.config(text=msg)
        self.progress_bar["value"] = pct
        self.status_var.set(f"[{pct:3d}%]  {msg}")

    def _on_pipeline_done(self, results):
        self._result_paths = results
        elapsed = results.get("elapsed_s", 0)

        self.run_btn.config(state="normal", text="▶  POBIERZ I POLEPSZ")
        self.progress_bar["value"] = 100
        self.progress_label.config(text="✓ Gotowe!")
        self.status_var.set(f"✓ Pipeline zakończony w {elapsed:.1f}s  —  wyniki w folderze output/")

        # Ukryj placeholder
        self._placeholder.place_forget()

        self._display_results(results)

    def _on_pipeline_error(self, msg, tb):
        self.run_btn.config(state="normal", text="▶  POBIERZ I POLEPSZ")
        self.progress_label.config(text="✗ Błąd!")
        self.status_var.set(f"✗ Błąd: {msg}")
        messagebox.showerror("Błąd pipeline'u", f"{msg}\n\n{tb[:600]}")

    # ─────────────────────────────────────────────
    # IMAGE DISPLAY
    # ─────────────────────────────────────────────

    def _display_results(self, paths):
        # Wyczyść stary widok
        for w in self.canvas_frame.winfo_children():
            w.destroy()
        self._images.clear()

        view = self._current_view
        frames_w = self.canvas_frame.winfo_width()  or 780
        frames_h = self.canvas_frame.winfo_height() or 500

        if view == "all":
            self._show_comparison(paths, frames_w, frames_h)
        elif view == "before":
            self._show_single(paths["original"],    "ORYGINAŁ  10 m/px",  frames_w, frames_h, THEME["text_muted"])
        elif view == "sen2sr":
            self._show_single(paths["sen2sr"],      "SEN2SR  2.5 m/px",   frames_w, frames_h, THEME["accent"])
        elif view == "ndvi":
            if paths.get("ndvi_png"):
                self._show_single(paths["ndvi_png"], "NDVI  (zielony=wegetacja, czerwony=słaba)",
                                  frames_w, frames_h, THEME["accent2"])
            else:
                self._show_info("NDVI dostępne po eksporcie GeoTIFF\n(wymaga georeferencji z cubo).")

    def _fit_image(self, pil_img, max_w, max_h):
        iw, ih = pil_img.size
        scale = min(max_w / iw, max_h / ih, 1.0)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        return pil_img.resize((nw, nh), Image.LANCZOS)

    def _show_comparison(self, paths, w, h):
        # ── toggle ZOOM / PELNY ──
        toggle_bar = tk.Frame(self.canvas_frame, bg=THEME["bg"])
        toggle_bar.place(x=0, y=0, width=w, height=24)
        tk.Button(
            toggle_bar,
            text="ZOOM 1:1" if not self._zoom_compare else "PELNY OBRAZ",
            bg=THEME["btn_bg"], fg=THEME["accent"],
            font=("Courier New", 8), relief="flat", cursor="hand2",
            padx=6, pady=1,
            command=self._toggle_zoom,
        ).pack(side="right", padx=6, pady=2)
        tk.Label(
            toggle_bar,
            text="ZOOM — ten sam obszar geo, rozne px/m" if self._zoom_compare
                 else "PELNY OBRAZ — kazdy obraz skalowany do panelu",
            bg=THEME["bg"], fg=THEME["text_muted"],
            font=("Courier New", 8),
        ).pack(side="left", padx=8, pady=2)

        img_top = 28
        img_h = h - img_top - 4

        items = [
            ("original",  "ORYGINAL  10 m/px",  THEME["text_muted"]),
            ("sen2sr",    "SEN2SR   2.5 m/px",   THEME["accent"]),
        ]

        COLS = len(items)
        col_w = (w - 16) // COLS

        # Wczytaj wszystkie obrazy z dysku
        pil_imgs = {}
        for key, _, _ in items:
            if key in paths:
                try:
                    pil_imgs[key] = Image.open(paths[key])
                except Exception:
                    pass

        # Wyznacz wspolczynniki skali z rzeczywistych rozmiarow
        sf = {"original": 1, "sen2sr": 1}
        orig = pil_imgs.get("original")
        if orig:
            if "sen2sr" in pil_imgs:
                sf["sen2sr"] = pil_imgs["sen2sr"].size[0] // max(orig.size[0], 1)

        # Rozmiar cropu: 25% obszaru oryginalu (ten sam kadr na kazde zdjecie)
        geo_w = max(32, orig.size[0] // 4) if orig else 64
        geo_h = max(32, orig.size[1] // 4) if orig else 64

        disp_w = col_w - 12
        disp_h = img_h - 44

        for col, (key, label, color) in enumerate(items):
            frame = tk.Frame(self.canvas_frame, bg=THEME["bg"])
            frame.place(x=col * col_w + 4, y=img_top,
                        width=col_w - 4, height=img_h)

            tk.Label(frame, text=label, bg=THEME["bg"], fg=color,
                     font=("Courier New", 9, "bold")).pack(pady=(4, 2))

            if key not in pil_imgs:
                continue

            try:
                pil = pil_imgs[key]

                if self._zoom_compare and orig:
                    # Crop srodka — ten sam obszar geograficzny
                    s = sf[key]
                    cx, cy = pil.size[0] // 2, pil.size[1] // 2
                    hw, hh = (geo_w * s) // 2, (geo_h * s) // 2
                    box = (max(0, cx - hw), max(0, cy - hh),
                           min(pil.size[0], cx + hw), min(pil.size[1], cy + hh))
                    crop = pil.crop(box)

                    # Skala do panelu (bez ograniczenia 1.0 — upscale dozwolony)
                    sc = min(disp_w / max(crop.width, 1), disp_h / max(crop.height, 1))
                    nw, nh = max(1, int(crop.width * sc)), max(1, int(crop.height * sc))
                    # Oryginal: NEAREST zeby widac piksele; SR: LANCZOS
                    resample = Image.NEAREST if key == "original" else Image.LANCZOS
                    pil_out = crop.resize((nw, nh), resample)
                    info = f"natywnie {pil.size[0]}x{pil.size[1]}  |  crop {crop.size[0]}x{crop.size[1]}"
                else:
                    pil_out = self._fit_image(pil, disp_w, disp_h)
                    info = f"natywnie {pil.size[0]}x{pil.size[1]} px"

                photo = ImageTk.PhotoImage(pil_out)
                self._images[key] = photo
                tk.Label(frame, image=photo, bg=THEME["bg"]).pack(expand=True)
                tk.Label(frame, text=info, bg=THEME["bg"], fg=THEME["text_muted"],
                         font=("Courier New", 7)).pack(pady=(2, 4))

            except Exception as ex:
                tk.Label(frame, text=f"Blad:\n{ex}", bg=THEME["bg"],
                         fg=THEME["warn"], font=("Courier New", 9)).pack()

    def _toggle_zoom(self):
        self._zoom_compare = not self._zoom_compare
        if self._result_paths:
            self._display_results(self._result_paths)

    def _show_info(self, text):
        tk.Label(
            self.canvas_frame, text=text,
            bg=THEME["bg"], fg=THEME["text_muted"],
            font=("Courier New", 11), justify="center",
        ).place(relx=0.5, rely=0.5, anchor="center")

    def _show_single(self, path, label, w, h, color):
        cap = tk.Label(self.canvas_frame, text=label, bg=THEME["bg"], fg=color,
                       font=("Courier New", 11, "bold"))
        cap.pack(pady=(8, 4))
        try:
            pil = Image.open(path)
            pil_fit = self._fit_image(pil, w - 20, h - 50)
            photo = ImageTk.PhotoImage(pil_fit)
            self._images["single"] = photo
            lbl = tk.Label(self.canvas_frame, image=photo, bg=THEME["bg"])
            lbl.pack(expand=True)
            sz_lbl = tk.Label(
                self.canvas_frame,
                text=f"Rozmiar: {pil.size[0]}×{pil.size[1]} px  |  Plik: {Path(path).name}",
                bg=THEME["bg"], fg=THEME["text_muted"],
                font=("Courier New", 8),
            )
            sz_lbl.pack(pady=(2, 6))
        except Exception as ex:
            tk.Label(self.canvas_frame, text=f"Błąd ładowania:\n{ex}",
                     bg=THEME["bg"], fg=THEME["warn"],
                     font=("Courier New", 10)).pack(expand=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = SentinelSRApp()
    app.mainloop()
