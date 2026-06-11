from __future__ import annotations

import datetime as dt
import json
import platform
import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import joblib
import numpy as np
import pandas as pd

from .config import CHEMICAL_FEATURES, CHEMICAL_TARGETS
from .features import build_prediction_row, standardize_dataset
from .modeling import train_best_model
from .paths import get_default_data_dir, get_legacy_data_dir
from .preprocessing import convert_excels
from .utils import log_safe


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
# Two palettes: light and dark. They are swapped at runtime by ``apply_theme``.
PALETTES = {
    "light": {
        "name": "light",
        "bg": "#f5f7fb",
        "card": "white",
        "card_border": "#d9e2ec",
        "sidebar_bg": "#0b1220",
        "sidebar_fg": "#cbd5e1",
        "sidebar_active_bg": "#1d4ed8",
        "sidebar_active_fg": "white",
        "sidebar_muted": "#94a3b8",
        "sidebar_footer_fg": "#64748b",
        "text": "#102a43",
        "text_muted": "#627d98",
        "text_soft": "#486581",
        "log_bg": "#0b1220",
        "log_fg": "#d9e2ec",
        "input_bg": "white",
        "input_fg": "#102a43",
        "input_error_bg": "#fff1f2",
        "input_error_border": "#dc2626",
        "ok": "#16a34a",
        "warn": "#d97706",
        "bad": "#dc2626",
        "ok_bg": "#dcfce7",
        "warn_bg": "#fef3c7",
        "bad_bg": "#fee2e2",
    },
    "dark": {
        "name": "dark",
        "bg": "#0f172a",
        "card": "#1e293b",
        "card_border": "#334155",
        "sidebar_bg": "#020617",
        "sidebar_fg": "#cbd5e1",
        "sidebar_active_bg": "#2563eb",
        "sidebar_active_fg": "white",
        "sidebar_muted": "#94a3b8",
        "sidebar_footer_fg": "#64748b",
        "text": "#f1f5f9",
        "text_muted": "#94a3b8",
        "text_soft": "#cbd5e1",
        "log_bg": "#020617",
        "log_fg": "#cbd5e1",
        "input_bg": "#0f172a",
        "input_fg": "#f1f5f9",
        "input_error_bg": "#3f1d1d",
        "input_error_border": "#f87171",
        "ok": "#4ade80",
        "warn": "#fbbf24",
        "bad": "#f87171",
        "ok_bg": "#14532d",
        "warn_bg": "#78350f",
        "bad_bg": "#7f1d1d",
    },
}

ACCENTS = {
    "policloruro_aluminio": "#14b8a6",
    "coagulante_organico": "#f59e0b",
    "floculante_cationico": "#8b5cf6",
}

CHEMICAL_DISPLAY = {
    "policloruro_aluminio": "Policloruro aluminio",
    "coagulante_organico": "Coagulante organico",
    "floculante_cationico": "Floculante cationico",
}


# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------
class Tooltip:
    """Lightweight hover tooltip. Attached to any widget via ``attach``."""

    def __init__(self, widget: tk.Widget, text: str, palette: dict) -> None:
        self.widget = widget
        self.text = text
        self.palette = palette
        self.tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _show(self, _event: tk.Event) -> None:
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#102a43",
            foreground="white",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=6,
            font=("Segoe UI", 9),
            wraplength=280,
        )
        label.pack()

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class FRC40App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FRC40 - Calculo de dosis quimica")
        self.geometry("1080x760")
        self.minsize(860, 560)

        # Silence the noisy pandas FutureWarnings about silent downcasting in
        # bfill/ffill/replace. The behaviour change is opt-in and harmless for
        # this app (we always re-coerce numerics afterwards). Setting this at
        # startup keeps the console output clean for the operator.
        pd.set_option("future.no_silent_downcasting", True)

        default_data_dir = get_default_data_dir(create=True)
        self.output_dir = tk.StringVar(value=str(default_data_dir))
        self.datos_path = tk.StringVar()
        self.quimicos_path = tk.StringVar()
        self.train_status = tk.StringVar(value="Listo para cargar los partes historicos.")
        self.predict_status = tk.StringVar(value="Carga una referencia antes de calcular dosis.")
        self.models_dir = Path(self.output_dir.get()) / "models"
        self.training = False
        self.dataset_rows = tk.StringVar(value="--")
        self.models_path_text = tk.StringVar(value="Sin referencia cargada")
        self.active_model_text = tk.StringVar(value="Sin referencia cargada")
        self.selected_reference_path = tk.StringVar(value="Carpeta: --")
        self.history_records: list[dict] = []
        self.current_metadata: dict = {}
        self.theme_name = self._load_preferred_theme()
        self.palette = PALETTES[self.theme_name]
        self.predict_log_visible = tk.BooleanVar(value=False)
        self.train_log_visible = tk.BooleanVar(value=False)
        self.history_refresh_needed = True
        self._predict_canvas: tk.Canvas | None = None
        self._predict_scroll_frame: tk.Frame | None = None

        # Predict inputs.
        self.pred_fecha = tk.StringVar(value=dt.date.today().isoformat())
        self.pred_caudal = tk.StringVar(value="50")
        self.pred_dqo_entrada = tk.StringVar(value="22000")
        self.pred_dqo_salida = tk.StringVar(value="900")
        self.pred_fecha.trace_add("write", lambda *_a: self._validate_predict_inputs())
        self.pred_caudal.trace_add("write", lambda *_a: self._validate_predict_inputs())
        self.pred_dqo_entrada.trace_add("write", lambda *_a: self._validate_predict_inputs())
        self.pred_dqo_salida.trace_add("write", lambda *_a: self._validate_predict_inputs())

        # Validation state per field.
        self.input_error_vars: dict[str, tk.StringVar] = {
            "caudal": tk.StringVar(value=""),
            "dqo_entrada": tk.StringVar(value=""),
            "dqo_salida": tk.StringVar(value=""),
            "fecha": tk.StringVar(value=""),
        }
        self._input_error_frames: dict[str, ttk.Frame] = {}
        self._input_entry_widgets: dict[str, ttk.Entry] = {}

        # Result vars (kg/dia) and reliability text per chemical.
        self.result_vars: dict[str, tk.StringVar] = {
            key: tk.StringVar(value="-- kg/dia") for key in CHEMICAL_TARGETS
        }
        self.result_summary_var = tk.StringVar(value="Aun no se ha calculado ninguna dosis.")
        self.best_model_vars: dict[str, tk.StringVar] = {
            key: tk.StringVar(value="--") for key in CHEMICAL_TARGETS
        }
        self.result_pills: dict[str, tk.Frame] = {}

        self.configure(bg=self.palette["bg"])
        self.setup_style()

        # Shell.
        shell = tk.Frame(self, bg=self.palette["bg"])
        shell.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(shell, bg=self.palette["sidebar_bg"], width=250)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.brand = tk.Frame(self.sidebar, bg=self.palette["sidebar_bg"], padx=22, pady=22)
        self.brand.pack(fill="x")
        self.brand_title = tk.Label(
            self.brand, text="FRC40", bg=self.palette["sidebar_bg"], fg="white", font=("Segoe UI", 24, "bold")
        )
        self.brand_title.pack(anchor="w")
        self.brand_subtitle = tk.Label(
            self.brand,
            text="Apoyo a dosificacion diaria",
            bg=self.palette["sidebar_bg"],
            fg=self.palette["sidebar_muted"],
            font=("Segoe UI", 10),
        )
        self.brand_subtitle.pack(anchor="w", pady=(2, 0))

        self.nav_buttons: dict[str, tk.Button] = {}
        self.add_nav_button("predict", "01  Calcular dosis")
        self.add_nav_button("train", "02  Actualizar datos")
        self.add_nav_button("history", "03  Referencias guardadas")
        self.add_nav_button("stats", "04  Control de calidad")

        # Footer: theme toggle + caption.
        sidebar_footer = tk.Frame(self.sidebar, bg=self.palette["sidebar_bg"], padx=22, pady=18)
        sidebar_footer.pack(side="bottom", fill="x")
        self.theme_button = self.make_button(
            sidebar_footer,
            "Modo claro" if self.theme_name == "light" else "Modo oscuro",
            self.toggle_theme,
            variant="sidebar",
            anchor="w",
        )
        self.theme_button.pack(fill="x", pady=(0, 12))
        self.sidebar_caption = tk.Label(
            sidebar_footer,
            text="Introduce caudal y DQO para obtener una dosis orientativa en kg/dia.",
            bg=self.palette["sidebar_bg"],
            fg=self.palette["sidebar_footer_fg"],
            wraplength=190,
            justify="left",
            font=("Segoe UI", 9),
        )
        self.sidebar_caption.pack(anchor="w")

        # Main area.
        self.main_area = tk.Frame(shell, bg=self.palette["bg"], padx=22, pady=20)
        self.main_area.pack(side="left", fill="both", expand=True)
        self.main_area.columnconfigure(0, weight=1)
        self.main_area.rowconfigure(1, weight=1)

        self.page_title = tk.StringVar(value="")
        self.page_subtitle = tk.StringVar(value="")
        self.page_header = tk.Frame(self.main_area, bg=self.palette["bg"])
        self.page_header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        self.page_title_label = tk.Label(
            self.page_header,
            textvariable=self.page_title,
            bg=self.palette["bg"],
            fg=self.palette["text"],
            font=("Segoe UI", 22, "bold"),
        )
        self.page_title_label.pack(anchor="w")
        self.page_subtitle_label = tk.Label(
            self.page_header,
            textvariable=self.page_subtitle,
            bg=self.palette["bg"],
            fg=self.palette["text_muted"],
            font=("Segoe UI", 10),
        )
        self.page_subtitle_label.pack(anchor="w", pady=(4, 0))

        self.pages = tk.Frame(self.main_area, bg=self.palette["bg"])
        self.pages.grid(row=1, column=0, sticky="nsew")
        self.pages.columnconfigure(0, weight=1)
        self.pages.rowconfigure(0, weight=1)

        self.train_tab = ttk.Frame(self.pages)
        self.history_tab = ttk.Frame(self.pages)
        self.stats_tab = ttk.Frame(self.pages)
        self.predict_tab = ttk.Frame(self.pages)
        for page in [self.train_tab, self.history_tab, self.stats_tab, self.predict_tab]:
            page.grid(row=0, column=0, sticky="nsew")

        self.build_train_tab()
        self.build_history_tab()
        self.build_stats_tab()
        self.build_predict_tab()
        self.refresh_model_history(select_latest=True)
        self.show_page("predict" if self.history_records else "train")
        self._validate_predict_inputs()

    # ------------------------------------------------------------------ Theme
    def _settings_path(self) -> Path:
        return get_default_data_dir() / "ui_settings.json"

    def _load_preferred_theme(self) -> str:
        path = self._settings_path()
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("theme") in PALETTES:
                    return data["theme"]
            except (OSError, json.JSONDecodeError):
                pass
        return "light"

    def _save_preferred_theme(self) -> None:
        path = self._settings_path()
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"theme": self.theme_name}, fh, ensure_ascii=False)
        except OSError:
            pass

    def toggle_theme(self) -> None:
        self.theme_name = "dark" if self.theme_name == "light" else "light"
        self.palette = PALETTES[self.theme_name]
        self.theme_button.configure(text="Modo claro" if self.theme_name == "light" else "Modo oscuro")
        self._save_preferred_theme()
        self.rebuild_for_theme()

    def rebuild_for_theme(self) -> None:
        # Cheap path: rebuild only widgets that are theme-sensitive. The card
        # content is recreated to apply the new background colours.
        self.configure(bg=self.palette["bg"])
        self.setup_style()
        self.theme_button.configure(
            text="Modo claro" if self.theme_name == "light" else "Modo oscuro"
        )
        # Recreate the visible page.
        key = self._current_page_key
        # Update sidebar colours.
        for w in (self.sidebar, self.brand, self.theme_button.master):
            try:
                w.configure(bg=self.palette["sidebar_bg"])
            except tk.TclError:
                pass
        self.brand_title.configure(bg=self.palette["sidebar_bg"], fg="white")
        self.brand_subtitle.configure(bg=self.palette["sidebar_bg"], fg=self.palette["sidebar_muted"])
        self.sidebar_caption.configure(bg=self.palette["sidebar_bg"], fg=self.palette["sidebar_footer_fg"])
        # Sidebar buttons need a fresh hover colour so their saved attribute is
        # in sync.
        for nav_key, button in self.nav_buttons.items():
            button.default_bg = (
                self.palette["sidebar_active_bg"] if nav_key == key else self.palette["sidebar_bg"]
            )
            button.hover_bg = self.palette["sidebar_active_bg"] if nav_key == key else "#111c2f"
            button.configure(
                bg=button.default_bg,
                fg=self.palette["sidebar_active_fg"] if nav_key == key else self.palette["sidebar_fg"],
                activebackground=button.hover_bg,
            )
        # Main area and headers.
        self.main_area.configure(bg=self.palette["bg"])
        self.pages.configure(bg=self.palette["bg"])
        self.page_header.configure(bg=self.palette["bg"])
        self.page_title_label.configure(bg=self.palette["bg"], fg=self.palette["text"])
        self.page_subtitle_label.configure(bg=self.palette["bg"], fg=self.palette["text_muted"])
        # Re-render visible page content.
        if key == "predict":
            self.build_predict_tab()
        elif key == "train":
            self.build_train_tab()
        elif key == "history":
            self.build_history_tab()
        elif key == "stats":
            self.build_stats_tab()
        # Re-show the current page so titles + nav highlights pick up theme.
        self.show_page(key)
        if self.history_records:
            self.update_stats(self.current_metadata, int(self.dataset_rows.get()) if self.dataset_rows.get().isdigit() else 0)
        self._validate_predict_inputs()

    # --------------------------------------------------------------- Helpers
    def add_nav_button(self, key: str, text: str) -> None:
        button = self.make_button(
            self.sidebar,
            text=text,
            command=lambda: self.show_page(key),
            variant="sidebar",
            anchor="w",
        )
        button.pack(fill="x", padx=10, pady=3)
        self.nav_buttons[key] = button

    @property
    def _current_page_key(self) -> str:
        for key, page in (
            ("predict", self.predict_tab),
            ("train", self.train_tab),
            ("history", self.history_tab),
            ("stats", self.stats_tab),
        ):
            if str(page) == str(self.pages.focus_get() or ""):
                return key
        # Fallback: tkraise is what really decides, so check grid_slaves.
        for key, page in (
            ("predict", self.predict_tab),
            ("train", self.train_tab),
            ("history", self.history_tab),
            ("stats", self.stats_tab),
        ):
            slaves = self.pages.grid_slaves(row=0, column=0)
            if slaves and str(slaves[0]) == str(page):
                return key
        return "predict"

    def show_page(self, key: str) -> None:
        self._current_page_key_resolved = key
        titles = {
            "predict": ("Calcular dosis", "Introduce los datos del dia y obtiene una recomendacion orientativa."),
            "train": ("Actualizar datos de referencia", "Carga los Excel historicos cuando haya informacion nueva."),
            "history": ("Referencias guardadas", "Elige con que historico quieres calcular las dosis."),
            "stats": ("Control de calidad", "Revisa el error esperado y la fiabilidad de la referencia cargada."),
        }
        pages = {
            "train": self.train_tab,
            "history": self.history_tab,
            "stats": self.stats_tab,
            "predict": self.predict_tab,
        }
        self.page_title.set(titles[key][0])
        self.page_subtitle.set(titles[key][1])
        for nav_key, button in self.nav_buttons.items():
            if nav_key == key:
                button.default_bg = self.palette["sidebar_active_bg"]
                button.hover_bg = "#2563eb"
                button.configure(
                    bg=self.palette["sidebar_active_bg"],
                    fg="white",
                    activebackground="#2563eb",
                )
            else:
                button.default_bg = self.palette["sidebar_bg"]
                button.hover_bg = "#111c2f"
                button.configure(
                    bg=self.palette["sidebar_bg"],
                    fg=self.palette["sidebar_fg"],
                    activebackground="#111c2f",
                )
        pages[key].tkraise()
        if key == "history" and self.history_refresh_needed:
            self.refresh_model_history(select_latest=False)
            self.history_refresh_needed = False
        # Rebuild the visible page FIRST so its widgets (stats_tree, etc.) exist
        # before we try to repaint them with the current metadata.
        if key == "predict":
            self.build_predict_tab()
        elif key == "train":
            self.build_train_tab()
        elif key == "history":
            self.build_history_tab()
        elif key == "stats":
            self.build_stats_tab()
        # Now it is safe to repaint the stats table with the active metadata
        # (read from the AppData / legacy reference that load_history_model
        # has already loaded into self.current_metadata).
        if key == "stats" and self.current_metadata:
            self.update_stats(self.current_metadata, self.dataset_rows.get())
        self._validate_predict_inputs()

    def setup_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Card.TFrame", background=self.palette["card"], relief="flat")
        style.configure("Page.TFrame", background=self.palette["bg"])
        style.configure("CardTitle.TLabel", background=self.palette["card"], foreground=self.palette["text"], font=("Segoe UI", 12, "bold"))
        style.configure("Muted.TLabel", background=self.palette["card"], foreground=self.palette["text_muted"])
        style.configure("MutedOnPage.TLabel", background=self.palette["bg"], foreground=self.palette["text_muted"])
        style.configure("Status.TLabel", background=self.palette["bg"], foreground=self.palette["text_soft"])
        style.configure("Help.TLabel", background=self.palette["card"], foreground=self.palette["bad"], font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7))
        style.configure("TButton", padding=(10, 6))
        style.configure(
            "TEntry",
            fieldbackground=self.palette["input_bg"],
            foreground=self.palette["input_fg"],
            insertcolor=self.palette["input_fg"],
            padding=(6, 4),
        )
        style.map(
            "TEntry",
            fieldbackground=[("disabled", self.palette["bg"])],
            foreground=[("disabled", self.palette["text_muted"])],
        )
        style.configure(
            "Error.TEntry",
            fieldbackground=self.palette["input_error_bg"],
            bordercolor=self.palette["input_error_border"],
            lightcolor=self.palette["input_error_border"],
            darkcolor=self.palette["input_error_border"],
        )
        style.configure("TNotebook", background=self.palette["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", self.palette["card"])], foreground=[("selected", self.palette["text"])])
        style.configure(
            "Treeview",
            rowheight=30,
            font=("Segoe UI", 10),
            fieldbackground=self.palette["card"],
            background=self.palette["card"],
            foreground=self.palette["text"],
        )
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def make_button(
        self,
        parent,
        text: str,
        command,
        variant: str = "primary",
        anchor: str = "center",
        width: int | None = None,
    ) -> tk.Button:
        variants = {
            "primary": {
                "bg": "#2563eb",
                "hover": "#1d4ed8",
                "fg": "white",
                "active": "#1e40af",
                "disabled": "#93c5fd",
            },
            "secondary": {
                "bg": "#e2e8f0",
                "hover": "#cbd5e1",
                "fg": "#102a43",
                "active": "#b6c2d1",
                "disabled": "#eef2f7",
            },
            "success": {
                "bg": "#16a34a",
                "hover": "#15803d",
                "fg": "white",
                "active": "#166534",
                "disabled": "#86efac",
            },
            "ghost": {
                "bg": self.palette["card"],
                "hover": self.palette["card_border"],
                "fg": self.palette["text"],
                "active": self.palette["card_border"],
                "disabled": self.palette["card"],
            },
            "sidebar": {
                "bg": self.palette["sidebar_bg"],
                "hover": "#111c2f",
                "fg": self.palette["sidebar_fg"],
                "active": "#111c2f",
                "disabled": self.palette["sidebar_bg"],
            },
        }
        colors = variants[variant]
        button = tk.Button(
            parent,
            text=text,
            command=command,
            anchor=anchor,
            bd=0,
            relief="flat",
            overrelief="flat",
            highlightthickness=0,
            padx=18,
            pady=11,
            bg=colors["bg"],
            fg=colors["fg"],
            activebackground=colors["active"],
            activeforeground=colors["fg"],
            disabledforeground="#64748b",
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            takefocus=False,
        )
        if width:
            button.configure(width=width)
        button.default_bg = colors["bg"]
        button.hover_bg = colors["hover"]
        button.disabled_bg = colors["disabled"]
        button.bind("<Enter>", lambda event: event.widget.configure(bg=event.widget.hover_bg))
        button.bind("<Leave>", lambda event: event.widget.configure(bg=event.widget.default_bg))
        return button

    def section(self, parent, title: str, subtitle: str | None = None) -> ttk.Frame:
        container = ttk.Frame(parent, style="Card.TFrame")
        ttk.Label(container, text=title, style="CardTitle.TLabel").pack(anchor="w")
        if subtitle:
            ttk.Label(container, text=subtitle, style="Muted.TLabel").pack(anchor="w", pady=(3, 0))
        return container

    def metric_card(self, parent, title: str, value_var: tk.StringVar, accent: str) -> tk.Frame:
        card = tk.Frame(parent, bg=self.palette["card"], highlightthickness=1, highlightbackground=self.palette["card_border"])
        tk.Frame(card, bg=accent, width=5).pack(side="left", fill="y")
        body = tk.Frame(card, bg=self.palette["card"], padx=14, pady=12)
        body.pack(side="left", fill="both", expand=True)
        tk.Label(body, text=title, bg=self.palette["card"], fg=self.palette["text_muted"], font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(body, textvariable=value_var, bg=self.palette["card"], fg=self.palette["text"], font=("Segoe UI", 16, "bold")).pack(
            anchor="w", pady=(4, 0)
        )
        return card

    def step_card(self, parent, number: str, title: str, text: str, accent: str) -> tk.Frame:
        card = tk.Frame(parent, bg=self.palette["card"], highlightthickness=1, highlightbackground=self.palette["card_border"])
        top = tk.Frame(card, bg=accent, height=4)
        top.pack(fill="x")
        body = tk.Frame(card, bg=self.palette["card"], padx=14, pady=12)
        body.pack(fill="both", expand=True)
        tk.Label(body, text=number, bg=self.palette["card"], fg=accent, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(body, text=title, bg=self.palette["card"], fg=self.palette["text"], font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(3, 0))
        tk.Label(
            body,
            text=text,
            bg=self.palette["card"],
            fg=self.palette["text_muted"],
            font=("Segoe UI", 9),
            wraplength=220,
            justify="left",
        ).pack(anchor="w", pady=(5, 0))
        return card

    def labeled_input(
        self,
        parent,
        label: str,
        var: tk.StringVar,
        help_text: str,
        key: str,
        tooltip: str,
        extra_widget: tk.Widget | None = None,
    ) -> ttk.Frame:
        """Build a labeled input row with inline error message and tooltip."""
        row = ttk.Frame(parent, style="Card.TFrame")
        ttk.Label(row, text=label, style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 14), pady=(0, 2))
        Tooltip(row, tooltip, self.palette)
        entry = ttk.Entry(row, textvariable=var, width=30)
        entry.grid(row=0, column=1, sticky="w")
        if extra_widget is not None:
            extra_widget.grid(row=0, column=2, sticky="w", padx=(8, 0))
        error_label = ttk.Label(row, textvariable=self.input_error_vars[key], style="Help.TLabel")
        error_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))
        self._input_error_frames[key] = row
        self._input_entry_widgets[key] = entry
        return row

    def _compact_number_input(
        self,
        parent: tk.Widget,
        *,
        label: str,
        unit: str,
        var: tk.StringVar,
        key: str,
        tooltip: str,
        col: int,
    ) -> ttk.Frame:
        """Compact labeled number input: label / entry+unit / error message."""
        cell = ttk.Frame(parent, style="Card.TFrame")
        cell.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 12, 0))
        cell.columnconfigure(0, weight=1)
        ttk.Label(cell, text=label, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        input_row = ttk.Frame(cell, style="Card.TFrame")
        input_row.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        entry = ttk.Entry(input_row, textvariable=var, width=14)
        entry.pack(side="left")
        Tooltip(entry, tooltip, self.palette)
        tk.Label(
            input_row,
            text=unit,
            bg=self.palette["card"],
            fg=self.palette["text_muted"],
            font=("Segoe UI", 10),
        ).pack(side="left", padx=(8, 0))
        ttk.Label(cell, textvariable=self.input_error_vars[key], style="Help.TLabel").grid(
            row=2, column=0, sticky="w", pady=(2, 0)
        )
        self._input_error_frames[key] = cell
        self._input_entry_widgets[key] = entry
        return cell

    # -------------------------------------------------------- Validation
    def _set_field_error(self, key: str, message: str) -> None:
        self.input_error_vars[key].set(message)
        entry = self._input_entry_widgets.get(key)
        if entry is None:
            return
        if message:
            entry.configure(style="Error.TEntry")
        else:
            entry.configure(style="TEntry")

    def _validate_predict_inputs(self) -> None:
        any_error = False

        # Fecha
        fecha_value = self.pred_fecha.get().strip()
        try:
            if fecha_value:
                dt.date.fromisoformat(fecha_value)
            self._set_field_error("fecha", "")
        except ValueError:
            self._set_field_error("fecha", "Usa el formato AAAA-MM-DD (ej. 2026-06-11).")
            any_error = True
        if not fecha_value:
            self._set_field_error("fecha", "La fecha es obligatoria.")
            any_error = True

        # Caudal
        caudal = self._parse_float(self.pred_caudal.get())
        if caudal is None or caudal <= 0:
            self._set_field_error("caudal", "Introduce un caudal positivo (m3/dia).")
            any_error = True
        else:
            self._set_field_error("caudal", "")

        # DQO entrada
        dqo_e = self._parse_float(self.pred_dqo_entrada.get())
        if dqo_e is None or dqo_e < 0:
            self._set_field_error("dqo_entrada", "Introduce una DQO de entrada valida (mg/L, >= 0).")
            any_error = True
        else:
            self._set_field_error("dqo_entrada", "")

        # DQO salida
        dqo_s = self._parse_float(self.pred_dqo_salida.get())
        if dqo_s is None or dqo_s < 0:
            self._set_field_error("dqo_salida", "Introduce una DQO de salida valida (mg/L, >= 0).")
            any_error = True
        elif dqo_s > dqo_e and dqo_e is not None:
            self._set_field_error(
                "dqo_salida",
                "La DQO de salida no puede ser mayor que la de entrada.",
            )
            any_error = True
        else:
            self._set_field_error("dqo_salida", "")

        if hasattr(self, "predict_button") and self.predict_button.winfo_exists():
            self.predict_button.configure(state="disabled" if any_error else "normal")

    @staticmethod
    def _parse_float(raw: str) -> float | None:
        s = raw.strip().replace(",", ".")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    # -------------------------------------------------------- Onboarding
    def _build_welcome_card(self, parent: ttk.Frame) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame")
        ttk.Label(card, text="Bienvenido a FRC40", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            card,
            text="Aun no has preparado ninguna referencia. Sigue estos pasos y en 5 minutos podras calcular la primera dosis:",
            style="Muted.TLabel",
            wraplength=640,
            justify="left",
        ).pack(anchor="w", pady=(6, 10))

        steps_frame = ttk.Frame(card, style="Card.TFrame")
        steps_frame.pack(fill="x")
        for col in range(3):
            steps_frame.columnconfigure(col, weight=1)
        self.step_card(steps_frame, "01", "Ve a Actualizar datos", "Carga los Excel 'datos' y 'quimicos' del historico.", "#2563eb").grid(
            row=0, column=0, sticky="ew", padx=(0, 10), pady=4
        )
        self.step_card(steps_frame, "02", "Pulsa 'Actualizar referencia'", "La app entrena los modelos y los guarda en AppData.", "#0f766e").grid(
            row=0, column=1, sticky="ew", padx=10, pady=4
        )
        self.step_card(steps_frame, "03", "Vuelve aqui a calcular", "Introduce caudal y DQO del dia y obtendras la dosis orientativa.", "#7c3aed").grid(
            row=0, column=2, sticky="ew", padx=(10, 0), pady=4
        )

        button_row = ttk.Frame(card, style="Card.TFrame")
        button_row.pack(fill="x", pady=(14, 0))
        self.make_button(button_row, "Ir a Actualizar datos", lambda: self.show_page("train"), variant="primary").pack(side="left")
        return card

    # -------------------------------------------------------- Train tab
    def build_train_tab(self) -> None:
        frame = self.train_tab
        for child in frame.winfo_children():
            child.destroy()
        frame.configure(style="Page.TFrame")

        intro = ttk.Frame(frame, style="Card.TFrame")
        intro.grid(row=0, column=0, columnspan=3, sticky="ew", padx=14, pady=(14, 6))
        ttk.Label(intro, text="Actualizar datos historicos", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            intro,
            text="Usa esta pantalla solo cuando tengas nuevos Excel de datos y consumos.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        steps = ttk.Frame(frame, style="Page.TFrame")
        steps.grid(row=1, column=0, columnspan=3, sticky="ew", padx=14, pady=8)
        for col in range(3):
            steps.columnconfigure(col, weight=1)
        self.step_card(steps, "01", "Elegir archivos", "Selecciona datos.xlsx y quimicos.xlsx.", "#2563eb").grid(
            row=0, column=0, sticky="ew", padx=(0, 10)
        )
        self.step_card(steps, "02", "Preparar referencia", "La app limpia los datos y aprende del historico.", "#0f766e").grid(
            row=0, column=1, sticky="ew", padx=10
        )
        self.step_card(steps, "03", "Calcular dosis", "La nueva referencia queda lista para el calculo diario.", "#7c3aed").grid(
            row=0, column=2, sticky="ew", padx=(10, 0)
        )

        form = ttk.Frame(frame, style="Card.TFrame")
        form.grid(row=2, column=0, columnspan=3, sticky="ew", padx=14, pady=8)
        form.columnconfigure(1, weight=1)
        for row, (label, var, command) in enumerate(
            [
                ("datos.xlsx", self.datos_path, self.pick_datos),
                ("quimicos.xlsx", self.quimicos_path, self.pick_quimicos),
                ("Carpeta salida", self.output_dir, self.pick_output_dir),
            ]
        ):
            ttk.Label(form, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=7)
            entry = ttk.Entry(form, textvariable=var, width=95)
            entry.grid(row=row, column=1, sticky="ew", padx=6, pady=7)
            self.make_button(form, "Seleccionar", command, variant="secondary", width=13).grid(
                row=row, column=2, padx=(8, 0), pady=7
            )
            Tooltip(entry, f"Selecciona el archivo Excel para {label}.", self.palette)

        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=3, column=0, columnspan=3, sticky="ew", padx=14, pady=(4, 10))
        actions.columnconfigure(0, weight=1)
        self.train_button = self.make_button(
            actions,
            "Actualizar referencia con estos Excel",
            self.start_training,
            variant="primary",
        )
        self.train_button.grid(row=0, column=0, sticky="ew")
        self.train_progress = ttk.Progressbar(actions, mode="indeterminate")
        self.train_progress.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(actions, textvariable=self.train_status, style="Status.TLabel").grid(row=2, column=0, sticky="w", pady=(8, 0))

        # Collapsible log.
        self.train_log_toggle = self.make_button(
            frame,
            "Mostrar detalles tecnicos" if not self.train_log_visible.get() else "Ocultar detalles tecnicos",
            self._toggle_train_log,
            variant="ghost",
            anchor="w",
        )
        self.train_log_toggle.grid(row=4, column=0, columnspan=3, sticky="w", padx=14, pady=(6, 0))

        self.train_log = tk.Text(frame, height=12, state="disabled")
        self.train_log.configure(
            bg=self.palette["log_bg"],
            fg=self.palette["log_fg"],
            insertbackground=self.palette["log_fg"],
            relief="flat",
            padx=12,
            pady=10,
            font=("Consolas", 10),
        )
        if self.train_log_visible.get():
            self.train_log.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=14, pady=(6, 14))
            frame.rowconfigure(5, weight=1)
        frame.columnconfigure(1, weight=1)

    def _toggle_train_log(self) -> None:
        self.train_log_visible.set(not self.train_log_visible.get())
        self.build_train_tab()

    # -------------------------------------------------------- History tab
    def build_history_tab(self) -> None:
        frame = self.history_tab
        for child in frame.winfo_children():
            child.destroy()
        frame.configure(style="Page.TFrame")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        intro = self.section(
            frame,
            "Referencias disponibles",
            "Cada fila es un historico preparado. La referencia activa es la que se usara al calcular.",
        )
        intro.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))

        table_card = ttk.Frame(frame, style="Card.TFrame")
        table_card.grid(row=1, column=0, sticky="nsew", padx=14, pady=8)
        table_card.columnconfigure(0, weight=1)
        table_card.rowconfigure(1, weight=1)
        ttk.Label(table_card, text="Historicos preparados", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        columns = ("fecha", "filas", "policloruro", "coagulante", "floculante")
        self.history_tree = ttk.Treeview(table_card, columns=columns, show="headings", height=12)
        headers = {
            "fecha": "Fecha preparacion",
            "filas": "Dias",
            "policloruro": "Policloruro",
            "coagulante": "Coagulante",
            "floculante": "Floculante",
        }
        widths = {"fecha": 170, "filas": 70, "policloruro": 170, "coagulante": 170, "floculante": 170}
        for col, header in headers.items():
            self.history_tree.heading(col, text=header)
            self.history_tree.column(col, width=widths[col], minwidth=70, anchor="w", stretch=False)
        self.history_tree.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.history_tree.bind("<Double-1>", lambda _event: self.use_selected_history_model())
        self.history_tree.bind("<Delete>", lambda _event: self.delete_selected_history_model())
        y_scroll = ttk.Scrollbar(table_card, orient="vertical", command=self.history_tree.yview)
        x_scroll = ttk.Scrollbar(table_card, orient="horizontal", command=self.history_tree.xview)
        self.history_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        y_scroll.grid(row=1, column=1, sticky="ns", pady=(10, 0))
        x_scroll.grid(row=2, column=0, sticky="ew")

        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 14))
        actions.columnconfigure(0, weight=0)
        actions.columnconfigure(1, weight=0)
        actions.columnconfigure(2, weight=1)
        actions.columnconfigure(3, weight=0)
        self.make_button(actions, "Usar referencia seleccionada", self.use_selected_history_model, variant="primary").grid(
            row=0, column=0, sticky="w"
        )
        self.make_button(actions, "Actualizar lista", self.refresh_model_history, variant="secondary").grid(
            row=0, column=1, sticky="w", padx=(10, 0)
        )
        # Botón Eliminar: variante ghost pero con texto/borde en rojo para que
        # se vea que es destructivo sin romper el sistema de variants.
        self.delete_button = tk.Button(
            actions,
            text="Eliminar referencia",
            command=self.delete_selected_history_model,
            anchor="w",
            bd=0,
            relief="flat",
            overrelief="flat",
            highlightthickness=0,
            padx=18,
            pady=11,
            bg=self.palette["card"],
            fg=self.palette["bad"],
            activebackground=self.palette["bad_bg"],
            activeforeground=self.palette["bad"],
            disabledforeground="#64748b",
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            takefocus=False,
        )
        self.delete_button.default_bg = self.palette["card"]
        self.delete_button.hover_bg = self.palette["bad_bg"]
        self.delete_button.bind("<Enter>", lambda event: event.widget.configure(bg=event.widget.hover_bg))
        self.delete_button.bind("<Leave>", lambda event: event.widget.configure(bg=event.widget.default_bg))
        self.delete_button.grid(row=0, column=3, sticky="e", padx=(10, 0))
        ttk.Label(actions, textvariable=self.active_model_text, style="Status.TLabel", wraplength=650).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(10, 0)
        )
        ttk.Label(actions, textvariable=self.selected_reference_path, style="Status.TLabel", wraplength=650).grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(4, 0)
        )

        # Re-populate if we already have records loaded in memory.
        self._repopulate_history_tree()

    def _repopulate_history_tree(self) -> None:
        if not hasattr(self, "history_tree"):
            return
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for index, record in enumerate(self.history_records):
            metadata = record["metadata"]
            self.history_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    record["trained_at"].replace("T", " "),
                    record["dataset_rows"],
                    self.history_metric_text(metadata, "policloruro_aluminio"),
                    self.history_metric_text(metadata, "coagulante_organico"),
                    self.history_metric_text(metadata, "floculante_cationico"),
                ),
            )

    # -------------------------------------------------------- Stats tab
    def build_stats_tab(self) -> None:
        frame = self.stats_tab
        for child in frame.winfo_children():
            child.destroy()
        frame.configure(style="Page.TFrame")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        intro = self.section(
            frame,
            "Control de calidad",
            "Resumen sencillo de fiabilidad y error medio esperado por producto.",
        )
        intro.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))

        cards = ttk.Frame(frame, style="Card.TFrame")
        cards.grid(row=1, column=0, sticky="ew", padx=14, pady=8)
        for col in range(4):
            cards.columnconfigure(col, weight=1)

        self.metric_card(cards, "Dias usados", self.dataset_rows, "#3b82f6").grid(
            row=0, column=0, sticky="ew", padx=(0, 10)
        )
        self.metric_card(cards, "Policloruro", self.best_model_vars["policloruro_aluminio"], "#14b8a6").grid(
            row=0, column=1, sticky="ew", padx=10
        )
        self.metric_card(cards, "Coagulante", self.best_model_vars["coagulante_organico"], "#f59e0b").grid(
            row=0, column=2, sticky="ew", padx=10
        )
        self.metric_card(cards, "Floculante", self.best_model_vars["floculante_cationico"], "#8b5cf6").grid(
            row=0, column=3, sticky="ew", padx=(10, 0)
        )

        table_card = ttk.Frame(frame, style="Card.TFrame")
        table_card.grid(row=2, column=0, sticky="nsew", padx=14, pady=(8, 14))
        table_card.columnconfigure(0, weight=1)
        table_card.rowconfigure(1, weight=1)
        ttk.Label(table_card, text="Detalle tecnico", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")

        columns = ("quimico", "modelo", "r2", "rmse", "params")
        self.stats_tree = ttk.Treeview(table_card, columns=columns, show="headings", height=10)
        headers = {
            "quimico": "Producto",
            "modelo": "Metodo",
            "r2": "Fiabilidad",
            "rmse": "Error medio",
            "params": "Detalle tecnico",
        }
        widths = {"quimico": 190, "modelo": 150, "r2": 90, "rmse": 90, "params": 480}
        for col, header in headers.items():
            self.stats_tree.heading(col, text=header)
            self.stats_tree.column(col, width=widths[col], minwidth=70, anchor="w", stretch=False)
        self.stats_tree.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        y_scroll = ttk.Scrollbar(table_card, orient="vertical", command=self.stats_tree.yview)
        x_scroll = ttk.Scrollbar(table_card, orient="horizontal", command=self.stats_tree.xview)
        self.stats_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        y_scroll.grid(row=1, column=1, sticky="ns", pady=(10, 0))
        x_scroll.grid(row=2, column=0, sticky="ew")

        footer = ttk.Label(frame, textvariable=self.models_path_text, style="Status.TLabel", wraplength=760)
        footer.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))

    # -------------------------------------------------------- Predict tab
    def build_predict_tab(self) -> None:
        """Build the predict tab. The whole tab lives inside a Canvas +
        Scrollbar so the page can scroll vertically when the window is short.
        """
        frame = self.predict_tab
        for child in frame.winfo_children():
            child.destroy()
        frame.configure(style="Page.TFrame")
        self._input_error_frames = {}
        self._input_entry_widgets = {}

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # --- Outer scroll container ---------------------------------------
        canvas = tk.Canvas(
            frame,
            bg=self.palette["bg"],
            highlightthickness=0,
            bd=0,
        )
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 0))
        scrollbar.grid(row=0, column=1, sticky="ns")

        scroll_frame = tk.Frame(canvas, bg=self.palette["bg"])
        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_scroll_frame_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
        scroll_frame.bind("<Configure>", _on_scroll_frame_configure)

        # Mouse wheel bindings (Windows + macOS). We unbind on leave to avoid
        # hijacking the wheel when the user is scrolling a child widget.
        def _on_mousewheel(event: tk.Event) -> None:
            delta = -1 if event.delta > 0 else 1
            if abs(event.delta) >= 120:
                delta = -int(event.delta / 120)
            canvas.yview_scroll(delta, "units")

        def _bind_wheel(_event: tk.Event) -> None:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(_event: tk.Event) -> None:
            canvas.unbind_all("<MouseWheel>")

        scroll_frame.bind("<Enter>", _bind_wheel)
        scroll_frame.bind("<Leave>", _unbind_wheel)

        self._predict_canvas = canvas
        self._predict_scroll_frame = scroll_frame

        # --- Page content -------------------------------------------------
        content = tk.Frame(scroll_frame, bg=self.palette["bg"])
        content.pack(fill="both", expand=True, padx=0, pady=0)
        content.columnconfigure(0, weight=1)

        # --- 0. Page header ---------------------------------------------
        intro = ttk.Frame(content, style="Card.TFrame")
        intro.grid(row=0, column=0, sticky="ew", padx=14, pady=(0, 6))
        intro_inner = tk.Frame(intro, bg=self.palette["card"], padx=20, pady=16)
        intro_inner.pack(fill="x")
        ttk.Label(intro_inner, text="Cálculo diario de dosis", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            intro_inner,
            text="Introduce los datos del día. El resultado es una ayuda operativa, no una orden automática.",
            style="Muted.TLabel",
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(6, 0), fill="x")
        ttk.Label(
            intro_inner,
            textvariable=self.active_model_text,
            style="Muted.TLabel",
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(8, 0), fill="x")

        next_row = 1

        # --- 0b. Welcome card shown only if there are no references at all.
        if not self.history_records:
            self._build_welcome_card(content).grid(
                row=next_row, column=0, sticky="ew", padx=14, pady=(0, 10)
            )
            next_row += 1

        # --- 1. Inputs card ---------------------------------------------
        inputs_card = ttk.Frame(content, style="Card.TFrame")
        inputs_card.grid(row=next_row, column=0, sticky="ew", padx=14, pady=(6, 6))
        next_row += 1
        header_row = tk.Frame(inputs_card, bg=self.palette["card"])
        header_row.pack(fill="x", padx=18, pady=(16, 4))
        tk.Label(
            header_row,
            text="1",
            bg=self.palette["card"],
            fg="#2563eb",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left", padx=(0, 8))
        tk.Label(
            header_row,
            text="Datos del día",
            bg=self.palette["card"],
            fg=self.palette["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(side="left")

        form = ttk.Frame(inputs_card, style="Card.TFrame")
        form.pack(fill="x", padx=18, pady=(0, 16))
        form.columnconfigure(0, weight=0)
        form.columnconfigure(1, weight=0)
        form.columnconfigure(2, weight=1)

        date_row = ttk.Frame(form, style="Card.TFrame")
        date_row.grid(row=0, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(date_row, text="Fecha", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 14))
        date_entry = ttk.Entry(date_row, textvariable=self.pred_fecha, width=14)
        date_entry.grid(row=0, column=1, sticky="w")
        Tooltip(date_entry, "Fecha del día para el que quieres calcular la dosis.", self.palette)
        for col, (label, callback) in enumerate(
            [("Hoy", lambda: self._set_date(0)), ("Ayer", lambda: self._set_date(-1)), ("Hace 2", lambda: self._set_date(-2))],
            start=2,
        ):
            self.make_button(date_row, label, callback, variant="secondary", width=8).grid(
                row=0, column=col, padx=(8, 0), sticky="w"
            )
        ttk.Label(form, textvariable=self.input_error_vars["fecha"], style="Help.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(2, 4)
        )
        self._input_error_frames["fecha"] = form
        self._input_entry_widgets["fecha"] = date_entry

        numbers_row = ttk.Frame(form, style="Card.TFrame")
        numbers_row.grid(row=2, column=0, columnspan=3, sticky="ew")
        for col in range(3):
            numbers_row.columnconfigure(col, weight=1, uniform="numbers")
        self._compact_number_input(
            numbers_row,
            label="Caudal previsto",
            unit="m3/dia",
            var=self.pred_caudal,
            key="caudal",
            tooltip="Volumen total de agua a tratar durante el día, en metros cúbicos.",
            col=0,
        )
        self._compact_number_input(
            numbers_row,
            label="DQO entrada",
            unit="mg/L",
            var=self.pred_dqo_entrada,
            key="dqo_entrada",
            tooltip="Demanda química de oxígeno del agua a la entrada del proceso.",
            col=1,
        )
        self._compact_number_input(
            numbers_row,
            label="DQO salida deseada",
            unit="mg/L",
            var=self.pred_dqo_salida,
            key="dqo_salida",
            tooltip="Objetivo de calidad: DQO que quieres conseguir a la salida.",
            col=2,
        )

        # --- 2. Primary action button (full width) ---------------------
        action_bar = tk.Frame(content, bg=self.palette["bg"])
        action_bar.grid(row=next_row, column=0, sticky="ew", padx=14, pady=(6, 6))
        next_row += 1
        action_bar.columnconfigure(0, weight=1)
        self.predict_button = self.make_button(
            action_bar,
            "Calcular dosis  \u2192",
            self.predict_chemicals,
            variant="primary",
        )
        self.predict_button.grid(row=0, column=0, sticky="ew", ipady=8)

        # --- 3. Hero result card ---------------------------------------
        result_card = tk.Frame(
            content,
            bg=self.palette["card"],
            highlightthickness=1,
            highlightbackground=self.palette["card_border"],
        )
        result_card.grid(row=next_row, column=0, sticky="ew", padx=14, pady=(6, 6))
        next_row += 1
        accent_strip = tk.Frame(result_card, bg="#2563eb", width=6)
        accent_strip.pack(side="left", fill="y")
        result_body = tk.Frame(result_card, bg=self.palette["card"], padx=22, pady=18)
        result_body.pack(side="left", fill="both", expand=True)
        result_body.columnconfigure(0, weight=1)

        header_row = tk.Frame(result_body, bg=self.palette["card"])
        header_row.grid(row=0, column=0, sticky="ew")
        tk.Label(
            header_row,
            text="2",
            bg=self.palette["card"],
            fg="#2563eb",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left", padx=(0, 8))
        tk.Label(
            header_row,
            text="Dosis recomendada para hoy",
            bg=self.palette["card"],
            fg=self.palette["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(side="left")
        tk.Label(
            result_body,
            text="Pulsa 'Calcular dosis' para obtener la recomendación.",
            bg=self.palette["card"],
            fg=self.palette["text_muted"],
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, sticky="w", pady=(2, 12))

        big_row = tk.Frame(result_body, bg=self.palette["card"])
        big_row.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        for col in range(3):
            big_row.columnconfigure(col, weight=1, uniform="result")
        self.result_pills = {}
        for col, key in enumerate(CHEMICAL_TARGETS):
            cell = tk.Frame(big_row, bg=self.palette["card"])
            cell.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 12, 0))
            tk.Label(
                cell,
                text=CHEMICAL_DISPLAY[key].upper(),
                bg=self.palette["card"],
                fg=ACCENTS[key],
                font=("Segoe UI", 9, "bold"),
            ).pack(anchor="w")
            tk.Label(
                cell,
                textvariable=self.result_vars[key],
                bg=self.palette["card"],
                fg=self.palette["text"],
                font=("Segoe UI", 32, "bold"),
            ).pack(anchor="w", pady=(2, 0))
            tk.Label(
                cell,
                text="kg/dia",
                bg=self.palette["card"],
                fg=self.palette["text_muted"],
                font=("Segoe UI", 9),
            ).pack(anchor="w")
            pill = self._make_pill(cell, "Sin calcular", "muted")
            pill.pack(anchor="w", pady=(8, 0))
            self.result_pills[key] = pill

        self.verdict_var = tk.StringVar(value="Sin cálculo todavía.")
        verdict_frame = tk.Frame(result_body, bg=self.palette["card"])
        verdict_frame.grid(row=3, column=0, sticky="ew", pady=(0, 4))
        self.verdict_dot = tk.Label(
            verdict_frame,
            text="\u25CF",
            bg=self.palette["card"],
            fg=self.palette["text_muted"],
            font=("Segoe UI", 14),
        )
        self.verdict_dot.pack(side="left", padx=(0, 8))
        self.verdict_label = tk.Label(
            verdict_frame,
            textvariable=self.verdict_var,
            bg=self.palette["card"],
            fg=self.palette["text"],
            font=("Segoe UI", 12, "bold"),
        )
        self.verdict_label.pack(side="left")

        summary_label = tk.Label(
            result_body,
            textvariable=self.result_summary_var,
            bg=self.palette["card"],
            fg=self.palette["text_muted"],
            font=("Segoe UI", 10),
            wraplength=640,
            justify="left",
        )
        summary_label.grid(row=4, column=0, sticky="ew", pady=(4, 0))

        copy_bar = tk.Frame(result_body, bg=self.palette["card"])
        copy_bar.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        self.copy_button = self.make_button(
            copy_bar,
            "Copiar resultado",
            self.copy_results_to_clipboard,
            variant="ghost",
            anchor="w",
        )
        self.copy_button.pack(side="left")

        # --- 4. Status line + collapsible log ---------------------------
        status_label = ttk.Label(content, textvariable=self.predict_status, style="Status.TLabel")
        status_label.grid(row=next_row, column=0, sticky="w", padx=14, pady=(0, 4))
        next_row += 1

        self.predict_log_toggle = self.make_button(
            content,
            "Mostrar detalles técnicos" if not self.predict_log_visible.get() else "Ocultar detalles técnicos",
            self._toggle_predict_log,
            variant="ghost",
            anchor="w",
        )
        self.predict_log_toggle.grid(row=next_row, column=0, sticky="w", padx=14, pady=(6, 0))
        next_row += 1

        self.predict_log = tk.Text(content, height=8, state="disabled")
        self.predict_log.configure(
            bg=self.palette["log_bg"],
            fg=self.palette["log_fg"],
            insertbackground=self.palette["log_fg"],
            relief="flat",
            padx=12,
            pady=10,
            font=("Consolas", 10),
        )
        if self.predict_log_visible.get():
            self.predict_log.grid(
                row=next_row, column=0, sticky="nsew", padx=14, pady=(6, 14)
            )

        # Default verdict colour until the first calculation.
        if hasattr(self, "verdict_dot"):
            self._set_verdict_color("muted")

    def _toggle_predict_log(self) -> None:
        self.predict_log_visible.set(not self.predict_log_visible.get())
        self.build_predict_tab()

    def _set_date(self, delta_days: int) -> None:
        target = dt.date.today() + dt.timedelta(days=delta_days)
        self.pred_fecha.set(target.isoformat())

    def _make_pill(self, parent, text: str, level: str) -> tk.Frame:
        levels = {
            "ok": (self.palette["ok_bg"], self.palette["ok"]),
            "warn": (self.palette["warn_bg"], self.palette["warn"]),
            "bad": (self.palette["bad_bg"], self.palette["bad"]),
            "muted": (self.palette["card_border"], self.palette["text_muted"]),
        }
        bg, fg = levels.get(level, levels["muted"])
        pill = tk.Frame(parent, bg=bg, padx=10, pady=4)
        label = tk.Label(pill, text=text, bg=bg, fg=fg, font=("Segoe UI", 9, "bold"))
        label.pack()
        pill._label = label  # type: ignore[attr-defined]
        pill._bg = bg  # type: ignore[attr-defined]
        pill._fg = fg  # type: ignore[attr-defined]
        return pill

    def _set_pill(self, pill: tk.Frame | None, text: str, level: str) -> None:
        if pill is None:
            return
        levels = {
            "ok": (self.palette["ok_bg"], self.palette["ok"]),
            "warn": (self.palette["warn_bg"], self.palette["warn"]),
            "bad": (self.palette["bad_bg"], self.palette["bad"]),
            "muted": (self.palette["card_border"], self.palette["text_muted"]),
        }
        bg, fg = levels.get(level, levels["muted"])
        pill._bg = bg  # type: ignore[attr-defined]
        pill._fg = fg  # type: ignore[attr-defined]
        pill.configure(bg=bg)
        pill._label.configure(text=text, bg=bg, fg=fg)  # type: ignore[attr-defined]

    # -------------------------------------------------------- File pickers
    def pick_datos(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if path:
            self.datos_path.set(path)

    def pick_quimicos(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if path:
            self.quimicos_path.set(path)

    def pick_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)
            self.history_refresh_needed = True
            self.refresh_model_history(select_latest=False)

    # -------------------------------------------------------- Model discovery
    def model_dir_is_valid(self, models_dir: Path) -> bool:
        return (models_dir / "metadata.json").exists() and all(
            (models_dir / f"{target_key}.joblib").exists() for target_key in CHEMICAL_TARGETS
        )

    def read_model_metadata(self, models_dir: Path) -> dict:
        with open(models_dir / "metadata.json", "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        metadata.setdefault("models_dir", str(models_dir))
        return metadata

    def infer_dataset_rows(self, models_dir: Path, metadata: dict) -> int | str:
        dataset_rows = metadata.get("dataset_rows")
        if dataset_rows:
            return dataset_rows

        candidate_csvs = [
            models_dir.parent / "dataset_modelos_app.csv",
            models_dir.parent / "prepared_model_data.csv",
            models_dir.parent / "frc40_full_app.csv",
        ]
        for csv_path in candidate_csvs:
            if csv_path.exists():
                try:
                    return len(pd.read_csv(csv_path))
                except Exception:
                    continue
        return "--"

    def discover_model_runs(self) -> list[dict]:
        base_dirs: list[Path] = []
        seen_base_dirs: set[Path] = set()

        def _add_base_dir(candidate: Path | None) -> None:
            if candidate is None:
                return
            resolved = candidate.resolve()
            if resolved in seen_base_dirs:
                return
            seen_base_dirs.add(resolved)
            base_dirs.append(candidate)

        _add_base_dir(Path(self.output_dir.get()))
        _add_base_dir(get_default_data_dir())
        _add_base_dir(get_legacy_data_dir())

        candidate_dirs: list[Path] = []
        seen_dirs: set[Path] = set()
        for base_dir in base_dirs:
            legacy_dir = base_dir / "models"
            if legacy_dir.exists() and legacy_dir not in seen_dirs:
                candidate_dirs.append(legacy_dir)
                seen_dirs.add(legacy_dir)
            runs_dir = base_dir / "training_runs"
            if runs_dir.exists():
                for path in runs_dir.iterdir():
                    models_dir = path / "models"
                    if path.is_dir() and models_dir not in seen_dirs:
                        candidate_dirs.append(models_dir)
                        seen_dirs.add(models_dir)

        records = []
        for models_dir in candidate_dirs:
            if not self.model_dir_is_valid(models_dir):
                continue
            metadata = self.read_model_metadata(models_dir)
            metadata_path = models_dir / "metadata.json"
            trained_at = metadata.get("trained_at")
            if not trained_at:
                trained_at = dt.datetime.fromtimestamp(metadata_path.stat().st_mtime).isoformat(timespec="seconds")
            records.append(
                {
                    "models_dir": models_dir,
                    "metadata": metadata,
                    "trained_at": trained_at,
                    "dataset_rows": self.infer_dataset_rows(models_dir, metadata),
                    "mtime": metadata_path.stat().st_mtime,
                }
            )
        records.sort(key=lambda item: item["mtime"], reverse=True)
        return records

    def history_metric_text(self, metadata: dict, target_key: str) -> str:
        meta = metadata.get("chemicals", {}).get(target_key)
        if not meta:
            return "--"
        best = meta["best"]
        reliability = max(min(best["r2_test"], 1), 0) * 100
        return f"{reliability:.0f}% | error {best['rmse_test']:.1f} kg/dia"

    def refresh_model_history(self, select_latest: bool = False) -> None:
        self.history_records = self.discover_model_runs()
        if hasattr(self, "history_tree"):
            self._repopulate_history_tree()

        if select_latest and self.history_records:
            self.history_tree.selection_set("0")
            self.history_tree.focus("0")
            self.load_history_model(self.history_records[0])
        elif not self.history_records:
            self.active_model_text.set("Sin referencia preparada. Ve a Actualizar datos para crear la primera.")

    def use_selected_history_model(self) -> None:
        selection = self.history_tree.selection()
        if not selection:
            messagebox.showwarning("Selecciona una referencia", "Elige una fila de la lista antes de continuar.")
            return
        record = self.history_records[int(selection[0])]
        self.load_history_model(record)
        self.play_success_sound()
        messagebox.showinfo("Referencia cargada", "La referencia seleccionada ya esta activa para calcular dosis.")

    def delete_selected_history_model(self) -> None:
        """Permanently delete the selected training run from disk.

        Asks for confirmation twice: once to explain what will be removed,
        once to make it explicit the action is irreversible. If the deleted
        reference was the active one, the next most recent reference (if
        any) is activated automatically; otherwise the app falls back to
        'no reference' state.
        """
        selection = self.history_tree.selection()
        if not selection:
            messagebox.showwarning(
                "Selecciona una referencia",
                "Elige una fila de la lista antes de eliminar.",
            )
            return
        record = self.history_records[int(selection[0])]
        models_dir = record["models_dir"]
        run_dir = models_dir.parent  # training_runs/<run_id>/models -> parent is the run dir
        was_active = self.models_dir == models_dir

        # First confirmation: explain what is going to be deleted.
        first = messagebox.askyesno(
            "Eliminar referencia",
            (
                f"Vas a eliminar la referencia con {record['dataset_rows']} dias, "
                f"creada el {record['trained_at'].replace('T', ' ')}.\n\n"
                f"Carpeta:\n{run_dir}\n\n"
                "Esta accion no se puede deshacer. \u00bfContinuar?"
            ),
            icon="warning",
        )
        if not first:
            return

        # Second confirmation: type-of-error guard, in case the first popup
        # was dismissed with a reflex click.
        second = messagebox.askyesno(
            "Confirmacion final",
            "Se eliminaran los modelos y el dataset asociados a esta referencia.\n\n"
            "\u00bfSeguro que quieres eliminarla?",
            icon="warning",
        )
        if not second:
            return

        try:
            self._delete_reference_folder(run_dir, models_dir)
        except Exception as exc:
            log_safe(self.predict_log, f"ERROR al eliminar referencia: {exc}")
            messagebox.showerror(
                "No se pudo eliminar",
                f"No se pudo borrar la carpeta:\n{run_dir}\n\n{exc}",
            )
            return

        # Drop the entry from in-memory records, repopulate the table.
        self.history_records = [
            r for r in self.history_records if r["models_dir"] != models_dir
        ]
        self._repopulate_history_tree()

        if was_active:
            if self.history_records:
                # Promote the most recent remaining record.
                next_record = self.history_records[0]
                self.load_history_model(next_record)
                self.active_model_text.set(
                    f"Referencia activa: {next_record['trained_at'].replace('T', ' ')} "
                    "(se reasigno tras eliminar la anterior)"
                )
            else:
                self.active_model_text.set(
                    "Sin referencia activa. Ve a 'Actualizar datos' para entrenar una nueva."
                )
                self.selected_reference_path.set("Carpeta: --")
                self.predict_status.set("No hay ninguna referencia cargada.")
                self.dataset_rows.set("--")
                self.models_path_text.set("Sin referencia cargada")
                # Clear stats tree so no stale data is shown.
                if hasattr(self, "stats_tree"):
                    for item in self.stats_tree.get_children():
                        self.stats_tree.delete(item)
                for key in CHEMICAL_TARGETS:
                    self.best_model_vars[key].set("--")

        self.play_success_sound()
        messagebox.showinfo(
            "Referencia eliminada",
            f"Se ha borrado la carpeta:\n{run_dir}",
        )

    def _delete_reference_folder(self, run_dir: Path, models_dir: Path) -> None:
        """Remove a reference's folder from disk.

        Handles the two layouts the app produces:
          * training_runs/<run_id>/models/  -> delete the parent run folder
          * legacy <base>/models/  (no run wrapper) -> delete models folder directly
        """
        if run_dir.name.lower() == "training_runs" or run_dir.parent.name.lower() != "training_runs":
            # Legacy layout (e.g. <base>/models): delete the models folder.
            target = models_dir
        else:
            target = run_dir
        if not target.exists():
            raise FileNotFoundError(f"La carpeta ya no existe: {target}")
        import shutil

        shutil.rmtree(target)

    def load_history_model(self, record: dict) -> None:
        self.models_dir = record["models_dir"]
        metadata = record["metadata"]
        self.update_stats(metadata, metadata.get("dataset_rows") or record["dataset_rows"])
        self.active_model_text.set(f"Referencia activa: {record['trained_at'].replace('T', ' ')}")
        self.selected_reference_path.set(f"Carpeta: {record['models_dir']}")
        self.predict_status.set("Referencia cargada. Ya puedes calcular dosis.")

    # -------------------------------------------------------- Training
    def start_training(self) -> None:
        if self.training:
            return
        if not self.datos_path.get() or not self.quimicos_path.get():
            messagebox.showwarning(
                "Faltan archivos",
                "Selecciona los Excel 'datos.xlsx' y 'quimicos.xlsx' antes de continuar.",
            )
            return
        self.training = True
        self.train_button.configure(state="disabled")
        self.train_status.set("Preparando la referencia. Esto puede tardar unos minutos...")
        self.train_progress.start(12)
        thread = threading.Thread(target=self.train_pipeline, daemon=True)
        thread.start()

    def finish_training_ui(self, success: bool, message: str) -> None:
        self.training = False
        self.train_progress.stop()
        self.train_button.configure(state="normal")
        self.train_status.set(message)
        if success:
            self.predict_status.set("Referencia actualizada. Ya puedes calcular dosis.")
            self.play_success_sound()
            messagebox.showinfo(
                "Listo",
                "La nueva referencia se ha guardado. Ya puedes calcular la primera dosis del dia.",
            )

    def update_stats(self, metadata: dict, dataset_rows) -> None:
        self.current_metadata = metadata
        rows_value = dataset_rows if dataset_rows not in (None, "--") else 0
        try:
            rows_int = int(rows_value)
        except (TypeError, ValueError):
            rows_int = 0
        self.dataset_rows.set(str(rows_int) if rows_int else "--")
        self.models_path_text.set(f"Referencia guardada en: {self.models_dir}")
        if not hasattr(self, "stats_tree"):
            return
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        for target_key, label in CHEMICAL_DISPLAY.items():
            meta = metadata.get("chemicals", {}).get(target_key)
            if not meta:
                continue
            best = meta["best"]
            model_name = best["model_name"]
            r2_value = best["r2_test"]
            rmse_value = best["rmse_test"]
            reliability = max(min(r2_value, 1), 0) * 100
            self.best_model_vars[target_key].set(f"{reliability:.0f}% fiable")
            self.stats_tree.insert(
                "",
                "end",
                values=(
                    label,
                    model_name,
                    f"{reliability:.0f}%",
                    f"{rmse_value:.1f} kg/dia",
                    json.dumps(best["best_params"], ensure_ascii=False),
                ),
            )

    def train_pipeline(self) -> None:
        try:
            datos = Path(self.datos_path.get())
            quimicos = Path(self.quimicos_path.get())
            output_dir = Path(self.output_dir.get())
            trained_at = dt.datetime.now().isoformat(timespec="seconds")
            run_name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_dir / "training_runs" / run_name
            self.models_dir = run_dir / "models"
            self.models_dir.mkdir(parents=True, exist_ok=True)

            if not datos.exists() or not quimicos.exists():
                raise FileNotFoundError("Selecciona datos.xlsx y quimicos.xlsx validos.")

            log_safe(self.train_log, "Preparando archivos historicos...")
            converted = convert_excels(datos, quimicos, run_dir)
            dataset = standardize_dataset(converted)
            dataset.to_csv(run_dir / "dataset_modelos_app.csv", index=False, encoding="utf-8-sig")

            log_safe(self.train_log, f"Dias preparados: {len(dataset)}")
            metadata = {
                "trained_at": trained_at,
                "dataset_rows": len(dataset),
                "run_dir": str(run_dir),
                "models_dir": str(self.models_dir),
                "source_files": {
                    "datos": str(datos),
                    "quimicos": str(quimicos),
                },
                "chemicals": {},
            }

            for target_key in CHEMICAL_TARGETS:
                log_safe(self.train_log, f"Calculando referencia para {target_key}...")
                chem_model, chem_meta = train_best_model(
                    dataset,
                    CHEMICAL_FEATURES,
                    target_key,
                    filter_outliers=True,
                )
                joblib.dump(chem_model, self.models_dir / f"{target_key}.joblib")
                metadata["chemicals"][target_key] = chem_meta
                best = chem_meta["best"]
                reliability = max(min(best["r2_test"], 1), 0) * 100
                log_safe(
                    self.train_log,
                    f"{target_key} -> fiabilidad {reliability:.0f}% | "
                    f"error medio {best['rmse_test']:.1f} kg/dia",
                )

            with open(self.models_dir / "metadata.json", "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, ensure_ascii=False)

            log_safe(self.train_log, f"Referencia guardada en: {self.models_dir}")
            self.after(0, self.update_stats, metadata, len(dataset))
            self.after(0, self.refresh_model_history, True)
            self.after(0, self._reload_predict_after_training)
            self.after(0, self.finish_training_ui, True, "Referencia actualizada correctamente.")
        except Exception as exc:
            log_safe(self.train_log, f"ERROR: {exc}")
            self.after(0, self.finish_training_ui, False, "No se pudo actualizar la referencia.")
            self.after(0, messagebox.showerror, "Error", str(exc))

    def _reload_predict_after_training(self) -> None:
        # When a new reference is trained, force the predict tab to refresh so
        # the welcome card disappears and the form is enabled.
        if self._current_page_key_resolved == "predict":
            self.build_predict_tab()
        self.history_refresh_needed = True

    # -------------------------------------------------------- Prediction
    def read_common_inputs(self) -> dict:
        fecha = pd.to_datetime(self.pred_fecha.get(), errors="raise")
        return build_prediction_row(
            fecha=fecha,
            caudal=float(self.pred_caudal.get().replace(",", ".")),
            dqo_entrada=float(self.pred_dqo_entrada.get().replace(",", ".")),
            dqo_salida=float(self.pred_dqo_salida.get().replace(",", ".")),
        )

    def predict_chemicals(self) -> None:
        if not self._inputs_valid():
            return
        try:
            row = self.read_common_inputs()
            X = pd.DataFrame([{col: row[col] for col in CHEMICAL_FEATURES}])
            log_safe(self.predict_log, "\nCalculo de dosis:")
            results: dict[str, float] = {}
            reliabilities: list[float] = []
            for target_key in CHEMICAL_TARGETS:
                model_path = self.models_dir / f"{target_key}.joblib"
                if not model_path.exists():
                    raise FileNotFoundError("Primero prepara o selecciona una referencia.")
                model = joblib.load(model_path)
                pred = max(float(model.predict(X.values)[0]), 0)
                results[target_key] = pred
                self.result_vars[target_key].set(f"{pred:.2f}")
                log_safe(self.predict_log, f"  {target_key}: {pred:.3f} kg/dia")
                reliability = self._reliability_for(target_key)
                if reliability is not None:
                    reliabilities.append(reliability)
                    self._set_pill(self.result_pills.get(target_key), self._pill_text(reliability), self._pill_level(reliability))
                else:
                    self._set_pill(self.result_pills.get(target_key), "Sin datos", "muted")
            self.predict_status.set("Dosis calculada correctamente.")
            verdict_text, verdict_color = self._verdict_for(reliabilities)
            self.verdict_var.set(verdict_text)
            self._set_verdict_color(verdict_color)
            self.result_summary_var.set(self._summary_for(results, reliabilities))
            self.play_success_sound()
        except Exception as exc:
            log_safe(self.predict_log, f"ERROR: {exc}")
            self.predict_status.set("No se pudo calcular la dosis.")
            messagebox.showerror("Error", str(exc))

    def _inputs_valid(self) -> bool:
        return all(not v.get() for v in self.input_error_vars.values())

    def _reliability_for(self, target_key: str) -> float | None:
        meta = self.current_metadata.get("chemicals", {}).get(target_key)
        if not meta:
            return None
        best = meta.get("best", {})
        r2 = best.get("r2_test")
        if r2 is None:
            return None
        return max(min(r2, 1), 0) * 100

    @staticmethod
    def _pill_level(reliability: float) -> str:
        if reliability >= 75:
            return "ok"
        if reliability >= 50:
            return "warn"
        return "bad"

    @staticmethod
    def _pill_text(reliability: float) -> str:
        level = FRC40App._pill_level(reliability)
        if level == "ok":
            return f"OK {reliability:.0f}%"
        if level == "warn":
            return f"Revisar {reliability:.0f}%"
        return f"Baja {reliability:.0f}%"

    def _summary_for(self, results: dict[str, float], reliabilities: list[float]) -> str:
        if not results:
            return "Aun no se ha calculado ninguna dosis."
        total = sum(results.values())
        if reliabilities:
            avg = sum(reliabilities) / len(reliabilities)
            level = self._pill_level(avg)
            verdict = {
                "ok": "Resultado fiable, util para operar.",
                "warn": "Resultado orientativo, valida con la planta.",
                "bad": "Fiabilidad baja, usa esta cifra con precaution.",
            }.get(level, "")
        else:
            verdict = "Resultado orientativo, valida con la planta."
        parts = ", ".join(f"{CHEMICAL_DISPLAY[k]}: {v:.2f} kg/dia" for k, v in results.items())
        return f"Total {total:.2f} kg/dia. {parts}. {verdict}"

    def _verdict_for(self, reliabilities: list[float]) -> tuple[str, str]:
        """Return (text, colour_key) describing overall prediction quality."""
        if not reliabilities:
            return ("Calculado, sin datos de fiabilidad.", "muted")
        avg = sum(reliabilities) / len(reliabilities)
        level = self._pill_level(avg)
        if level == "ok":
            return (f"Resultado fiable · {avg:.0f}% medio", "ok")
        if level == "warn":
            return (f"Fiabilidad media · {avg:.0f}% · valida con la planta", "warn")
        return (f"Fiabilidad baja · {avg:.0f}% · usar con precaution", "bad")

    def _set_verdict_color(self, level: str) -> None:
        palette = {
            "ok": (self.palette["ok"], self.palette["ok_bg"]),
            "warn": (self.palette["warn"], self.palette["warn_bg"]),
            "bad": (self.palette["bad"], self.palette["bad_bg"]),
            "muted": (self.palette["text_muted"], self.palette["card_border"]),
        }
        fg, bg = palette.get(level, palette["muted"])
        if hasattr(self, "verdict_dot"):
            self.verdict_dot.configure(fg=fg, bg=self.palette["card"])
        if hasattr(self, "verdict_label"):
            self.verdict_label.configure(fg=fg, bg=self.palette["card"])

    def copy_results_to_clipboard(self) -> None:
        lines = ["Dosis recomendada FRC40:"]
        any_value = False
        for key in CHEMICAL_TARGETS:
            value = self.result_vars[key].get()
            if value == "-- kg/dia":
                continue
            any_value = True
            lines.append(f"- {CHEMICAL_DISPLAY[key]}: {value}")
        if not any_value:
            self.predict_status.set("No hay un calculo reciente para copiar.")
            return
        summary = self.result_summary_var.get()
        if summary:
            lines.append("")
            lines.append(summary)
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        self.update()  # Flush clipboard on Windows.
        self.predict_status.set("Resultado copiado al portapapeles.")

    # -------------------------------------------------------- Sound
    def play_success_sound(self) -> None:
        """Play a short success tone. Windows-only via winsound; no-op elsewhere."""
        if not getattr(sys, "frozen", False) and platform.system() != "Windows":
            return
        try:
            import winsound  # type: ignore[import-not-found]

            threading.Thread(
                target=lambda: winsound.MessageBeep(winsound.MB_ICONASTERISK),
                daemon=True,
            ).start()
        except Exception:
            pass


def main() -> None:
    app = FRC40App()
    app.mainloop()
