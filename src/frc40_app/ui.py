from __future__ import annotations

import datetime as dt
import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import joblib
import pandas as pd

from .config import CHEMICAL_FEATURES, CHEMICAL_TARGETS
from .features import build_prediction_row, standardize_dataset
from .modeling import train_best_model
from .preprocessing import convert_excels
from .utils import log_safe


class FRC40App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FRC40 - Calculo de dosis quimica")
        self.geometry("1080x760")
        self.minsize(860, 560)
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "app_outputs"))
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

        self.configure(bg="#f5f7fb")
        self.setup_style()

        shell = tk.Frame(self, bg="#f5f7fb")
        shell.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(shell, bg="#0b1220", width=250)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        brand = tk.Frame(self.sidebar, bg="#0b1220", padx=22, pady=22)
        brand.pack(fill="x")
        tk.Label(brand, text="FRC40", bg="#0b1220", fg="white", font=("Segoe UI", 24, "bold")).pack(anchor="w")
        tk.Label(
            brand,
            text="Apoyo a dosificacion diaria",
            bg="#0b1220",
            fg="#94a3b8",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 0))

        self.nav_buttons: dict[str, tk.Button] = {}
        self.add_nav_button("predict", "01  Calcular dosis")
        self.add_nav_button("train", "02  Actualizar datos")
        self.add_nav_button("history", "03  Referencias guardadas")
        self.add_nav_button("stats", "04  Control de calidad")

        sidebar_footer = tk.Frame(self.sidebar, bg="#0b1220", padx=22, pady=18)
        sidebar_footer.pack(side="bottom", fill="x")
        tk.Label(
            sidebar_footer,
            text="Introduce caudal y DQO para obtener una dosis orientativa en kg/dia.",
            bg="#0b1220",
            fg="#64748b",
            wraplength=190,
            justify="left",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        self.main_area = tk.Frame(shell, bg="#f5f7fb", padx=22, pady=20)
        self.main_area.pack(side="left", fill="both", expand=True)
        self.main_area.columnconfigure(0, weight=1)
        self.main_area.rowconfigure(1, weight=1)

        self.page_title = tk.StringVar(value="")
        self.page_subtitle = tk.StringVar(value="")
        page_header = tk.Frame(self.main_area, bg="#f5f7fb")
        page_header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        tk.Label(
            page_header,
            textvariable=self.page_title,
            bg="#f5f7fb",
            fg="#102a43",
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            page_header,
            textvariable=self.page_subtitle,
            bg="#f5f7fb",
            fg="#627d98",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 0))

        self.pages = tk.Frame(self.main_area, bg="#f5f7fb")
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

    def show_page(self, key: str) -> None:
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
                button.default_bg = "#1d4ed8"
                button.hover_bg = "#2563eb"
                button.configure(bg="#1d4ed8", fg="white", activebackground="#2563eb")
            else:
                button.default_bg = "#0b1220"
                button.hover_bg = "#111c2f"
                button.configure(bg="#0b1220", fg="#cbd5e1", activebackground="#111c2f")
        pages[key].tkraise()

    def setup_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Header.TFrame", background="#102a43")
        style.configure("HeaderTitle.TLabel", background="#102a43", foreground="white", font=("Segoe UI", 18, "bold"))
        style.configure("HeaderSubtitle.TLabel", background="#102a43", foreground="#bcccdc", font=("Segoe UI", 11))
        style.configure("Card.TFrame", background="white", relief="flat")
        style.configure("Page.TFrame", background="#f5f7fb")
        style.configure("CardTitle.TLabel", background="white", foreground="#102a43", font=("Segoe UI", 12, "bold"))
        style.configure("Muted.TLabel", background="white", foreground="#627d98")
        style.configure("Result.TLabel", background="white", foreground="#102a43", font=("Segoe UI", 13, "bold"))
        style.configure("Status.TLabel", background="#f5f7fb", foreground="#486581")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7))
        style.configure("TButton", padding=(10, 6))
        style.configure("TEntry", padding=(6, 4))
        style.configure("TNotebook", background="#f5f7fb", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "white")], foreground=[("selected", "#102a43")])
        style.configure("Treeview", rowheight=30, font=("Segoe UI", 10), fieldbackground="white", background="white")
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
            "sidebar": {
                "bg": "#0b1220",
                "hover": "#111c2f",
                "fg": "#cbd5e1",
                "active": "#111c2f",
                "disabled": "#0b1220",
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
        card = tk.Frame(parent, bg="white", highlightthickness=1, highlightbackground="#d9e2ec")
        tk.Frame(card, bg=accent, width=5).pack(side="left", fill="y")
        body = tk.Frame(card, bg="white", padx=14, pady=12)
        body.pack(side="left", fill="both", expand=True)
        tk.Label(body, text=title, bg="white", fg="#627d98", font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(body, textvariable=value_var, bg="white", fg="#102a43", font=("Segoe UI", 16, "bold")).pack(
            anchor="w", pady=(4, 0)
        )
        return card

    def step_card(self, parent, number: str, title: str, text: str, accent: str) -> tk.Frame:
        card = tk.Frame(parent, bg="white", highlightthickness=1, highlightbackground="#d9e2ec")
        top = tk.Frame(card, bg=accent, height=4)
        top.pack(fill="x")
        body = tk.Frame(card, bg="white", padx=14, pady=12)
        body.pack(fill="both", expand=True)
        tk.Label(body, text=number, bg="white", fg=accent, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(body, text=title, bg="white", fg="#102a43", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(3, 0))
        tk.Label(
            body,
            text=text,
            bg="white",
            fg="#627d98",
            font=("Segoe UI", 9),
            wraplength=220,
            justify="left",
        ).pack(anchor="w", pady=(5, 0))
        return card

    def build_train_tab(self) -> None:
        frame = self.train_tab
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
            ttk.Entry(form, textvariable=var, width=95).grid(row=row, column=1, sticky="ew", padx=6, pady=7)
            self.make_button(form, "Seleccionar", command, variant="secondary", width=13).grid(
                row=row, column=2, padx=(8, 0), pady=7
            )

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

        self.train_log = tk.Text(frame, height=30, state="disabled")
        self.train_log.configure(
            bg="#0b1220",
            fg="#d9e2ec",
            insertbackground="#d9e2ec",
            relief="flat",
            padx=12,
            pady=10,
            font=("Consolas", 10),
        )
        self.train_log.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=14, pady=(6, 14))
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

    def build_history_tab(self) -> None:
        frame = self.history_tab
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
        y_scroll = ttk.Scrollbar(table_card, orient="vertical", command=self.history_tree.yview)
        x_scroll = ttk.Scrollbar(table_card, orient="horizontal", command=self.history_tree.xview)
        self.history_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        y_scroll.grid(row=1, column=1, sticky="ns", pady=(10, 0))
        x_scroll.grid(row=2, column=0, sticky="ew")

        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 14))
        actions.columnconfigure(1, weight=1)
        self.make_button(actions, "Usar referencia seleccionada", self.use_selected_history_model, variant="primary").grid(
            row=0, column=0, sticky="w"
        )
        self.make_button(actions, "Actualizar lista", self.refresh_model_history, variant="secondary").grid(
            row=0, column=1, sticky="w", padx=(10, 0)
        )
        ttk.Label(actions, textvariable=self.active_model_text, style="Status.TLabel", wraplength=650).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        ttk.Label(actions, textvariable=self.selected_reference_path, style="Status.TLabel", wraplength=650).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )

    def build_stats_tab(self) -> None:
        frame = self.stats_tab
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

        self.best_model_vars = {
            "policloruro_aluminio": tk.StringVar(value="--"),
            "coagulante_organico": tk.StringVar(value="--"),
            "floculante_cationico": tk.StringVar(value="--"),
        }
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

    def build_predict_tab(self) -> None:
        frame = self.predict_tab
        frame.configure(style="Page.TFrame")
        self.pred_fecha = tk.StringVar(value=dt.date.today().isoformat())
        self.pred_caudal = tk.StringVar(value="50")
        self.pred_dqo_entrada = tk.StringVar(value="22000")
        self.pred_dqo_salida = tk.StringVar(value="900")

        intro = ttk.Frame(frame, style="Card.TFrame")
        intro.grid(row=0, column=0, columnspan=3, sticky="ew", padx=14, pady=(14, 6))
        ttk.Label(intro, text="Calculo diario de dosis", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            intro,
            text="Introduce los datos del dia. El resultado es una ayuda operativa, no una orden automatica.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        ttk.Label(intro, textvariable=self.active_model_text, style="Muted.TLabel").pack(anchor="w", pady=(6, 0))

        form = ttk.Frame(frame, style="Card.TFrame")
        form.grid(row=1, column=0, sticky="nw", padx=14, pady=10)
        fields = [
            ("Fecha (YYYY-MM-DD)", self.pred_fecha),
            ("Caudal previsto (m3/dia)", self.pred_caudal),
            ("DQO entrada (mg/L)", self.pred_dqo_entrada),
            ("DQO salida deseada (mg/L)", self.pred_dqo_salida),
        ]
        for row, (label, var) in enumerate(fields):
            ttk.Label(form, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 14), pady=8)
            ttk.Entry(form, textvariable=var, width=30).grid(row=row, column=1, sticky="w", pady=8)

        result_card = ttk.Frame(frame, style="Card.TFrame")
        result_card.grid(row=1, column=1, columnspan=2, sticky="nsew", padx=14, pady=10)
        ttk.Label(result_card, text="Dosis orientativa", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        self.result_vars = {
            "policloruro_aluminio": tk.StringVar(value="-- kg/dia"),
            "coagulante_organico": tk.StringVar(value="-- kg/dia"),
            "floculante_cationico": tk.StringVar(value="-- kg/dia"),
        }
        result_labels = [
            ("Policloruro aluminio", "policloruro_aluminio"),
            ("Coagulante organico", "coagulante_organico"),
            ("Floculante cationico", "floculante_cationico"),
        ]
        accents = {"policloruro_aluminio": "#14b8a6", "coagulante_organico": "#f59e0b", "floculante_cationico": "#8b5cf6"}
        for row, (label, key) in enumerate(result_labels, start=1):
            card = self.metric_card(result_card, label, self.result_vars[key], accents[key])
            card.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        result_card.columnconfigure(1, weight=1)

        self.predict_button = self.make_button(
            frame,
            "Calcular dosis",
            self.predict_chemicals,
            variant="primary",
        )
        self.predict_button.grid(row=2, column=0, columnspan=3, sticky="ew", padx=14, pady=(4, 8))
        ttk.Label(frame, textvariable=self.predict_status, style="Status.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w", padx=14, pady=(0, 4)
        )

        self.predict_log = tk.Text(frame, height=24, state="disabled")
        self.predict_log.configure(
            bg="#0b1220",
            fg="#d9e2ec",
            insertbackground="#d9e2ec",
            relief="flat",
            padx=12,
            pady=10,
            font=("Consolas", 10),
        )
        self.predict_log.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=14, pady=(8, 14))
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

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
            self.refresh_model_history(select_latest=False)

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
        base_dirs = [Path(self.output_dir.get())]
        app_default_dir = Path(__file__).resolve().parents[2] / "app_outputs"
        if app_default_dir not in base_dirs:
            base_dirs.append(app_default_dir)

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
        messagebox.showinfo("Referencia cargada", "La referencia seleccionada ya esta activa para calcular dosis.")

    def load_history_model(self, record: dict) -> None:
        self.models_dir = record["models_dir"]
        metadata = record["metadata"]
        self.update_stats(metadata, metadata.get("dataset_rows") or record["dataset_rows"])
        self.active_model_text.set(f"Referencia activa: {record['trained_at'].replace('T', ' ')}")
        self.selected_reference_path.set(f"Carpeta: {record['models_dir']}")
        self.predict_status.set("Referencia cargada. Ya puedes calcular dosis.")

    def start_training(self) -> None:
        if self.training:
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

    def update_stats(self, metadata: dict, dataset_rows: int) -> None:
        self.dataset_rows.set(str(dataset_rows))
        self.models_path_text.set(f"Referencia guardada en: {self.models_dir}")
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        labels = {
            "policloruro_aluminio": "Policloruro aluminio",
            "coagulante_organico": "Coagulante organico",
            "floculante_cationico": "Floculante cationico",
        }
        for target_key, label in labels.items():
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
            self.after(0, self.finish_training_ui, True, "Referencia actualizada correctamente.")
            self.after(0, messagebox.showinfo, "Referencia actualizada", "Los datos han quedado listos para calcular dosis.")
        except Exception as exc:
            log_safe(self.train_log, f"ERROR: {exc}")
            self.after(0, self.finish_training_ui, False, "No se pudo actualizar la referencia.")
            self.after(0, messagebox.showerror, "Error", str(exc))

    def read_common_inputs(self) -> dict:
        fecha = pd.to_datetime(self.pred_fecha.get(), errors="raise")
        return build_prediction_row(
            fecha=fecha,
            caudal=float(self.pred_caudal.get()),
            dqo_entrada=float(self.pred_dqo_entrada.get()),
            dqo_salida=float(self.pred_dqo_salida.get()),
        )

    def predict_chemicals(self) -> None:
        try:
            row = self.read_common_inputs()
            X = pd.DataFrame([{col: row[col] for col in CHEMICAL_FEATURES}])
            log_safe(self.predict_log, "\nCalculo de dosis:")
            for target_key in CHEMICAL_TARGETS:
                model_path = self.models_dir / f"{target_key}.joblib"
                if not model_path.exists():
                    raise FileNotFoundError("Primero prepara o selecciona una referencia.")
                model = joblib.load(model_path)
                pred = max(float(model.predict(X.values)[0]), 0)
                self.result_vars[target_key].set(f"{pred:.3f} kg/dia")
                log_safe(self.predict_log, f"  {target_key}: {pred:.3f} kg/dia")
            self.predict_status.set("Dosis calculada correctamente.")
        except Exception as exc:
            log_safe(self.predict_log, f"ERROR: {exc}")
            self.predict_status.set("No se pudo calcular la dosis.")
            messagebox.showerror("Error", str(exc))


def main() -> None:
    app = FRC40App()
    app.mainloop()
