import math
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import fields, is_dataclass
from functools import partial
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.patches import Rectangle as PlotRectangle
from matplotlib.path import Path as PlotPath

from .config_io import (
    append_history,
    load_configuration,
    load_history,
    save_configuration,
)
from .export import save_hdf5
from .materials import Material
from .models import Aperture, Detector, LensStack, RayBundle
from .profiles import (
    gaussian_fit_fwhm,
    interpolated_fwhm,
    profile_crop_slice,
    statistics,
    statistics2d,
    weighted_profile,
    weighted_profile2d,
)
from .simulator import Simulator
from .sources import Source
from .wavefront import simulate_wavefront_adaptive

BG = "#f3f5f8"
PANEL = "#ffffff"
INK = "#172033"
MUTED = "#687386"
BLUE = "#2367d1"
RED = "#c43d4b"


class CRLSuiteApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CRL Suite")
        self.geometry("1800x1000")
        self.minsize(1120, 720)
        self.configure(background=BG)
        self.pending = None
        self.drag_detector = False
        self.result: Any = None
        self.stack_rows = []
        self.aperture_rows = []
        self.optic_rows = []
        self.detector_rows = []
        self.detector_results: list[Any] = []
        self.selected_detector = 0
        self.multi_running = False
        self._calculation_executor = ThreadPoolExecutor(max_workers=1)
        self._calculation_future: Future[list[Any]] | None = None
        self._calculation_job_id = 0
        self.vars = {}
        self.input_widgets = {}
        self.field_labels = {}
        self.history = load_history()
        self.last_mode = "1D"
        self.applying_configuration = False
        self._style()
        self._layout()
        try:
            if not self.history:
                raise ValueError
            self._apply_configuration(self.history[0]["configuration"])
        except Exception:
            self.history = []
            self.add_stack()
            self.after(200, self.preview)

    def _style(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Page.TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL, relief="flat")
        style.configure("Card.TLabel", background=PANEL, foreground=INK)
        style.configure(
            "Heading.TLabel",
            background=BG,
            foreground=INK,
            font=("Segoe UI Semibold", 18),
        )
        style.configure(
            "Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9)
        )
        style.configure(
            "Section.TLabel",
            background=PANEL,
            foreground=INK,
            font=("Segoe UI Semibold", 11),
        )
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("Error.TLabel", background=PANEL, foreground=RED)
        style.configure(
            "Accent.TButton", background=BLUE, foreground="white", padding=(14, 8)
        )
        style.map("Accent.TButton", background=[("active", "#174fa9")])
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=4)

    def _layout(self):
        header = ttk.Frame(self, style="Page.TFrame", padding=(20, 14, 20, 10))
        header.pack(fill="x")
        ttk.Label(header, text="CRL Suite", style="Heading.TLabel").pack(side="left")
        ttk.Label(
            header, text="Compound refractive lens simulation", style="Sub.TLabel"
        ).pack(side="left", padx=14, pady=(7, 0))
        self.status = ttk.Label(
            header,
            text="Ready",
            style="Sub.TLabel",
            anchor="w",
            width=18,
        )
        self.status.pack(side="left", padx=(12, 0), pady=(7, 0))
        self.calculate_button = ttk.Button(
            header,
            text="Calculate",
            style="Accent.TButton",
            command=lambda: self.calculate(False),
        )
        self.calculate_button.pack(side="right")
        self.cancel_button = ttk.Button(
            header,
            text="Cancel",
            command=self.cancel_calculation,
            state="disabled",
        )
        self.cancel_button.pack(side="right", padx=(0, 5))
        ttk.Button(header, text="Save", command=self.save_configuration).pack(
            side="right", padx=(0, 5)
        )
        ttk.Button(header, text="Generate Report", command=self.generate_report).pack(
            side="right", padx=(0, 5)
        )
        ttk.Button(header, text="Save Results", command=self.save_results).pack(
            side="right", padx=(0, 5)
        )
        ttk.Button(header, text="Load", command=self.load_configuration).pack(
            side="right", padx=(0, 5)
        )
        ttk.Button(header, text="Restore", command=self.restore_history).pack(
            side="right", padx=(0, 5)
        )
        self.history_var = tk.StringVar()
        self.history_combo = ttk.Combobox(
            header, textvariable=self.history_var, state="readonly", width=24
        )
        self.history_combo.pack(side="right", padx=(0, 5))
        self._refresh_history()
        self.control_body = ttk.Frame(self, style="Page.TFrame")
        self.control_body.pack(fill="x", padx=16, pady=(0, 10))
        self.card_column = 0
        self._source_card()
        self._optics_card()
        self._detector_card()
        output = ttk.Panedwindow(self, orient="horizontal")
        output.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        schematic_panel = ttk.Frame(output, style="Card.TFrame")
        output.add(schematic_panel, weight=3)
        result_panel = ttk.Panedwindow(output, orient="horizontal")
        output.add(result_panel, weight=2)
        profile_panel = ttk.Frame(result_panel, style="Card.TFrame")
        result_panel.add(profile_panel, weight=4)
        statistics_panel = ttk.Frame(result_panel, style="Card.TFrame", padding=8)
        result_panel.add(statistics_panel, weight=2)
        self.figure = Figure(
            figsize=(7, 6), dpi=100, facecolor=PANEL, constrained_layout=True
        )
        scheme_grid = self.figure.add_gridspec(2, 1, hspace=0.28)
        self.ax_scheme = self.figure.add_subplot(scheme_grid[0, 0])
        self.ax_scheme_y = self.figure.add_subplot(
            scheme_grid[1, 0], sharex=self.ax_scheme
        )
        schematic_toolbar = ttk.Frame(schematic_panel, style="Card.TFrame")
        schematic_toolbar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(
            schematic_toolbar,
            text="Save layout PNG",
            command=self.save_schematic_png,
        ).pack(side="right")
        self.canvas = FigureCanvasTkAgg(self.figure, master=schematic_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._plot_press)
        self.canvas.mpl_connect("motion_notify_event", self._plot_motion)
        self.canvas.mpl_connect("button_release_event", self._plot_release)
        self.profile_figure = Figure(
            figsize=(5, 3), dpi=100, facecolor=PANEL, constrained_layout=True
        )
        profile_grid = self.profile_figure.add_gridspec(
            2,
            2,
            width_ratios=(4, 1),
            height_ratios=(4, 1),
            hspace=0.05,
            wspace=0.05,
        )
        self.ax_profile = self.profile_figure.add_subplot(profile_grid[0, 0])
        self.ax_profile_right = self.profile_figure.add_subplot(
            profile_grid[0, 1], sharey=self.ax_profile
        )
        self.ax_profile_bottom = self.profile_figure.add_subplot(
            profile_grid[1, 0], sharex=self.ax_profile
        )
        self.ax_profile_corner = self.profile_figure.add_subplot(profile_grid[1, 1])
        self.ax_profile_corner.axis("off")
        profile_toolbar = ttk.Frame(profile_panel, style="Card.TFrame")
        profile_toolbar.pack(fill="x", padx=6, pady=(6, 0))
        self.detector_selector = ttk.Frame(profile_toolbar, style="Card.TFrame")
        self.detector_selector.pack(side="left")
        ttk.Button(
            profile_toolbar,
            text="Save profile PNG",
            command=self.save_profile_png,
        ).pack(side="right")
        self.profile_canvas = FigureCanvasTkAgg(
            self.profile_figure, master=profile_panel
        )
        self.profile_canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Label(statistics_panel, text="Statistics", style="Section.TLabel").pack(
            anchor="w", pady=(0, 4)
        )
        text_frame = ttk.Frame(statistics_panel, style="Card.TFrame")
        text_frame.pack(fill="both", expand=True)
        statistics_scroll = ttk.Scrollbar(text_frame, orient="vertical")
        statistics_scroll.pack(side="right", fill="y")
        self.statistics_text = tk.Text(
            text_frame,
            wrap="word",
            relief="flat",
            borderwidth=0,
            background=PANEL,
            foreground=INK,
            font=("Segoe UI", 9),
            padx=3,
            pady=3,
            yscrollcommand=statistics_scroll.set,
            state="disabled",
        )
        self.statistics_text.pack(side="left", fill="both", expand=True)
        statistics_scroll.configure(command=self.statistics_text.yview)

    def _card(self, title, weight=1):
        outer = ttk.Frame(self.control_body, style="Card.TFrame", padding=14)
        outer.grid(
            row=0,
            column=self.card_column,
            sticky="nsew",
            padx=(0, 10) if self.card_column < 2 else 0,
        )
        self.control_body.columnconfigure(self.card_column, weight=weight)
        self.card_column += 1
        ttk.Label(outer, text=title, style="Section.TLabel").pack(
            anchor="w", pady=(0, 9)
        )
        return outer

    def _var(self, name, value):
        var = tk.StringVar(value=str(value))
        var.trace_add("write", self._changed)
        self.vars[name] = var
        return var

    def _field(
        self, parent, label, name, default, row, column=0, width=12, options=None
    ):
        box = ttk.Frame(parent, style="Card.TFrame")
        box.grid(row=row, column=column, sticky="ew", padx=(0, 8), pady=3)
        field_label = ttk.Label(box, text=label, style="Muted.TLabel")
        field_label.pack(anchor="w")
        var = self._var(name, default)
        combo: tk.Widget
        if options:
            combo = ttk.Combobox(
                box, textvariable=var, values=options, state="readonly", width=width
            )
        else:
            combo = ttk.Entry(box, textvariable=var, width=width)
        combo.pack(fill="x", pady=(2, 0))
        self.input_widgets[name] = combo
        self.field_labels[name] = field_label
        parent.columnconfigure(column, weight=1)
        return var

    def _source_card(self):
        card = self._card("Source and sampling")
        form = ttk.Frame(card, style="Card.TFrame")
        form.pack(fill="x")
        self._field(form, "Mode", "mode", "1D", 0, 0, options=("1D", "2D"))
        self._field(form, "Energy (keV)", "energy", "12.0", 0, 1)
        self._field(
            form,
            "Engine",
            "engine",
            "ray tracing",
            1,
            0,
            options=("ray tracing", "wavefront"),
        )
        self._field(
            form,
            "Spatial kind",
            "spatial_kind",
            "gaussian",
            1,
            1,
            options=("gaussian", "uniform", "point"),
        )
        self._field(form, "Size X (µm)", "size_x", "0.10", 2, 0)
        self._field(form, "Size Y (µm)", "size_y", "0.10", 2, 1)
        self._field(
            form,
            "Spatial boundary ± half-width (µm)",
            "spatial_bound",
            "0.30",
            3,
            0,
        )
        self._field(
            form,
            "Divergence kind",
            "divergence_kind",
            "gaussian",
            3,
            1,
            options=("gaussian", "uniform", "collimated"),
        )
        self._field(form, "Divergence X (µrad)", "div_x", "0.05", 4, 0)
        self._field(form, "Divergence Y (µrad)", "div_y", "0.05", 4, 1)
        self._field(
            form,
            "Angular boundary ± half-width (µrad)",
            "div_bound",
            "0.15",
            5,
            0,
        )
        self._field(form, "Source offset (m)", "source_z", "0.0", 5, 1)
        self._field(form, "Ray count", "ray_count", "20000", 6, 0)
        self._field(form, "Seed", "seed", "1", 6, 1)
        self._field(form, "Wave points", "wave_points", "65536", 7, 0)
        self._field(form, "Wave window (mm)", "wave_window", "2.0", 7, 1)
        self._update_source_field_labels()

    def _optics_card(self):
        card = self._card("Optics", weight=4)
        stack_canvas = tk.Canvas(
            card,
            height=220,
            highlightthickness=0,
            background=PANEL,
        )
        stack_scroll = ttk.Scrollbar(
            card, orient="horizontal", command=stack_canvas.xview
        )
        self.stack_holder = ttk.Frame(stack_canvas, style="Card.TFrame")
        self.aperture_holder = self.stack_holder
        self.stack_window = stack_canvas.create_window(
            (0, 0), window=self.stack_holder, anchor="nw"
        )
        self.stack_holder.bind(
            "<Configure>",
            lambda event: stack_canvas.configure(scrollregion=stack_canvas.bbox("all")),
        )
        stack_canvas.configure(xscrollcommand=stack_scroll.set)
        stack_canvas.pack(fill="both", expand=True)
        stack_scroll.pack(fill="x", pady=(3, 0))
        buttons = ttk.Frame(card, style="Card.TFrame")
        buttons.pack(fill="x", pady=(5, 0))
        ttk.Button(buttons, text="＋ Add CRL", command=self.add_stack).pack(side="left")
        ttk.Button(buttons, text="＋ Add aperture", command=self.add_aperture).pack(
            side="left", padx=(5, 0)
        )

    def add_aperture(self, data=None):
        values = data or {
            "location": "1.0",
            "shape": "rectangular",
            "size_x": "1.0",
            "size_y": "1.0",
            "offset_x": "0.0",
            "offset_y": "0.0",
            "enabled": "true",
        }
        values.setdefault("enabled", "true")
        row = {key: tk.StringVar(value=value) for key, value in values.items()}
        for variable in row.values():
            variable.trace_add("write", self._changed)
        self.aperture_rows.append(row)
        self.optic_rows.append(("aperture", row))
        self._render_optics()
        self.schedule_preview()

    def _render_aperture_card(self, index, row, column):
        frame = ttk.Frame(self.aperture_holder, style="Page.TFrame", padding=6)
        frame.grid(row=0, column=column, sticky="ns", padx=(0, 5), pady=2)
        top = ttk.Frame(frame, style="Page.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=f"Aperture {index + 1}", style="Sub.TLabel").pack(
            side="left"
        )
        for text, command in (
            ("×", partial(self.remove_aperture, index)),
            ("↓", partial(self.move_optic, column, 1)),
            ("↑", partial(self.move_optic, column, -1)),
        ):
            ttk.Button(top, text=text, width=3, command=command).pack(side="right")
        enabled = row["enabled"].get().lower() == "true"
        ttk.Button(
            top,
            text="Hide" if enabled else "Show",
            width=5,
            command=partial(self.toggle_aperture, index),
        ).pack(side="right", padx=2)
        form = ttk.Frame(frame, style="Page.TFrame")
        form.pack(fill="x")
        fieldspec = (
            ("Location (m)", "location", None),
            ("Shape", "shape", ("rectangular", "elliptical")),
            ("X size (mm)", "size_x", None),
            ("Y size (mm)", "size_y", None),
            ("X offset (mm)", "offset_x", None),
            ("Y offset (mm)", "offset_y", None),
        )
        for position, (label, key, options) in enumerate(fieldspec):
            box = ttk.Frame(form, style="Page.TFrame")
            box.grid(row=position, column=0, sticky="ew", pady=1)
            ttk.Label(box, text=label, style="Sub.TLabel").pack(anchor="w")
            control = (
                ttk.Combobox(
                    box, textvariable=row[key], values=options, state="readonly"
                )
                if options
                else ttk.Entry(box, textvariable=row[key])
            )
            control.pack(fill="x")
            if key in ("size_y", "offset_y") and self.vars["mode"].get() == "1D":
                control.configure(state="disabled")

    def toggle_aperture(self, index):
        row = self.aperture_rows[index]
        enabled = row["enabled"].get().lower() == "true"
        row["enabled"].set("false" if enabled else "true")
        self._render_optics()

    def remove_aperture(self, index):
        row = self.aperture_rows.pop(index)
        self.optic_rows = [
            item
            for item in self.optic_rows
            if not (item[0] == "aperture" and item[1] is row)
        ]
        self._render_optics()
        self.schedule_preview()

    def _detector_card(self):
        card = self._card("Detectors")
        self.detector_holder = ttk.Frame(card, style="Card.TFrame")
        self.detector_holder.pack(fill="x")
        self.add_detector({"location": "10.0", "bins_x": "256", "bins_y": "256"})
        ttk.Button(card, text="＋ Add detector", command=self.add_detector).pack(
            anchor="w", pady=(5, 0)
        )
        ttk.Label(
            card,
            text="Drag the red detector bar in the schematic",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(8, 0))
        self.error_label = ttk.Label(
            card, text="", style="Error.TLabel", wraplength=370
        )
        self.error_label.pack(fill="x", pady=(8, 0))

    def add_detector(self, data=None):
        values = data or {"location": "10.0", "bins_x": "256", "bins_y": "256"}
        row = {key: tk.StringVar(value=str(value)) for key, value in values.items()}
        for variable in row.values():
            variable.trace_add("write", self._changed)
        self.detector_rows.append(row)
        self._invalidate_results()
        self._render_detectors()
        self._refresh_detector_selector()
        self.schedule_preview()

    def _render_detectors(self):
        for child in self.detector_holder.winfo_children():
            child.destroy()
        for index, row in enumerate(self.detector_rows):
            frame = ttk.Frame(self.detector_holder, style="Page.TFrame", padding=5)
            frame.pack(fill="x", pady=2)
            top = ttk.Frame(frame, style="Page.TFrame")
            top.pack(fill="x")
            ttk.Label(top, text=f"Detector {index + 1}", style="Sub.TLabel").pack(
                side="left"
            )
            if len(self.detector_rows) > 1:
                ttk.Button(
                    top,
                    text="×",
                    width=3,
                    command=partial(self.remove_detector, index),
                ).pack(side="right")
            for label, key in (
                ("Location (m)", "location"),
                ("Bins X", "bins_x"),
                ("Bins Y", "bins_y"),
            ):
                ttk.Label(frame, text=label, style="Sub.TLabel").pack(anchor="w")
                entry = ttk.Entry(frame, textvariable=row[key], width=12)
                entry.pack(fill="x")
                if (
                    key == "bins_y"
                    and self.vars.get("mode", tk.StringVar()).get() == "1D"
                ):
                    entry.configure(state="disabled")

    def remove_detector(self, index):
        self.detector_rows.pop(index)
        self.selected_detector = max(
            0, min(self.selected_detector, len(self.detector_rows) - 1)
        )
        self._invalidate_results()
        self._render_detectors()
        self._refresh_detector_selector()
        self.schedule_preview()

    def _detector_values(self):
        values = []
        for index, row in enumerate(self.detector_rows):
            try:
                item = {
                    "location": float(row["location"].get()),
                    "bins_x": int(row["bins_x"].get()),
                    "bins_y": int(row["bins_y"].get()),
                }
                if item["bins_x"] <= 0 or item["bins_y"] <= 0:
                    raise ValueError
                values.append(item)
            except ValueError as exc:
                raise ValueError(
                    f"Detector {index + 1} contains an invalid value"
                ) from exc
        return values

    def add_stack(self, data=None, index=None):
        values = data or {
            "location": "2.0",
            "material": "Be",
            "shape": "parabolic",
            "count": "10",
            "radius": "0.05",
            "aperture": "0.5",
            "apex": "0.03",
            "pitch": "0.08",
            "axis": "2D",
            "enabled": "true",
        }
        values.setdefault("enabled", "true")
        row = {key: tk.StringVar(value=value) for key, value in values.items()}
        for var in row.values():
            var.trace_add("write", self._changed)
        if index is None:
            self.stack_rows.append(row)
        else:
            self.stack_rows.insert(index, row)
        self.optic_rows.append(("stack", row))
        self._render_optics()
        self.schedule_preview()

    def _render_optics(self):
        for child in self.stack_holder.winfo_children():
            child.destroy()
        for column, (kind, row) in enumerate(self.optic_rows):
            if kind == "stack":
                self._render_stack_card(self.stack_rows.index(row), row, column)
            else:
                self._render_aperture_card(self.aperture_rows.index(row), row, column)

    def _render_stack_card(self, index, row, column):
        frame = ttk.Frame(self.stack_holder, style="Page.TFrame", padding=8)
        frame.grid(row=0, column=column, sticky="ns", padx=(0, 5), pady=2)
        top = ttk.Frame(frame, style="Page.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=f"CRL{index + 1}", style="Sub.TLabel").pack(side="left")
        enabled = row["enabled"].get().lower() == "true"
        ttk.Button(
            top,
            text="Hide" if enabled else "Show",
            width=5,
            command=partial(self.toggle_stack, index),
        ).pack(side="right", padx=1)
        for text, command in (
            ("↑", partial(self.move_optic, column, -1)),
            ("↓", partial(self.move_optic, column, 1)),
            ("⧉", lambda i=index: self.duplicate_stack(i)),
            ("×", lambda i=index: self.remove_stack(i)),
        ):
            ttk.Button(top, text=text, width=3, command=command).pack(
                side="right", padx=1
            )
        form = ttk.Frame(frame, style="Page.TFrame")
        form.pack(fill="x")
        if not enabled:
            ttk.Label(form, text="Excluded from calculation", style="Sub.TLabel").grid(
                row=0, column=0, columnspan=3, sticky="w", padx=3
            )
        fieldspec = (
            ("Location (m)", "location", None),
            ("Focus axis", "axis", ("2D", "x", "y")),
            (
                "Material",
                "material",
                ("Be", "diamond", "Al", "Si", "Ni", "IP-Dip", "IP-S"),
            ),
            ("Shape", "shape", ("parabolic", "spherical")),
            ("Lens count", "count", None),
            ("R (mm)", "radius", None),
            ("Aperture (mm)", "aperture", None),
            ("Apex (mm)", "apex", None),
            ("Pitch (mm)", "pitch", None),
        )
        for pos, (label, key, options) in enumerate(fieldspec):
            box = ttk.Frame(form, style="Page.TFrame")
            box.grid(row=1 + pos // 2, column=pos % 2, sticky="ew", padx=3, pady=3)
            ttk.Label(box, text=label, style="Sub.TLabel").pack(anchor="w")
            control: tk.Widget
            if options:
                control = ttk.Combobox(
                    box,
                    textvariable=row[key],
                    values=options,
                    state="readonly",
                    width=10,
                )
            else:
                control = ttk.Entry(box, textvariable=row[key], width=10)
            control.pack(fill="x")
            if key == "axis" and self.vars["mode"].get() == "1D":
                control.configure(state="disabled")
            form.columnconfigure(pos % 2, weight=1)

    def move_optic(self, index, delta):
        target = index + delta
        if 0 <= target < len(self.optic_rows):
            self.optic_rows[index], self.optic_rows[target] = (
                self.optic_rows[target],
                self.optic_rows[index],
            )
            self._render_optics()
            self.schedule_preview()

    def toggle_stack(self, index):
        row = self.stack_rows[index]
        enabled = row["enabled"].get().lower() == "true"
        row["enabled"].set("false" if enabled else "true")
        self._render_optics()
        self.schedule_preview()

    def duplicate_stack(self, index):
        source = self.stack_rows[index]
        self.add_stack({key: value.get() for key, value in source.items()}, index + 1)
        duplicate = self.stack_rows[index + 1]
        self.optic_rows.remove(("stack", duplicate))
        optic_index = next(
            i for i, item in enumerate(self.optic_rows) if item[1] is source
        )
        self.optic_rows.insert(optic_index + 1, ("stack", duplicate))
        self._render_optics()

    def remove_stack(self, index):
        if len(self.stack_rows) > 1:
            row = self.stack_rows.pop(index)
            self.optic_rows = [item for item in self.optic_rows if item[1] is not row]
            self._render_optics()
            self.schedule_preview()

    def _changed(self, *args):
        if self.applying_configuration:
            return
        if self._calculation_future is not None:
            self.cancel_calculation()
        self._invalidate_results()
        mode_variable = self.vars.get("mode")
        points_variable = self.vars.get("wave_points")
        if mode_variable is not None and points_variable is not None:
            mode = mode_variable.get()
            self._update_source_field_labels()
            if mode != self.last_mode:
                self.last_mode = mode
                is_1d = mode == "1D"
                for name in ("size_y", "div_y"):
                    widget = self.input_widgets.get(name)
                    if widget is not None:
                        widget.configure(state="disabled" if is_1d else "normal")
                if hasattr(self, "stack_holder"):
                    self._render_optics()
                if hasattr(self, "detector_holder"):
                    self._render_detectors()
        self.schedule_preview()

    def _invalidate_results(self):
        self.result = None
        self.detector_results = []
        self.selected_detector = max(
            0, min(self.selected_detector, len(self.detector_rows) - 1)
        )

    def _update_source_field_labels(self):
        if not self.field_labels:
            return
        spatial_kind = self.vars.get("spatial_kind")
        divergence_kind = self.vars.get("divergence_kind")
        spatial_prefix = (
            "RMS size"
            if spatial_kind is not None and spatial_kind.get() == "gaussian"
            else "Half size"
            if spatial_kind is not None and spatial_kind.get() == "uniform"
            else "Size"
        )
        divergence_prefix = (
            "RMS divergence"
            if divergence_kind is not None and divergence_kind.get() == "gaussian"
            else "Half divergence"
            if divergence_kind is not None and divergence_kind.get() == "uniform"
            else "Divergence"
        )
        point_source = spatial_kind is not None and spatial_kind.get() == "point"
        collimated = (
            divergence_kind is not None and divergence_kind.get() == "collimated"
        )
        wavefront = (
            self.vars.get("engine") is not None
            and self.vars["engine"].get() == "wavefront"
        )
        for name in ("ray_count", "seed"):
            widget = self.input_widgets.get(name)
            if widget is not None:
                widget.configure(state="disabled" if wavefront else "normal")
        for name in ("wave_points", "wave_window"):
            widget = self.input_widgets.get(name)
            if widget is not None:
                widget.configure(state="normal" if wavefront else "disabled")
        one_dimensional = (
            self.vars.get("mode") is not None and self.vars["mode"].get() == "1D"
        )
        for axis in ("x", "y"):
            size_label = self.field_labels.get(f"size_{axis}")
            divergence_label = self.field_labels.get(f"div_{axis}")
            size_widget = self.input_widgets.get(f"size_{axis}")
            divergence_widget = self.input_widgets.get(f"div_{axis}")
            if size_label is not None:
                size_label.configure(text=f"{spatial_prefix} {axis.upper()} (µm)")
            if size_widget is not None:
                size_widget.configure(
                    state="disabled"
                    if point_source or one_dimensional and axis == "y"
                    else "normal"
                )
            if divergence_label is not None:
                divergence_label.configure(
                    text=f"{divergence_prefix} {axis.upper()} (µrad)"
                )
            if divergence_widget is not None:
                divergence_widget.configure(
                    state="disabled"
                    if collimated or one_dimensional and axis == "y"
                    else "normal"
                )

    def schedule_preview(self):
        if self.applying_configuration:
            return
        if self.pending is not None:
            self.after_cancel(self.pending)
        self.pending = self.after(450, self.preview)

    def _number(self, name, positive=False, integer=False):
        raw = self.vars[name].get().strip()
        try:
            value = int(raw) if integer else float(raw)
        except ValueError as exc:
            label = name.replace("_", " ").title()
            raise ValueError(f"{label} must be a number") from exc
        if not math.isfinite(value) or positive and value <= 0:
            raise ValueError(
                "{} must be a positive number".format(name.replace("_", " "))
            )
        return value

    def _stack_values(self):
        output = []
        for index, row in enumerate(self.stack_rows):
            try:
                item = {key: value.get().strip() for key, value in row.items()}
                for key in ("location", "radius", "aperture", "apex", "pitch"):
                    item[key] = float(item[key])
                item["count"] = float(item["count"])
                if item["shape"] == "cylindrical":
                    item["shape"] = "parabolic"
                if item["axis"] == "both":
                    item["axis"] = "2D"
                item["enabled"] = item["enabled"].lower() == "true"
                if (
                    item["count"] < 0.5
                    or not np.isclose(item["count"] * 2.0, round(item["count"] * 2.0))
                    or item["radius"] <= 0
                    or item["aperture"] <= 0
                    or item["pitch"] <= 0
                ):
                    raise ValueError
                output.append(item)
            except ValueError as exc:
                raise ValueError(
                    f"Stack {index + 1} contains an invalid value"
                ) from exc
        return output

    def _aperture_values(self):
        output = []
        for index, row in enumerate(self.aperture_rows):
            try:
                item = {key: value.get().strip() for key, value in row.items()}
                for key in ("location", "size_x", "size_y", "offset_x", "offset_y"):
                    item[key] = float(item[key])
                item["enabled"] = item["enabled"].lower() == "true"
                if item["size_x"] <= 0 or item["size_y"] <= 0:
                    raise ValueError
                output.append(item)
            except ValueError as exc:
                raise ValueError(
                    f"Aperture {index + 1} contains an invalid value"
                ) from exc
        return output

    def _inputs(self):
        stack_values = self._stack_values()
        aperture_values = self._aperture_values()
        detector_values = self._detector_values()
        optics = []
        for kind, row in self.optic_rows:
            values = stack_values if kind == "stack" else aperture_values
            rows = self.stack_rows if kind == "stack" else self.aperture_rows
            optics.append({"type": kind, **values[rows.index(row)]})
        data = {
            "mode": self.vars["mode"].get(),
            "energy": self._number("energy", True),
            "engine": self.vars["engine"].get(),
            "wave_points": self._number("wave_points", True, True),
            "wave_window": self._number("wave_window", True),
            "spatial_kind": self.vars["spatial_kind"].get(),
            "size_x": 0.0
            if self.vars["spatial_kind"].get() == "point"
            else self._number("size_x", True),
            "size_y": 0.0
            if self.vars["spatial_kind"].get() == "point"
            else self._number("size_y", True),
            "spatial_bound": self._number("spatial_bound", True),
            "divergence_kind": self.vars["divergence_kind"].get(),
            "div_x": 0.0
            if self.vars["divergence_kind"].get() == "collimated"
            else self._number("div_x", False),
            "div_y": 0.0
            if self.vars["divergence_kind"].get() == "collimated"
            else self._number("div_y", False),
            "div_bound": self._number("div_bound", True),
            "source_z": self._number("source_z"),
            "ray_count": self._number("ray_count", True, True),
            "seed": self._number("seed", False, True),
            "detector_z": detector_values[0]["location"],
            "bins_x": detector_values[0]["bins_x"],
            "bins_y": detector_values[0]["bins_y"],
            "detectors": detector_values,
            "stacks": stack_values,
            "apertures": aperture_values,
            "optics": optics,
        }
        if any(item["location"] <= 0 for item in detector_values):
            raise ValueError("Every detector location must be greater than zero")
        return data

    def _refresh_history(self):
        labels = [entry.get("timestamp", "Unknown") for entry in self.history]
        self.history_combo.configure(values=labels)
        if labels:
            self.history_combo.current(0)

    def _apply_configuration(self, data):
        if not isinstance(data, dict):
            raise ValueError("Configuration must be an object")
        stacks = data.get("stacks")
        if not isinstance(stacks, list) or not stacks:
            raise ValueError("Configuration requires at least one CRL stack")
        detector_values = data.get("detectors")
        if isinstance(detector_values, list) and detector_values:
            if not all(isinstance(detector, dict) for detector in detector_values):
                raise ValueError("Invalid detector in configuration")
            detectors = detector_values
        else:
            detectors = [
                {
                    "location": data.get("detector_z", 10.0),
                    "bins_x": data.get("bins_x", 256),
                    "bins_y": data.get("bins_y", 256),
                }
            ]
        optics = data.get("optics")
        prepared_optics = []
        if isinstance(optics, list):
            for optic in optics:
                if not isinstance(optic, dict):
                    raise ValueError("Invalid optic in configuration")
                kind = optic.get("type")
                if kind not in ("stack", "aperture"):
                    raise ValueError("Invalid optic type in configuration")
                prepared_optics.append((kind, optic))
            if not any(kind == "stack" for kind, _ in prepared_optics):
                raise ValueError("Configuration requires at least one CRL stack")
        else:
            for stack in stacks:
                if not isinstance(stack, dict):
                    raise ValueError("Invalid CRL stack in configuration")
                prepared_optics.append(("stack", stack))
            for aperture in data.get("apertures", []):
                if not isinstance(aperture, dict):
                    raise ValueError("Invalid aperture in configuration")
                prepared_optics.append(("aperture", aperture))
        self._validate_configuration_rows(detectors, prepared_optics)
        self.applying_configuration = True
        try:
            for name, variable in self.vars.items():
                if name in data and name not in ("stacks", "apertures", "optics"):
                    variable.set(str(data[name]))
            self.stack_rows.clear()
            self.aperture_rows.clear()
            self.optic_rows.clear()
            self.detector_rows.clear()
            for detector in detectors:
                self.add_detector(detector)
            for kind, optic in prepared_optics:
                values = {
                    key: str(value) for key, value in optic.items() if key != "type"
                }
                if kind == "aperture":
                    self.add_aperture(values)
                else:
                    self.add_stack(values)
        finally:
            self.applying_configuration = False
        self._invalidate_results()
        self._render_optics()
        self._render_detectors()
        self.schedule_preview()

    @staticmethod
    def _validate_configuration_rows(detectors, optics):
        for detector in detectors:
            try:
                if float(detector["location"]) <= 0:
                    raise ValueError
                if int(detector["bins_x"]) <= 0 or int(detector["bins_y"]) <= 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid detector in configuration") from exc
        stack_count = 0
        for kind, optic in optics:
            required = (
                ("location", "count", "radius", "aperture", "apex", "pitch")
                if kind == "stack"
                else ("location", "size_x", "size_y", "offset_x", "offset_y")
            )
            text_required = (
                ("material", "shape", "axis") if kind == "stack" else ("shape",)
            )
            try:
                values = {key: float(optic[key]) for key in required}
                if any(not str(optic[key]).strip() for key in text_required):
                    raise ValueError
                if kind == "stack":
                    stack_count += 1
                    if values["count"] < 0.5 or any(
                        values[key] <= 0 for key in ("radius", "aperture", "pitch")
                    ):
                        raise ValueError
                elif values["size_x"] <= 0 or values["size_y"] <= 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {kind} in configuration") from exc
        if stack_count == 0:
            raise ValueError("Configuration requires at least one CRL stack")

    def save_configuration(self):
        try:
            data = self._inputs()
            selected = filedialog.asksaveasfilename(
                parent=self,
                title="Save CRL configuration",
                defaultextension=".json",
                filetypes=(("JSON configuration", "*.json"),),
            )
            if selected:
                save_configuration(Path(selected), data)
                self.status.configure(text="Configuration saved")
        except Exception as exc:
            self._show_error(exc)

    def load_configuration(self):
        try:
            selected = filedialog.askopenfilename(
                parent=self,
                title="Load CRL configuration",
                filetypes=(("JSON configuration", "*.json"),),
            )
            if selected:
                self._apply_configuration(load_configuration(Path(selected)))
                self.status.configure(text="Configuration loaded")
        except Exception as exc:
            self._show_error(exc)

    def save_results(self):
        try:
            if self.result is None:
                raise ValueError("Run Calculate before saving results")
            selected = filedialog.asksaveasfilename(
                parent=self,
                title="Save CRLSuite results",
                defaultextension=".h5",
                filetypes=(("HDF5 results", "*.h5"), ("HDF5 results", "*.hdf5")),
            )
            if selected:
                results = self.detector_results or [self.result]
                if len(results) == 1:
                    save_hdf5(Path(selected), self._inputs(), results[0])
                else:
                    from .export import save_hdf5_multiple

                    save_hdf5_multiple(Path(selected), self._inputs(), results)
                self.status.configure(text="Results saved")
        except Exception as exc:
            self._show_error(exc)

    def generate_report(self):
        try:
            results = self.detector_results or ([self.result] if self.result else [])
            if not results:
                raise ValueError("Run Calculate before generating a report")
            selected = filedialog.asksaveasfilename(
                parent=self,
                title="Save CRLSuite report",
                defaultextension=".png",
                filetypes=(("PNG image", "*.png"),),
            )
            if not selected:
                return
            report = Figure(figsize=(12, 10), dpi=150, constrained_layout=True)
            grid = report.add_gridspec(2, 1, height_ratios=(3, 4))
            layout_axes = []
            if self.vars["mode"].get() == "2D":
                subgrid = grid[0].subgridspec(2, 1)
                layout_axes = [
                    report.add_subplot(subgrid[0]),
                    report.add_subplot(subgrid[1]),
                ]
            else:
                layout_axes = [report.add_subplot(grid[0])]
            data = self._inputs()
            current_result = self.result
            try:
                self.result = results[-1]
                self._draw_schematic_axis(data, layout_axes[0], "x")
                if len(layout_axes) == 2:
                    self._draw_schematic_axis(data, layout_axes[1], "y")
            finally:
                self.result = current_result
            parameter_tables = self._report_parameter_tables(data)
            table_axis = report.add_subplot(grid[1])
            table_axis.axis("off")
            headers = ["Item", "Position (m)", "X FWHM (µm)", "X Gauss (µm)"]
            headers += ["X div. FWHM (µrad)", "X div. Gauss (µrad)"]
            if data["mode"] == "2D":
                headers += [
                    "Y FWHM (µm)",
                    "Y Gauss (µm)",
                    "Y div. FWHM (µrad)",
                    "Y div. Gauss (µrad)",
                ]
            rows = []
            for index, result in enumerate(results):
                stats = result.stats or {}
                if data["mode"] == "1D":
                    rows.append(
                        [
                            f"Detector {index + 1}",
                            self._report_number(
                                self.detector_rows[index]["location"].get(), 1.0
                            ),
                            self._report_number(stats.get("fwhm"), 1e6),
                            self._report_number(stats.get("gaussian_fwhm"), 1e6),
                            self._report_number(
                                stats.get(
                                    "angular_fwhm_x",
                                    stats.get("wavefront_divergence_fwhm_x"),
                                ),
                                1e6,
                            ),
                            self._report_number(
                                stats.get(
                                    "angular_gaussian_fwhm_x",
                                    stats.get("wavefront_divergence_gaussian_fwhm_x"),
                                ),
                                1e6,
                            ),
                        ]
                    )
                else:
                    rows.append(
                        [
                            f"Detector {index + 1}",
                            self._report_number(
                                self.detector_rows[index]["location"].get(), 1.0
                            ),
                            self._report_number(stats.get("x_fwhm"), 1e6),
                            self._report_number(stats.get("x_gaussian_fwhm"), 1e6),
                            self._report_number(
                                stats.get(
                                    "angular_fwhm_x",
                                    stats.get("wavefront_divergence_fwhm_x"),
                                ),
                                1e6,
                            ),
                            self._report_number(
                                stats.get(
                                    "angular_gaussian_fwhm_x",
                                    stats.get("wavefront_divergence_gaussian_fwhm_x"),
                                ),
                                1e6,
                            ),
                            self._report_number(stats.get("y_fwhm"), 1e6),
                            self._report_number(stats.get("y_gaussian_fwhm"), 1e6),
                            self._report_number(
                                stats.get(
                                    "angular_fwhm_y",
                                    stats.get("wavefront_divergence_fwhm_y"),
                                ),
                                1e6,
                            ),
                            self._report_number(
                                stats.get(
                                    "angular_gaussian_fwhm_y",
                                    stats.get("wavefront_divergence_gaussian_fwhm_y"),
                                ),
                                1e6,
                            ),
                        ]
                    )
            focus_stats = results[-1].stats or {}
            stack_names = sorted(
                {key.split(" focus ", 1)[0] for key in focus_stats if " focus " in key},
                key=self._focus_sort_key,
            )
            focus_headers = ["CRL combination", "X focus (m)"]
            if data["mode"] == "2D":
                focus_headers.append("Y focus (m)")
            focus_rows = []
            for stack_name in stack_names:
                x_focus = focus_stats.get(f"{stack_name} focus x")
                y_focus = focus_stats.get(f"{stack_name} focus y")
                if data["mode"] == "1D":
                    focus_rows.append([stack_name, self._report_number(x_focus, 1.0)])
                else:
                    focus_rows.append(
                        [
                            stack_name,
                            self._report_number(x_focus, 1.0),
                            self._report_number(y_focus, 1.0),
                        ]
                    )
            report_tables = parameter_tables + [("Results", headers, rows)]
            if focus_rows:
                report_tables.append(("CRL Focus Positions", focus_headers, focus_rows))
            self._draw_report_tables(table_axis, report_tables)
            report.savefig(selected, dpi=300, bbox_inches="tight", facecolor=PANEL)
            self.status.configure(text="Report saved")
        except Exception as exc:
            self._show_error(exc)

    @staticmethod
    def _report_number(value, scale):
        return "—" if value is None else f"{float(value) * scale:.5g}"

    @staticmethod
    def _draw_report_tables(axis, tables):
        required_rows = sum(len(rows) + 1 for _, _, rows in tables)
        row_height = min(0.042, 0.82 / max(required_rows, 1))
        title_height = min(0.032, row_height * 0.8)
        section_gap = min(0.018, row_height * 0.45)
        top = 1.0
        for title, headers, rows in tables:
            axis.text(
                0.0,
                top,
                title,
                fontsize=9,
                fontweight="semibold",
                ha="left",
                va="top",
                transform=axis.transAxes,
            )
            table_height = row_height * (len(rows) + 1)
            table_top = top - title_height
            table = axis.table(
                cellText=rows,
                colLabels=headers,
                cellLoc="center",
                bbox=(0.0, table_top - table_height, 1.0, table_height),
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for cell in table.get_celld().values():
                cell.set_height(row_height)
            top = table_top - table_height - section_gap

    @staticmethod
    def _focus_sort_key(name):
        numbers = []
        for part in str(name).split("+"):
            suffix = part.rsplit("CRL", 1)[-1]
            numbers.append(int(suffix) if suffix.isdigit() else math.inf)
        return numbers, str(name)

    @staticmethod
    def _report_parameter_tables(data):
        source_headers = [
            "Energy (keV)",
            "z (m)",
            "Spatial",
            "X size (µm)",
        ]
        source_row = [
            f"{data['energy']:.6g}",
            f"{data['source_z']:.6g}",
            data["spatial_kind"],
            f"{data['size_x']:.6g}",
        ]
        if data["mode"] == "2D":
            source_headers.append("Y size (µm)")
            source_row.append(f"{data['size_y']:.6g}")
        source_headers.extend(["Divergence", "X div. (µrad)"])
        source_row.extend([data["divergence_kind"], f"{data['div_x']:.6g}"])
        if data["mode"] == "2D":
            source_headers.append("Y div. (µrad)")
            source_row.append(f"{data['div_y']:.6g}")
        crl_rows = []
        for index, stack in enumerate(data["stacks"]):
            crl_rows.append(
                [
                    f"CRL{index + 1}",
                    f"{stack['location']:.5g}",
                    "On" if stack["enabled"] else "Off",
                    stack["axis"],
                    stack["material"],
                    stack["shape"],
                    f"{stack['count']:.5g}",
                    f"{stack['radius']:.5g}",
                    f"{stack['aperture']:.5g}",
                    f"{stack['apex']:.5g}",
                    f"{stack['pitch']:.5g}",
                ]
            )
        tables = [
            ("Source", source_headers, [source_row]),
            (
                "CRLs",
                [
                    "CRL",
                    "z (m)",
                    "State",
                    "Axis",
                    "Material",
                    "Shape",
                    "N",
                    "R (mm)",
                    "Ap. (mm)",
                    "Apex (mm)",
                    "Pitch (mm)",
                ],
                crl_rows,
            ),
        ]
        aperture_rows = []
        for index, aperture in enumerate(data["apertures"]):
            aperture_rows.append(
                [
                    f"A{index + 1}",
                    f"{aperture['location']:.5g}",
                    "On" if aperture["enabled"] else "Off",
                    aperture["shape"],
                    f"{aperture['size_x']:.5g}",
                    f"{aperture['size_y']:.5g}",
                    f"{aperture['offset_x']:.5g}",
                    f"{aperture['offset_y']:.5g}",
                ]
            )
        if aperture_rows:
            tables.append(
                (
                    "Apertures",
                    [
                        "Aperture",
                        "z (m)",
                        "State",
                        "Shape",
                        "X size (mm)",
                        "Y size (mm)",
                        "X offset (mm)",
                        "Y offset (mm)",
                    ],
                    aperture_rows,
                )
            )
        return tables

    def _save_figure(self, figure, title):
        selected = filedialog.asksaveasfilename(
            parent=self,
            title=title,
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"),),
        )
        if selected:
            figure.savefig(selected, dpi=300, bbox_inches="tight", facecolor=PANEL)
            self.status.configure(text="Image saved")

    def save_schematic_png(self):
        try:
            self._save_figure(self.figure, "Save schematic image")
        except Exception as exc:
            self._show_error(exc)

    def save_profile_png(self):
        try:
            self._save_figure(self.profile_figure, "Save profile image")
        except Exception as exc:
            self._show_error(exc)

    def restore_history(self):
        try:
            index = self.history_combo.current()
            if index < 0 or index >= len(self.history):
                raise ValueError("Select a history entry to restore")
            self._apply_configuration(self.history[index]["configuration"])
            self.status.configure(text="History restored")
        except Exception as exc:
            self._show_error(exc)

    def _config(self, data):
        dimension = 1 if data["mode"] == "1D" else 2
        spatial_size = np.asarray((data["size_x"], data["size_y"])[:dimension]) * 1e-6
        angular_size = np.asarray((data["div_x"], data["div_y"])[:dimension]) * 1e-6
        spatial_limit = data["spatial_bound"] * 1e-6
        angular_limit = data["div_bound"] * 1e-6
        divergence = data["divergence_kind"]
        if divergence == "collimated":
            divergence = "uniform"
            angular_size[:] = 0.0
        source = Source(
            z=data["source_z"],
            dimension=dimension,
            spatial=data["spatial_kind"],
            size=spatial_size,
            divergence=divergence,
            angular_size=angular_size,
            spatial_bounds=np.tile(
                (-spatial_limit, spatial_limit), (dimension, 1)
            ).tolist(),
            angular_bounds=np.tile(
                (-angular_limit, angular_limit), (dimension, 1)
            ).tolist(),
        )
        stacks = []
        for item in data["stacks"]:
            if not item["enabled"]:
                continue
            stacks.append(
                LensStack(
                    z=item["location"],
                    material=Material.preset(item["material"]),
                    count=item["count"],
                    radius=item["radius"] * 1e-3,
                    aperture=item["aperture"] * 1e-3,
                    apex_thickness=item["apex"] * 1e-3,
                    pitch=item["pitch"] * 1e-3,
                    shape=item["shape"],
                    axis="x" if dimension == 1 else item["axis"],
                )
            )
        apertures = [
            Aperture(
                item["location"],
                item["shape"],
                item["size_x"] * 1e-3,
                item["size_y"] * 1e-3,
                item["offset_x"] * 1e-3,
                item["offset_y"] * 1e-3,
            )
            for item in data["apertures"]
            if item["enabled"]
        ]
        return source, stacks, apertures, Detector(data["detector_z"])

    def preview(self):
        self.pending = None
        try:
            data = self._inputs()
            self.error_label.configure(text="")
            self.status.configure(text="Preview")
            self._draw_schematic(data)
            self._draw_result(self.result, data)
            if self.detector_results:
                self._draw_all_statistics()
        except Exception as exc:
            self._show_error(exc)

    def calculate(self, automatic=False):
        try:
            data = self._inputs()
        except Exception as exc:
            self._show_error(exc)
            return
        self.cancel_calculation(update_status=False)
        self._calculation_job_id += 1
        job_id = self._calculation_job_id
        self.multi_running = True
        self.calculate_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.error_label.configure(text="")
        self.status.configure(text="Calculating…")
        future = self._calculation_executor.submit(self._calculate_data, data)
        self._calculation_future = future
        self.after(25, self._poll_calculation, job_id, future, data)

    def cancel_calculation(self, update_status=True):
        future = self._calculation_future
        if future is None:
            return
        self._calculation_job_id += 1
        future.cancel()
        self._calculation_future = None
        self.multi_running = False
        self.calculate_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if update_status:
            self.status.configure(text="Cancelled")

    def destroy(self):
        self._calculation_job_id += 1
        self._calculation_executor.shutdown(wait=False, cancel_futures=True)
        super().destroy()

    def _poll_calculation(self, job_id, future, data):
        if job_id != self._calculation_job_id:
            return
        if not future.done():
            self.after(25, self._poll_calculation, job_id, future, data)
            return
        self._calculation_future = None
        self.multi_running = False
        self.calculate_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        try:
            completed = future.result()
            self.detector_results = completed
            self.selected_detector = max(
                0, min(self.selected_detector, len(completed) - 1)
            )
            self.result = completed[self.selected_detector]
            self._draw_schematic(data)
            self._draw_result(self.result, data)
            self._draw_all_statistics()
            self.history = append_history(data)
            self._refresh_history()
            self._refresh_detector_selector()
            self.status.configure(text="Complete")
        except Exception as exc:
            self.result = None
            self.detector_results = []
            self._show_error(exc)

    def _calculate_data(self, data):
        completed = []
        for detector in data["detectors"]:
            detector_data = {
                **data,
                "detector_z": detector["location"],
                "bins_x": detector["bins_x"],
                "bins_y": detector["bins_y"],
            }
            worker = object.__new__(CRLSuiteApp)
            worker.multi_running = True
            worker.detector_rows = [None]
            result = worker._calculate_one(detector_data)
            if result is None:
                raise RuntimeError("Detector calculation did not produce a result")
            completed.append(result)
        return completed

    def _calculate_one(self, data):
        try:
            source, stacks, apertures, detector = self._config(data)
            if data["engine"] == "wavefront":
                self.result = simulate_wavefront_adaptive(
                    source,
                    stacks,
                    apertures,
                    detector,
                    data["energy"],
                    data["wave_points"],
                    data["wave_window"] * 1e-3,
                )
                self.result.stats["engine"] = "Coherent Fresnel wavefront"
                self.result.stats["transmission"] = self.result.transmission
                self._add_stack_focus_statistics(self.result, data)
                if not self.multi_running:
                    self.error_label.configure(text="")
                    self._draw_schematic(data)
                    self._draw_result(self.result, data)
                if not self.multi_running:
                    self.history = append_history(data)
                    self._refresh_history()
                return self.result
            self.result = Simulator(data["energy"]).run(
                source,
                stacks,
                detector,
                data["ray_count"],
                data["seed"],
                apertures=apertures,
            )
            boundary_count = 401 * source.dimension
            boundary_positions = np.zeros((boundary_count, source.dimension))
            boundary_angles = np.zeros((boundary_count, source.dimension))
            for axis_index in range(source.dimension):
                start = axis_index * 401
                stop = start + 401
                angular_boundary = self._source_boundary(
                    data["divergence_kind"],
                    data["div_x"] if axis_index == 0 else data["div_y"],
                    data["div_bound"],
                )
                boundary_angles[start:stop, axis_index] = np.linspace(
                    -angular_boundary, angular_boundary, 401
                )
            boundary_bundle = RayBundle(
                boundary_positions * 1e-6,
                boundary_angles * 1e-6,
                np.ones(boundary_count),
            )
            boundary_result = Simulator(data["energy"]).run(
                boundary_bundle,
                stacks,
                detector,
                source_plane=data["source_z"],
                apertures=apertures,
            )
            self.result.metadata["boundary_z"] = boundary_result.trajectory.z
            for axis_index, axis_name in enumerate(("x", "y")[: source.dimension]):
                start = axis_index * 401
                stop = start + 401
                lower = []
                upper = []
                for position, weight in zip(
                    boundary_result.trajectory.position,
                    boundary_result.trajectory.weight,
                    strict=False,
                ):
                    surviving = weight[start:stop] > 0
                    values = position[start:stop, axis_index][surviving]
                    lower.append(float(np.min(values)) if len(values) else np.nan)
                    upper.append(float(np.max(values)) if len(values) else np.nan)
                self.result.metadata[f"boundary_lower_{axis_name}"] = lower
                self.result.metadata[f"boundary_upper_{axis_name}"] = upper
            bundle = self.result.detector
            if data["mode"] == "1D":
                centers, profile, _ = weighted_profile(
                    bundle.position[:, 0], bundle.weight, data["bins_x"]
                )
                self.result.profile = profile
                self.result.profile_coordinates = centers
                self.result.stats = statistics(
                    bundle.position[:, 0],
                    bundle.weight,
                    data["bins_x"],
                    self.result.initial_weight,
                    self.result.aperture_loss,
                    self.result.absorption_loss,
                )
                self.result.stats.update(
                    self._angular_statistics(bundle.angle, bundle.weight)
                )
            else:
                profile, xedges, yedges = weighted_profile2d(
                    bundle.position[:, 0],
                    bundle.position[:, 1],
                    bundle.weight,
                    (data["bins_x"], data["bins_y"]),
                )
                self.result.profile = profile.T
                self.result.profile_extent = (
                    xedges[0],
                    xedges[-1],
                    yedges[0],
                    yedges[-1],
                )
                self.result.stats = statistics2d(
                    bundle.position,
                    bundle.weight,
                    self.result.initial_weight,
                    self.result.aperture_loss,
                    self.result.absorption_loss,
                )
                x_centers = 0.5 * (xedges[:-1] + xedges[1:])
                y_centers = 0.5 * (yedges[:-1] + yedges[1:])
                x_profile = np.sum(profile, axis=1)
                y_profile = np.sum(profile, axis=0)
                self.result.stats.update(
                    {
                        "x_fwhm": interpolated_fwhm(x_centers, x_profile),
                        "y_fwhm": interpolated_fwhm(y_centers, y_profile),
                        "x_gaussian_fwhm": gaussian_fit_fwhm(x_centers, x_profile),
                        "y_gaussian_fwhm": gaussian_fit_fwhm(y_centers, y_profile),
                    }
                )
                self.result.stats.update(
                    self._angular_statistics(bundle.angle, bundle.weight)
                )
            self.result.stats["attenuation_model"] = self.result.metadata[
                "attenuation_model"
            ]
            self.result.stats["mu_total_m_inv"] = self.result.metadata["mu_total_m_inv"]
            self._add_stack_focus_statistics(self.result, data)
            if not self.multi_running:
                self.error_label.configure(text="")
                self._draw_schematic(data)
                self._draw_result(self.result, data)
            if not self.multi_running:
                self.history = append_history(data)
                self._refresh_history()
            return self.result
        except Exception as exc:
            self.result = None
            if self.multi_running:
                raise
            self._show_error(exc)

    def _refresh_detector_selector(self):
        if not hasattr(self, "detector_selector"):
            return
        for child in self.detector_selector.winfo_children():
            child.destroy()
        for index in range(len(self.detector_rows)):
            ttk.Button(
                self.detector_selector,
                text=f"D{index + 1}",
                style="Accent.TButton"
                if index == self.selected_detector
                else "TButton",
                command=partial(self.select_detector, index),
            ).pack(side="left", padx=(0, 3))

    def select_detector(self, index):
        try:
            if 0 <= index < len(self.detector_results):
                self.selected_detector = index
                self.result = self.detector_results[index]
                self._draw_result(self.result, self._inputs())
                self._draw_all_statistics()
                self._refresh_detector_selector()
        except Exception as exc:
            self._show_error(exc)

    def _draw_all_statistics(self):
        sections = []
        for index, result in enumerate(self.detector_results):
            stats = result.stats or {}
            lines = [
                "Detector {} ({} m)".format(
                    index + 1, self.detector_rows[index]["location"].get()
                )
            ]
            for key in (
                "fwhm",
                "gaussian_fwhm",
                "x_fwhm",
                "y_fwhm",
                "x_gaussian_fwhm",
                "y_gaussian_fwhm",
                "transmission",
                "angular_fwhm_x",
                "angular_fwhm_y",
                "angular_gaussian_fwhm_x",
                "angular_gaussian_fwhm_y",
                "wavefront_divergence_fwhm_x",
                "wavefront_divergence_fwhm_y",
                "wavefront_divergence_gaussian_fwhm_x",
                "wavefront_divergence_gaussian_fwhm_y",
            ):
                if key in stats:
                    lines.append(
                        "{}: {}".format(
                            key.replace("_", " ").title(),
                            self._format_statistic(key, stats[key]),
                        )
                    )
            sections.append("\n".join(lines))
        if self.detector_results:
            focus_stats = self.detector_results[-1].stats or {}
            focus_lines = [
                f"{key.replace('_', ' ').title()}: {self._format_statistic(key, value)}"
                for key, value in focus_stats.items()
                if " focus " in key
            ]
            if focus_lines:
                sections.append("Stack Foci\n" + "\n".join(focus_lines))
        self.statistics_text.configure(state="normal")
        self.statistics_text.delete("1.0", "end")
        self.statistics_text.insert("1.0", "\n\n".join(sections))
        self.statistics_text.configure(state="disabled")

    @staticmethod
    def _add_stack_focus_statistics(result, data):
        focal_positions = result.metadata.get("focal_positions", [])
        if not isinstance(focal_positions, list):
            raise RuntimeError("Invalid focal-position results")
        for focal in focal_positions:
            if not isinstance(focal, dict) or not isinstance(
                focal.get("focus_m"), dict
            ):
                continue
            label_value = str(focal["label"])
            try:
                active_index = int(label_value.removeprefix("stack_"))
                active_labels = [
                    f"CRL{index + 1}"
                    for index, row in enumerate(data["stacks"])
                    if row["enabled"]
                ]
                label = "+".join(active_labels[: active_index + 1])
            except ValueError:
                label = label_value.replace("stack_", "Stack ")
            for axis, position in focal["focus_m"].items():
                result.stats[f"{label} focus {axis}"] = position

    @staticmethod
    def _cumulative_crl_label(count):
        return "+".join(f"CRL{index}" for index in range(1, count + 1))

    def _show_error(self, exc):
        message = str(exc) or exc.__class__.__name__
        self.error_label.configure(text=message)
        self.status.configure(text="Input error")

    @staticmethod
    def _source_boundary(kind, size, bound):
        if kind in ("point", "collimated"):
            return 0.0
        if kind == "uniform":
            return min(size, bound)
        return min(3.0 * size, bound)

    @staticmethod
    def _angular_statistics(angle, weight):
        valid = np.all(np.isfinite(angle), axis=1) & (weight > 0)
        values = angle[valid]
        weights = weight[valid]
        if not len(values):
            return {
                "angular_centroid": np.full(angle.shape[1], np.nan),
                "angular_sigma": np.full(angle.shape[1], np.nan),
                "angular_rms": np.full(angle.shape[1], np.nan),
            }
        total = np.sum(weights)
        centroid = np.sum(values * weights[:, None], axis=0) / total
        sigma = np.sqrt(
            np.sum((values - centroid) ** 2 * weights[:, None], axis=0) / total
        )
        rms = np.sqrt(np.sum(values**2 * weights[:, None], axis=0) / total)
        if angle.shape[1] == 1:
            centroid = float(centroid[0])
            sigma = float(sigma[0])
            rms = float(rms[0])
        result = {
            "angular_centroid": centroid,
            "angular_sigma": sigma,
            "angular_rms": rms,
        }
        for axis in range(angle.shape[1]):
            centers, profile, _ = weighted_profile(values[:, axis], weights, bins=256)
            result[f"angular_fwhm_{('x', 'y')[axis]}"] = interpolated_fwhm(
                centers, profile
            )
            result[f"angular_gaussian_fwhm_{('x', 'y')[axis]}"] = gaussian_fit_fwhm(
                centers, profile
            )
        return result

    def _draw_schematic(self, data):
        self.detector_artists = []
        self._draw_schematic_axis(data, self.ax_scheme, "x")
        if data["mode"] == "2D":
            self.ax_scheme_y.set_visible(True)
            self._draw_schematic_axis(data, self.ax_scheme_y, "y")
        else:
            self.ax_scheme_y.set_visible(False)
        self.canvas.draw_idle()

    def _draw_schematic_axis(self, data, ax, axis_name):
        ax.clear()
        schematic_result = (
            max(self.detector_results, key=lambda result: result.detector_z)
            if self.detector_results
            else self.result
        )
        detectors = data["detectors"]
        detector = detectors[self.selected_detector]["location"]
        source_z = data["source_z"]
        active_stacks = [item for item in data["stacks"] if item["enabled"]]
        locations = [item["location"] for item in data["stacks"]]
        zmax = max([item["location"] for item in detectors] + [1.0] + locations) * 1.08
        zmin = min(source_z, 0.0)
        margin = max((zmax - zmin) * 0.03, 0.01)
        ax.set_xlim(zmin - margin, zmax)
        envelope = max(data[f"size_{axis_name}"], 0.05)
        slope = data[f"div_{axis_name}"] * 0.025
        zpoints = [source_z]
        upper = [envelope]
        focus_marks = []
        last_z = source_z
        for _index, item in sorted(
            enumerate(active_stacks), key=lambda pair: pair[1]["location"]
        ):
            z = item["location"]
            before = upper[-1] + slope * (z - last_z)
            zpoints.append(z)
            upper.append(before)
            powered = data["mode"] == "1D" or item["axis"] in ("2D", axis_name)
            power = (
                item["count"] / max(item["radius"], 1e-9) * 0.0008 if powered else 0.0
            )
            slope -= power * before
            if slope < 0:
                focus = z - before / slope
                if z < focus <= zmax * 1.3:
                    focus_marks.append(focus)
            last_z = z
        zpoints.append(detector)
        upper.append(upper[-1] + slope * (detector - last_z))
        ax.fill_between(
            zpoints, upper, [-value for value in upper], color="#dce9fb", alpha=0.75
        )
        ax.plot(zpoints, upper, color=BLUE, linewidth=1.6)
        ax.plot(zpoints, [-value for value in upper], color=BLUE, linewidth=1.6)
        if schematic_result is not None and "boundary_z" in schematic_result.metadata:
            boundary_z, lower, boundary_upper = self._merged_schematic_envelope(
                schematic_result, axis_name, data
            )
            boundary_z, lower, boundary_upper = self._insert_focus_crossings(
                boundary_z,
                lower,
                boundary_upper,
                schematic_result.metadata.get("focal_positions", []),
                axis_name,
                data,
            )
            for artist in list(ax.collections) + list(ax.lines):
                artist.remove()
            ax.fill_between(
                boundary_z,
                lower,
                boundary_upper,
                color="#dce9fb",
                alpha=0.35,
                interpolate=True,
            )
            ax.plot(boundary_z, boundary_upper, color=BLUE, linewidth=1.6)
            ax.plot(boundary_z, lower, color=BLUE, linewidth=1.6)
            ymax = (
                max(float(np.nanmax(np.abs(np.r_[boundary_upper, lower]))), 0.1) * 1.5
            )
            result_focus = getattr(schematic_result, "focus", {})
            axis_focus = result_focus.get(axis_name, np.nan)
            focus_marks = [axis_focus] if np.isfinite(axis_focus) else []
        ax.axhline(0, color="#aab3c2", linewidth=0.8)
        ax.plot(source_z, 0, marker="o", color="#25845b", markersize=6)
        aperture_um = [item["aperture"] * 1000.0 for item in data["stacks"]]
        if schematic_result is not None and "boundary_z" in schematic_result.metadata:
            beam_values = np.r_[lower, boundary_upper]
            finite_beam_values = beam_values[np.isfinite(beam_values)]
            beam_half_height = (
                float(np.max(np.abs(finite_beam_values)))
                if len(finite_beam_values)
                else 0.1
            )
        else:
            beam_half_height = max(max(abs(value) for value in upper), 0.1)
        lens_half_height = max(aperture_um, default=0.0) * 0.5
        aperture_extent = max(
            (
                abs(item[f"offset_{axis_name}"]) * 1000.0
                + item[f"size_{axis_name}"] * 500.0
                for item in data["apertures"]
            ),
            default=0.0,
        )
        ymax = max(
            beam_half_height * 1.15,
            lens_half_height * 1.15,
            aperture_extent * 1.15,
            0.1,
        )
        label_items = [("source", 0, source_z)]
        label_items.extend(
            ("crl", index, item["location"])
            for index, item in enumerate(data["stacks"])
        )
        label_items.extend(
            ("aperture", index, item["location"])
            for index, item in enumerate(data["apertures"])
        )
        label_items.extend(
            ("detector", index, item["location"])
            for index, item in enumerate(detectors)
        )
        label_x = self._schematic_label_positions(label_items, zmin, zmax)
        source_label_x, source_lane = label_x[("source", 0)]
        self._schematic_annotation(
            ax,
            source_label_x,
            source_lane,
            source_z,
            f"Source\n{source_z:.4g} m",
            ymax,
            "#25845b",
        )
        for index, item in enumerate(data["stacks"]):
            z = item["location"]
            height = item["aperture"] * 1000.0
            enabled = item["enabled"]
            powered = data["mode"] == "1D" or item["axis"] in ("2D", axis_name)
            half_width = zmax * 0.008
            waist = half_width * 0.22
            half_height = height * 0.5
            lens_path = PlotPath(
                [
                    (z - half_width, half_height),
                    (z - waist, half_height * 0.45),
                    (z - waist, -half_height * 0.45),
                    (z - half_width, -half_height),
                    (z + half_width, -half_height),
                    (z + waist, -half_height * 0.45),
                    (z + waist, half_height * 0.45),
                    (z + half_width, half_height),
                    (z - half_width, half_height),
                ],
                [
                    PlotPath.MOVETO,
                    PlotPath.CURVE4,
                    PlotPath.CURVE4,
                    PlotPath.CURVE4,
                    PlotPath.LINETO,
                    PlotPath.CURVE4,
                    PlotPath.CURVE4,
                    PlotPath.CURVE4,
                    PlotPath.CLOSEPOLY,
                ],
            )
            patch = (
                PathPatch(lens_path)
                if powered
                else PlotRectangle(
                    (z - half_width, -half_height), 2 * half_width, height
                )
            )
            patch.set_facecolor("#7ba6e6" if enabled else "#c2c6cc")
            patch.set_edgecolor(BLUE if enabled else "#7b818a")
            patch.set_linewidth(1.2)
            patch.set_alpha(1.0 if enabled else 0.45)
            ax.add_patch(patch)
            crl_label_x, crl_lane = label_x[("crl", index)]
            self._schematic_annotation(
                ax,
                crl_label_x,
                crl_lane,
                z,
                (
                    f"CRL{index + 1}\n{z:.4g} m"
                    if enabled
                    else f"CRL{index + 1} hidden\n{z:.4g} m"
                ),
                ymax,
                color=MUTED if enabled else "#9aa0a8",
            )
        for index, item in enumerate(data["apertures"]):
            z = item["location"]
            height = item[f"size_{axis_name}"] * 1000.0
            enabled = item["enabled"]
            ax.add_patch(
                PlotRectangle(
                    (z - zmax * 0.003, -height * 0.5),
                    zmax * 0.006,
                    height,
                    facecolor="#31363f" if enabled else "#b6bac0",
                    edgecolor="#111318" if enabled else "#7b818a",
                    alpha=0.8 if enabled else 0.4,
                )
            )
            aperture_label_x, aperture_lane = label_x[("aperture", index)]
            self._schematic_annotation(
                ax,
                aperture_label_x,
                aperture_lane,
                z,
                (
                    f"A{index + 1}\n{z:.4g} m"
                    if enabled
                    else f"A{index + 1} hidden\n{z:.4g} m"
                ),
                ymax,
                color=MUTED,
            )
        for focus in focus_marks:
            ax.plot(focus, 0, marker="x", color="#6c3bbd", markersize=8)
        for detector_index, detector_item in enumerate(detectors):
            detector_z = detector_item["location"]
            selected = detector_index == self.selected_detector
            artist = ax.plot(
                [detector_z, detector_z],
                [-ymax * 0.72, ymax * 0.72],
                color=RED if selected else "#d77a1f",
                linewidth=3 if selected else 1.8,
                picker=8,
            )[0]
            if ax in (self.ax_scheme, self.ax_scheme_y):
                self.detector_artists.append(artist)
            detector_label_x, detector_lane = label_x[("detector", detector_index)]
            self._schematic_annotation(
                ax,
                detector_label_x,
                detector_lane,
                detector_z,
                f"D{detector_index + 1}\n{detector_z:.4g} m",
                ymax,
                color=RED if selected else "#d77a1f",
            )
        ax.set_ylim(-ymax, ymax * 1.45)
        ax.set_xlabel("z (m)")
        ax.set_ylabel(f"{axis_name.upper()} envelope (µm)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e8ebf0", linewidth=0.7)

    def _merged_schematic_envelope(self, schematic_result, axis_name, data):
        results = self.detector_results or [schematic_result]
        farthest = max(results, key=lambda result: result.detector_z)
        z = np.asarray(farthest.metadata["boundary_z"], dtype=float)
        lower = np.asarray(
            farthest.metadata[f"boundary_lower_{axis_name}"], dtype=float
        )
        upper = np.asarray(
            farthest.metadata[f"boundary_upper_{axis_name}"], dtype=float
        )
        points = list(zip(z.tolist(), lower.tolist(), upper.tolist(), strict=False))
        for result in results:
            detector_z = float(result.detector_z)
            result_z = np.asarray(result.metadata["boundary_z"], dtype=float)
            detector_index = int(np.argmin(np.abs(result_z - detector_z)))
            result_lower = np.asarray(
                result.metadata[f"boundary_lower_{axis_name}"], dtype=float
            )
            result_upper = np.asarray(
                result.metadata[f"boundary_upper_{axis_name}"], dtype=float
            )
            points = [point for point in points if not np.isclose(point[0], detector_z)]
            points.append(
                (
                    detector_z,
                    float(result_lower[detector_index]),
                    float(result_upper[detector_index]),
                )
            )
        if data["engine"] == "wavefront":
            source_z = float(data["source_z"])
            points = [point for point in points if not np.isclose(point[0], source_z)]
            points.append((source_z, 0.0, 0.0))
        points.sort(key=lambda point: point[0])
        merged = tuple(np.asarray(values) for values in zip(*points, strict=False))
        return merged[0], merged[1] * 1e6, merged[2] * 1e6

    @staticmethod
    def _insert_focus_crossings(
        boundary_z,
        lower,
        upper,
        focal_positions,
        axis_name,
        data,
    ):
        points = list(
            zip(boundary_z.tolist(), lower.tolist(), upper.tolist(), strict=False)
        )
        active_positions = sorted(
            [item["location"] for item in data["stacks"] if item["enabled"]]
            + [item["location"] for item in data["apertures"] if item["enabled"]]
        )
        for focal in focal_positions:
            if not isinstance(focal, dict):
                continue
            stack_z = float(focal.get("z_m", np.nan))
            focus_values = focal.get("focus_m", {})
            if not isinstance(focus_values, dict):
                continue
            focus_z = float(focus_values.get(axis_name, np.nan))
            if not np.isfinite(focus_z) or focus_z <= stack_z:
                continue
            next_optic = min(
                (
                    position
                    for position in active_positions
                    if position > stack_z + 1e-9
                ),
                default=np.inf,
            )
            if focus_z >= next_optic or focus_z > float(np.max(boundary_z)):
                continue
            points.append((focus_z, 0.0, 0.0))
        points.sort(key=lambda point: point[0])
        return tuple(np.asarray(values) for values in zip(*points, strict=False))

    @staticmethod
    def _schematic_label_positions(items, zmin, zmax):
        span = max(zmax - zmin, 1e-9)
        minimum_gap = span * 0.12
        shift = span * 0.065
        positions = {}
        ordered = sorted(items, key=lambda item: item[2])
        cluster: list[tuple[str, int, float]] = []
        for item in ordered:
            if cluster and item[2] - cluster[-1][2] >= minimum_gap:
                for offset_index, clustered in enumerate(cluster):
                    centered = offset_index - (len(cluster) - 1) / 2
                    positions[(clustered[0], clustered[1])] = (
                        clustered[2] + centered * shift,
                        offset_index % 2,
                    )
                cluster = []
            cluster.append(item)
        for offset_index, clustered in enumerate(cluster):
            centered = offset_index - (len(cluster) - 1) / 2
            positions[(clustered[0], clustered[1])] = (
                clustered[2] + centered * shift,
                offset_index % 2,
            )
        return positions

    @staticmethod
    def _schematic_annotation(ax, label_x, lane, element_x, text, ymax, color):
        shifted = not np.isclose(label_x, element_x)
        ax.annotate(
            text,
            xy=(element_x, ymax * 0.74),
            xytext=(label_x, ymax * (1.08 + 0.16 * lane)),
            color=color,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="semibold",
            clip_on=False,
            arrowprops=(
                {"arrowstyle": "-", "color": color, "linewidth": 0.8}
                if shifted
                else None
            ),
        )

    def _draw_result(self, result, data):
        ax = self.ax_profile
        ax.clear()
        self.ax_profile_right.clear()
        self.ax_profile_bottom.clear()
        self.ax_profile_corner.clear()
        self.ax_profile_corner.axis("off")
        is_2d = data["mode"] == "2D"
        self.ax_profile_right.set_visible(is_2d)
        self.ax_profile_bottom.set_visible(is_2d)
        self.ax_profile_corner.set_visible(is_2d)
        profile = getattr(result, "profile", None) if result is not None else None
        if profile is None:
            ax.text(
                0.5,
                0.5,
                "Run Calculate to display the profile",
                ha="center",
                va="center",
                color=MUTED,
                transform=ax.transAxes,
            )
        else:
            array = np.asarray(profile)
            if data["mode"] == "2D" or array.ndim == 2:
                image = array if array.ndim == 2 else np.atleast_2d(array)
                raw_extent = result.profile_extent or (
                    0,
                    image.shape[1],
                    0,
                    image.shape[0],
                )
                extent = np.asarray(raw_extent) * 1e6
                x_slice = profile_crop_slice(image)
                y_slice = profile_crop_slice(image.T)
                x_step = (extent[1] - extent[0]) / image.shape[1]
                y_step = (extent[3] - extent[2]) / image.shape[0]
                full_x = extent[0] + (np.arange(image.shape[1]) + 0.5) * x_step
                full_y = extent[2] + (np.arange(image.shape[0]) + 0.5) * y_step
                image = image[y_slice, x_slice]
                x_centers = full_x[x_slice]
                y_centers = full_y[y_slice]
                display_extent = (
                    x_centers[0],
                    x_centers[-1],
                    y_centers[0],
                    y_centers[-1],
                )
                ax.imshow(
                    image,
                    origin="lower",
                    aspect="auto",
                    cmap="magma",
                    extent=display_extent,
                )
                x_integrated = np.sum(image, axis=0)
                y_integrated = np.sum(image, axis=1)
                self.ax_profile_bottom.plot(x_centers, x_integrated, color=BLUE)
                self.ax_profile_bottom.fill_between(
                    x_centers, x_integrated, color=BLUE, alpha=0.12
                )
                self.ax_profile_right.plot(y_integrated, y_centers, color=BLUE)
                self.ax_profile_right.fill_betweenx(
                    y_centers, y_integrated, color=BLUE, alpha=0.12
                )
                self.ax_profile_bottom.set_xlabel("X (µm)")
                self.ax_profile_bottom.set_ylabel("ΣY", fontsize=8)
                self.ax_profile_right.set_xlabel("ΣX", fontsize=8)
                self.ax_profile_bottom.tick_params(labelsize=7)
                self.ax_profile_right.tick_params(labelsize=7)
                ax.tick_params(labelbottom=False)
                self.ax_profile_right.tick_params(labelleft=False)
                ax.set_xlabel("X (µm)")
                ax.set_ylabel("Y (µm)")
            else:
                line = array.ravel()
                coordinate = (
                    np.asarray(
                        getattr(result, "profile_coordinates", np.arange(line.size))
                    )
                    * 1e6
                )
                crop = profile_crop_slice(line)
                line = line[crop]
                coordinate = coordinate[crop]
                ax.plot(coordinate, line, color=BLUE, linewidth=1.7)
                ax.fill_between(coordinate, line, color=BLUE, alpha=0.12)
                peak = float(np.max(line)) if len(line) else 0.0
                if peak > 0:
                    ax.set_ylim(0.0, peak * 1.08)
                if len(coordinate) > 1:
                    ax.set_xlim(float(coordinate[0]), float(coordinate[-1]))
                ax.set_xlabel("Position (µm)")
                ax.set_ylabel("Intensity")
                self.ax_profile_right.set_visible(False)
                self.ax_profile_bottom.set_visible(False)
                self.ax_profile_corner.set_visible(False)
        if is_2d:
            self.ax_profile_right.set_visible(True)
            self.ax_profile_bottom.set_visible(True)
            self.ax_profile_corner.set_visible(True)
            self.ax_profile_corner.axis("off")
        ax.set_title(
            "Detector profile",
            loc="left",
            color=INK,
            fontsize=11,
            fontweight="semibold",
        )
        ax.spines[["top", "right"]].set_visible(False)
        stats = getattr(result, "stats", None) if result is not None else None
        hidden_2d_stats = {
            "covariance",
            "correlation",
            "principal_rms",
            "principal_axes",
        }
        hidden_stats = {
            "attenuation_model",
            "mu_total_m_inv",
            "encircled_radii",
            "weight",
        }
        if stats is None:
            text = "No simulation results"
        elif isinstance(stats, dict):
            text = "\n".join(
                "{}: {}".format(
                    str(key).replace("_", " ").title(),
                    self._format_statistic(str(key), value),
                )
                for key, value in stats.items()
                if key not in hidden_stats
                and not (data["mode"] == "2D" and key in hidden_2d_stats)
            )
        elif is_dataclass(stats):
            text = "\n".join(
                "{}: {}".format(
                    item.name.replace("_", " ").title(),
                    self._format_value(getattr(stats, item.name)),
                )
                for item in fields(stats)
            )
        else:
            entries = (
                [
                    (key, value)
                    for key, value in vars(stats).items()
                    if not key.startswith("_")
                ]
                if hasattr(stats, "__dict__")
                else [("Result", stats)]
            )
            text = "\n".join(
                "{}: {}".format(
                    key.replace("_", " ").title(), self._format_value(value)
                )
                for key, value in entries
            )
        self.statistics_text.configure(state="normal")
        self.statistics_text.delete("1.0", "end")
        self.statistics_text.insert("1.0", text)
        self.statistics_text.configure(state="disabled")
        self.profile_canvas.draw_idle()

    def _format_value(self, value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.5g}"
        if isinstance(value, np.ndarray):
            return np.array2string(
                value,
                precision=5,
                suppress_small=False,
                separator=", ",
                max_line_width=30,
            )
        return str(value)

    def _format_statistic(self, key, value):
        array = (
            np.asarray(value) if isinstance(value, (np.ndarray, list, tuple)) else None
        )
        if key.startswith("angular_"):
            scaled = array * 1e6 if array is not None else float(value) * 1e6
            return f"{self._format_value(scaled)} µrad"
        if key.startswith("wavefront_divergence_"):
            return f"{self._format_value(float(value) * 1e6)} µrad"
        if key.startswith("wavefront_virtual_source_"):
            return f"{self._format_value(value)} m"
        if key in {
            "centroid",
            "sigma",
            "rms",
            "fwhm",
            "gaussian_fwhm",
            "x_fwhm",
            "y_fwhm",
            "x_gaussian_fwhm",
            "y_gaussian_fwhm",
        }:
            scaled = array * 1e6 if array is not None else float(value) * 1e6
            return f"{self._format_value(scaled)} µm"
        if key == "principal_rms":
            return f"{self._format_value(np.asarray(value) * 1e6)} µm"
        if key == "encircled_radii":
            return ", ".join(
                f"{fraction:g}: {radius * 1e6:.5g} µm"
                for fraction, radius in value.items()
            )
        if key == "covariance":
            return f"{self._format_value(np.asarray(value) * 1e12)} µm²"
        if key == "wave_pixel_size":
            return f"{self._format_value(float(value) * 1e6)} µm"
        if key.endswith(" focus x") or key.endswith(" focus y"):
            return f"{self._format_value(value)} m"
        return self._format_value(value)

    def _plot_press(self, event):
        if (
            event.inaxes in (self.ax_scheme, self.ax_scheme_y)
            and event.xdata is not None
        ):
            span = self.ax_scheme.get_xlim()[1] - self.ax_scheme.get_xlim()[0]
            try:
                locations = [float(row["location"].get()) for row in self.detector_rows]
            except ValueError:
                self._show_error(ValueError("Detector locations must be numbers"))
                return
            nearest = int(np.argmin(np.abs(np.asarray(locations) - event.xdata)))
            if abs(event.xdata - locations[nearest]) < span * 0.025:
                self.selected_detector = nearest
                self.drag_detector = True

    def _plot_motion(self, event):
        if (
            self.drag_detector
            and event.inaxes in (self.ax_scheme, self.ax_scheme_y)
            and event.xdata is not None
        ):
            value = max(0.001, event.xdata)
            self.detector_rows[self.selected_detector]["location"].set(f"{value:.4g}")
            for artist in getattr(self, "detector_artists", []):
                artist.set_xdata([value, value])
            self.canvas.draw_idle()

    def _plot_release(self, event):
        if self.drag_detector:
            self.drag_detector = False
            self.schedule_preview()


def main():
    app = CRLSuiteApp()
    app.mainloop()


if __name__ == "__main__":
    main()
