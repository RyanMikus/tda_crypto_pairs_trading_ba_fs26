"""All interactive controls for the dashboard, wired back to DashboardApp via
callbacks. Continuous sliders (correlation lookback) debounce through
DashboardApp.on_control_changed(debounce=True); discrete controls (buttons,
dropdowns, entries, basket bounds apply) trigger an immediate recompute."""
from __future__ import annotations

import pandas as pd
import customtkinter as ctk

from ..state import min_valid_date

MAX_CORR_LOOKBACK_DAYS = 100
MAX_BASKET_LOOKBACK_DAYS = 60


class Sidebar(ctk.CTkScrollableFrame):
    def __init__(self, master, *, app, data, state):
        super().__init__(master, width=320, label_text="Controls")
        self.app = app
        self.data = data
        self.state = state

        self._date_lower_days = 0
        self._date_upper_days = 0

        self._build_date_controls()
        self._add_divider()
        self._build_basket_lookback_controls()
        self._build_basket_controls()
        self._add_divider()
        self._build_corr_lookback_controls()
        self._build_regression_controls()
        self._build_heatmap_kind_controls()
        self._build_epsilon_controls()
        self._build_clusters_controls()

    def _add_divider(self) -> None:
        divider = ctk.CTkFrame(self, height=2, fg_color=("gray70", "gray30"))
        divider.pack(fill="x", pady=(8, 8))

    # ------------------------------------------------------------------ #
    # Date
    # ------------------------------------------------------------------ #
    def _build_date_controls(self) -> None:
        ctk.CTkLabel(self, text="Date", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(4, 0))

        self.date_value_label = ctk.CTkLabel(self, text=str(self.state.date.date()))
        self.date_value_label.pack(anchor="w")

        total_days = (self.data.data_max - self.data.data_min).days
        lower_days = self._min_valid_day_offset()
        self._date_lower_days = lower_days
        self._date_upper_days = total_days

        self.date_slider = ctk.CTkSlider(
            self, from_=lower_days, to=total_days,
            number_of_steps=max(1, total_days - lower_days),
            command=self._on_date_slider,
        )
        self.date_slider.set(self._day_offset(self.state.date))
        self.date_slider.pack(fill="x", pady=(0, 4))

        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.pack(fill="x", pady=(0, 8))
        ctk.CTkButton(button_row, text="< Previous", width=90, command=self._on_previous_day).pack(side="left")
        ctk.CTkButton(button_row, text="Next >", width=90, command=self._on_next_day).pack(side="right")

    def _min_valid_day_offset(self) -> int:
        lower_date = min_valid_date(
            self.data.data_min, self.state.basket_lookback_days, self.state.corr_lookback_days
        )
        return max(0, (lower_date - self.data.data_min).days)

    def _day_offset(self, date: pd.Timestamp) -> int:
        return (date - self.data.data_min).days

    def _apply_date(self, day_offset: int, *, debounce: bool) -> None:
        self.state.date = self.data.data_min + pd.Timedelta(days=day_offset)
        self.date_value_label.configure(text=str(self.state.date.date()))
        self.app.on_control_changed(debounce=debounce)

    def _on_date_slider(self, value: float) -> None:
        self._apply_date(int(round(value)), debounce=True)

    def _on_previous_day(self) -> None:
        new_value = max(self._date_lower_days, self.date_slider.get() - 1)
        self.date_slider.set(new_value)
        self._apply_date(int(round(new_value)), debounce=False)

    def _on_next_day(self) -> None:
        new_value = min(self._date_upper_days, self.date_slider.get() + 1)
        self.date_slider.set(new_value)
        self._apply_date(int(round(new_value)), debounce=False)

    def sync_date(self, date: pd.Timestamp) -> None:
        """Called by DashboardApp after clamping the date -- updates the slider
        position/label only, does not re-trigger a recompute."""
        self.date_slider.set(self._day_offset(date))
        self.date_value_label.configure(text=str(date.date()))

    def refresh_bounds(self, data, state) -> None:
        lower_days = self._min_valid_day_offset()
        total_days = (data.data_max - data.data_min).days
        self._date_lower_days = lower_days
        self._date_upper_days = total_days
        self.date_slider.configure(
            from_=lower_days, to=total_days, number_of_steps=max(1, total_days - lower_days)
        )

    # ------------------------------------------------------------------ #
    # Basket smoothing lookback -- grouped with basket bounds/apply below,
    # since both define the basket config together.
    # ------------------------------------------------------------------ #
    def _build_basket_lookback_controls(self) -> None:
        ctk.CTkLabel(
            self, text="Basket Smoothing Lookback (days)", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", pady=(8, 0))
        self.basket_lookback_label = ctk.CTkLabel(self, text=str(self.state.basket_lookback_days))
        self.basket_lookback_label.pack(anchor="w")
        self.basket_lookback_slider = ctk.CTkSlider(
            self, from_=1, to=MAX_BASKET_LOOKBACK_DAYS,
            number_of_steps=MAX_BASKET_LOOKBACK_DAYS - 1,
            command=self._on_basket_lookback,
        )
        self.basket_lookback_slider.set(self.state.basket_lookback_days)
        self.basket_lookback_slider.pack(fill="x", pady=(0, 8))

    # ------------------------------------------------------------------ #
    # Correlation lookback -- its own section, after basket config.
    # ------------------------------------------------------------------ #
    def _build_corr_lookback_controls(self) -> None:
        ctk.CTkLabel(
            self, text="Correlation Lookback (days)", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", pady=(8, 0))
        self.corr_lookback_label = ctk.CTkLabel(self, text=str(self.state.corr_lookback_days))
        self.corr_lookback_label.pack(anchor="w")
        self.corr_lookback_slider = ctk.CTkSlider(
            self, from_=1, to=MAX_CORR_LOOKBACK_DAYS,
            number_of_steps=MAX_CORR_LOOKBACK_DAYS - 1,
            command=self._on_corr_lookback,
        )
        self.corr_lookback_slider.set(self.state.corr_lookback_days)
        self.corr_lookback_slider.pack(fill="x", pady=(0, 8))

    def _on_basket_lookback(self, value: float) -> None:
        self.state.basket_lookback_days = int(round(value))
        self.basket_lookback_label.configure(text=str(self.state.basket_lookback_days))
        self.app.on_control_changed(debounce=True)

    def _on_corr_lookback(self, value: float) -> None:
        self.state.corr_lookback_days = int(round(value))
        self.corr_lookback_label.configure(text=str(self.state.corr_lookback_days))
        self.app.on_control_changed(debounce=True)

    # ------------------------------------------------------------------ #
    # Basket bounds
    # ------------------------------------------------------------------ #
    def _build_basket_controls(self) -> None:
        ctk.CTkLabel(
            self, text="Basket Upper Bounds (USD, comma-separated)", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", pady=(8, 0))
        self.basket_bounds_entry = ctk.CTkEntry(self)
        self.basket_bounds_entry.insert(0, ",".join(str(int(b)) for b in self.state.basket_upper_bounds))
        self.basket_bounds_entry.pack(fill="x", pady=(0, 4))

        self.basket_bounds_error_label = ctk.CTkLabel(self, text="", text_color="red")
        self.basket_bounds_error_label.pack(anchor="w")

        ctk.CTkButton(self, text="Apply Basket Bounds", command=self._on_apply_basket_bounds).pack(
            fill="x", pady=(0, 4)
        )

    def _on_apply_basket_bounds(self) -> None:
        raw = self.basket_bounds_entry.get()
        try:
            bounds = tuple(sorted(float(part.strip()) for part in raw.split(",") if part.strip()))
        except ValueError:
            self.basket_bounds_error_label.configure(text="Invalid input -- use comma-separated numbers.")
            return

        if len(bounds) == 0:
            self.basket_bounds_error_label.configure(text="At least one bound is required.")
            return

        self.basket_bounds_error_label.configure(text="")
        self.state.basket_upper_bounds = bounds
        self.app.on_control_changed(debounce=False)

    # ------------------------------------------------------------------ #
    # Regression target -- Raw and Regressed are always both shown
    # (Heatmap/Dendrogram/All Baskets), this only controls what "Regressed"
    # means.
    # ------------------------------------------------------------------ #
    def _build_regression_controls(self) -> None:
        ctk.CTkLabel(self, text="Regression Target", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(8, 0))
        self.regression_menu = ctk.CTkOptionMenu(
            self, values=["BTC", "ETH", "BTC+ETH", "PCA"], command=self._on_regression_target,
        )
        self.regression_menu.set(self.state.regression_target)
        self.regression_menu.pack(fill="x", pady=(0, 8))

    def _on_regression_target(self, value: str) -> None:
        self.state.regression_target = value
        self.app.on_control_changed(debounce=False)

    # ------------------------------------------------------------------ #
    # Heatmap kind
    # ------------------------------------------------------------------ #
    def _build_heatmap_kind_controls(self) -> None:
        ctk.CTkLabel(self, text="Heatmap Shows", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(8, 0))
        self.heatmap_kind_menu = ctk.CTkOptionMenu(
            self, values=["Correlation", "Correlation Distance"], command=self._on_heatmap_kind,
        )
        self.heatmap_kind_menu.set(
            "Correlation" if self.state.heatmap_kind == "correlation" else "Correlation Distance"
        )
        self.heatmap_kind_menu.pack(fill="x", pady=(0, 8))

    def _on_heatmap_kind(self, value: str) -> None:
        self.state.heatmap_kind = "distance" if value == "Correlation Distance" else "correlation"
        self.app.on_control_changed(debounce=False)

    # ------------------------------------------------------------------ #
    # Merge Scale (epsilon*) -- numeric input, standard value 1.0. Named
    # after the corrected/preferred TDA terminology (the "death scale" /
    # merge height at which the dendrogram is cut), not the informal
    # "birth scale" used in earlier drafts.
    # ------------------------------------------------------------------ #
    def _build_epsilon_controls(self) -> None:
        ctk.CTkLabel(
            self, text="Merge Scale (ε*)", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", pady=(8, 0))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(0, 4))

        self.epsilon_entry = ctk.CTkEntry(row, width=100)
        self.epsilon_entry.insert(0, f"{self.state.epsilon:.4f}")
        self.epsilon_entry.pack(side="left", padx=(0, 8))
        self.epsilon_entry.bind("<Return>", lambda _event: self._on_apply_epsilon())

        ctk.CTkButton(row, text="Apply", width=70, command=self._on_apply_epsilon).pack(side="left")

        self.epsilon_error_label = ctk.CTkLabel(self, text="", text_color="red")
        self.epsilon_error_label.pack(anchor="w", pady=(0, 8))

    def _on_apply_epsilon(self) -> None:
        raw = self.epsilon_entry.get()
        try:
            value = float(raw)
        except ValueError:
            self.epsilon_error_label.configure(text="Invalid number.")
            return

        if value < 0:
            self.epsilon_error_label.configure(text="Epsilon must be >= 0.")
            return

        self.epsilon_error_label.configure(text="")
        self.state.epsilon = value
        self.app.on_control_changed(debounce=False)

    # ------------------------------------------------------------------ #
    # Clusters history scan -- heavier than the basket-count scan (builds a
    # point cloud + linkage per date per basket per mode), so this is a
    # separate manual trigger. Once run, changing epsilon just re-cuts the
    # cached linkages (cheap, no new scan needed).
    # ------------------------------------------------------------------ #
    def _build_clusters_controls(self) -> None:
        ctk.CTkLabel(
            self, text="Clusters", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", pady=(8, 0))
        ctk.CTkButton(self, text="Recompute Clusters", command=self._on_recompute_clusters).pack(
            fill="x", pady=(0, 4)
        )
        self.clusters_progress = ctk.CTkProgressBar(self)
        self.clusters_progress.set(0)
        self.clusters_progress.pack(fill="x", pady=(0, 4))
        self.clusters_status_label = ctk.CTkLabel(self, text="Not computed yet.")
        self.clusters_status_label.pack(anchor="w", pady=(0, 8))

    def _on_recompute_clusters(self) -> None:
        self.app.recompute_clusters()

    def set_clusters_progress(self, done: int, total: int) -> None:
        fraction = (done / total) if total else 0.0
        self.clusters_progress.set(fraction)
        if total and done >= total:
            self.clusters_status_label.configure(text=f"Done ({total} dates).")
        else:
            self.clusters_status_label.configure(text=f"{done}/{total} dates scanned...")

    def set_clusters_error(self, message: str) -> None:
        self.clusters_status_label.configure(text=f"Error: {message}")
