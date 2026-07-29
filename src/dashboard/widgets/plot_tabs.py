"""CTkTabview hosting: a merged Universe & Baskets overview (bar chart +
all-time boxplot + clickable line chart), a Heatmap tab (Raw stacked above
Regressed, scrollable), a Dendrogram tab (Raw next to Regressed), and a
scrollable All Baskets tab (one row per basket, each combining Raw and
Regressed sub-rows). Heatmap/dendrogram drawing reuses cloudvisualisation.py
directly (ax=..., show=False).

Raw and Regressed are always shown together everywhere -- there is no
Raw/Regressed toggle; "compare" is simply how the app always behaves."""
from __future__ import annotations

import numpy as np
import pandas as pd
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .. import _bootstrap  # noqa: F401
from cloudvisualisation import plot_correlation_heatmap, plot_dendrogram

from .. import plots as dashboard_plots
from ..compute import build_wide_metric_df

_SQRT2 = float(np.sqrt(2))
_SQRT2_LINE = ("sqrt(2) -- no correlation", _SQRT2, "#4a3aa7")
# Fixed dendrogram y-ceiling so the axis stays comparable across dates/windows
# instead of rescaling to whatever the current window's max merge height is.
_DENDROGRAM_Y_MAX = _SQRT2 * 1.15

# All Baskets rows use one constant size regardless of a basket's asset count,
# so every row is directly comparable while scrolling (unlike the main tabs,
# which scale with asset count to stay legible for the full ~29-asset Universe).
_ALL_BASKETS_ROW_FIGSIZE = (12.0, 6.0)

# Clusters tab: one constant per-panel size, used for every panel in both the
# Universe superplot (3x2 grid) and each basket superplot (2x2 grid), so
# every panel of a given kind is directly comparable regardless of section.
_CLUSTERS_CELL_FIGSIZE = (6.0, 4.0)


def _status_message(result: dict) -> str:
    status = result.get("status", "unknown")
    if status == "missing_factor":
        return f"Regression factor asset(s) {result.get('missing')} not available on this date."
    if status == "single":
        return f"Only one asset ({result.get('symbol')}) available -- no pairwise distance possible."
    if status == "empty":
        return "No assets available for this scope on this date."
    if status == "no_data":
        return "Not enough shared price history for this window."
    return f"No data ({status})."


def _draw_result_or_status(ax, result: dict, title: str, draw_fn) -> None:
    """Draws via draw_fn(ax, result) if result is ok, otherwise shows an
    inline status message on ax -- keeps one bad panel (e.g. a missing
    regression factor) from blanking out an entire multi-panel figure."""
    if result.get("status") == "ok":
        draw_fn(ax, result)
    else:
        ax.text(
            0.5, 0.5, _status_message(result), ha="center", va="center",
            transform=ax.transAxes, wrap=True, fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        ax.set_title(title)


def _heatmap_style(heatmap_kind: str):
    """Returns (matrix_key, value_range, cmap, colorbar_label) for the given
    heatmap kind ("correlation" or "distance")."""
    if heatmap_kind == "distance":
        return "dist_matrix", (0.0, _SQRT2), "BuPu_r", "Correlation Distance (epsilon)"
    return "corr_matrix", (-1.0, 1.0), None, "Correlation"


def _heatmap_figsize(n: int) -> tuple:
    per_axis = max(6.0, n * 0.35)
    return (per_axis, per_axis)


def _dendrogram_figsize(n: int) -> tuple:
    return (max(7.0, n * 0.35), 6.0)


def _result_n(result: dict, default: int = 8) -> int:
    return len(result["symbols"]) if result.get("status") == "ok" else default


def _build_stacked_heatmap_figure(raw_result: dict, regressed_result: dict, *, heatmap_kind: str, regression_target: str):
    """Raw heatmap above Regressed heatmap, one figure, sized to fit both."""
    matrix_key, value_range, cmap, colorbar_label = _heatmap_style(heatmap_kind)

    def draw(ax, result):
        plot_correlation_heatmap(
            result[matrix_key], result["symbols"], ax=ax, show=False,
            annotate=True, value_range=value_range, cmap=cmap, colorbar_label=colorbar_label,
        )

    w_raw, h_raw = _heatmap_figsize(_result_n(raw_result))
    w_reg, h_reg = _heatmap_figsize(_result_n(regressed_result))

    fig = Figure(figsize=(max(w_raw, w_reg), h_raw + h_reg))
    ax_raw = fig.add_subplot(2, 1, 1)
    ax_reg = fig.add_subplot(2, 1, 2)

    _draw_result_or_status(ax_raw, raw_result, "Raw -- Universe", draw)
    _draw_result_or_status(ax_reg, regressed_result, f"Regressed ({regression_target}) -- Universe", draw)

    fig.tight_layout()
    return fig


def _build_side_by_side_dendrogram_figure(raw_result: dict, regressed_result: dict, *, epsilon: float, regression_target: str):
    """Raw dendrogram next to Regressed dendrogram, one figure."""

    def draw(ax, result):
        plot_dendrogram(
            result["dist_matrix"], result["symbols"], color_threshold=epsilon,
            ax=ax, show=False, extra_hlines=[_SQRT2_LINE], y_max=_DENDROGRAM_Y_MAX,
        )

    w_raw, h_raw = _dendrogram_figsize(_result_n(raw_result))
    w_reg, h_reg = _dendrogram_figsize(_result_n(regressed_result))

    fig = Figure(figsize=(w_raw + w_reg, max(h_raw, h_reg)))
    ax_raw = fig.add_subplot(1, 2, 1)
    ax_reg = fig.add_subplot(1, 2, 2)

    _draw_result_or_status(ax_raw, raw_result, "Raw -- Universe", draw)
    _draw_result_or_status(ax_reg, regressed_result, f"Regressed ({regression_target}) -- Universe", draw)

    fig.tight_layout()
    return fig


def _build_all_baskets_row_figure(result: dict, *, epsilon: float, heatmap_kind: str, row_title: str):
    """One basket/mode row: heatmap next to dendrogram, fixed size regardless
    of the basket's asset count (see _ALL_BASKETS_ROW_FIGSIZE)."""
    matrix_key, value_range, cmap, colorbar_label = _heatmap_style(heatmap_kind)

    def draw_heatmap(ax, res):
        plot_correlation_heatmap(
            res[matrix_key], res["symbols"], ax=ax, show=False,
            annotate=True, value_range=value_range, cmap=cmap, colorbar_label=colorbar_label,
        )

    def draw_dendrogram(ax, res):
        plot_dendrogram(
            res["dist_matrix"], res["symbols"], color_threshold=epsilon,
            ax=ax, show=False, extra_hlines=[_SQRT2_LINE], y_max=_DENDROGRAM_Y_MAX,
        )

    fig = Figure(figsize=_ALL_BASKETS_ROW_FIGSIZE)
    ax_heat = fig.add_subplot(1, 2, 1)
    ax_dendro = fig.add_subplot(1, 2, 2)

    _draw_result_or_status(ax_heat, result, f"{row_title}: Heatmap", draw_heatmap)
    _draw_result_or_status(ax_dendro, result, f"{row_title}: Dendrogram", draw_dendrogram)

    fig.tight_layout()
    return fig


class CanvasHost(ctk.CTkFrame):
    """Hosts a single matplotlib FigureCanvasTkAgg, replacing it whenever a
    new figure is rendered. Also wires button_press_event -> figure.pick()
    so that any pickable artist (e.g. a clickable legend line) actually
    fires pick_event -- matplotlib does NOT do this automatically."""

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        # NOTE: named _mpl_canvas / _mpl_canvas_widget, not _canvas -- ctk.CTkFrame
        # already uses self._canvas internally for its own background/border
        # rendering tk.Canvas; a plain "_canvas" attribute here silently shadows
        # it and crashes CTkFrame._draw() (AttributeError on winfo_exists()).
        self._mpl_canvas = None
        self._mpl_canvas_widget = None

    def show_figure(self, fig):
        self._clear_canvas()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.mpl_connect("button_press_event", lambda event: fig.pick(event))
        widget = canvas.get_tk_widget()
        widget.pack(expand=True, fill="both")
        self._mpl_canvas_widget = widget
        self._mpl_canvas = canvas
        return canvas

    def _clear_canvas(self) -> None:
        if self._mpl_canvas_widget is not None:
            self._mpl_canvas_widget.destroy()
            self._mpl_canvas_widget = None
            self._mpl_canvas = None


def _connect_legend_picking(canvas, legend_map: dict, hidden_labels: set) -> None:
    """hidden_labels is whatever set should record which lines are toggled
    off -- state.history_hidden_labels for the persisted Universe & Baskets
    chart, or a plain throwaway set() for charts that don't need their toggle
    state to survive a redraw."""
    # matplotlib's Artist.pick() traversal fires pick_event *twice* per real
    # click for a legend line (it's reachable from the artist tree via two
    # paths) -- without this guard a single click toggles visibility on then
    # immediately back off, i.e. it appears to do nothing.
    last_mouseevent_id = [None]

    def _on_pick(event):
        mouseevent_id = id(event.mouseevent)
        if mouseevent_id == last_mouseevent_id[0]:
            return
        last_mouseevent_id[0] = mouseevent_id

        legend_line = event.artist
        original_line = legend_map.get(legend_line)
        if original_line is None:
            return
        visible = not original_line.get_visible()
        original_line.set_visible(visible)
        legend_line.set_alpha(1.0 if visible else 0.2)

        label = original_line.get_label()
        if visible:
            hidden_labels.discard(label)
        else:
            hidden_labels.add(label)

        canvas.draw_idle()

    canvas.mpl_connect("pick_event", _on_pick)


class AllBasketsSection(ctk.CTkScrollableFrame):
    """Scrollable list of per-basket groups, separated by a horizontal rule.
    Each group has a Raw row and a Regressed row (always both), each row
    combining heatmap + dendrogram into one fixed-size figure."""

    def __init__(self, master):
        super().__init__(master, label_text="")
        self._children_to_clear: list = []

    def render(self, state, per_basket_results: list) -> None:
        for widget in self._children_to_clear:
            widget.destroy()
        self._children_to_clear = []

        for i, entry in enumerate(per_basket_results):
            label = entry["label"]
            raw_result = entry["raw"]
            regressed_result = entry["regressed"]

            if i > 0:
                divider = ctk.CTkFrame(self, height=2, fg_color=("gray70", "gray30"))
                divider.pack(fill="x", pady=(8, 16))
                self._children_to_clear.append(divider)

            group = ctk.CTkFrame(self, fg_color="transparent")
            group.pack(fill="x", pady=(0, 8))
            self._children_to_clear.append(group)

            header_text = label
            if raw_result.get("status") == "ok":
                header_text += f" ({len(raw_result['symbols'])} assets)"
            ctk.CTkLabel(
                group, text=header_text, font=ctk.CTkFont(weight="bold", size=14)
            ).pack(anchor="w", pady=(4, 4), padx=4)

            raw_host = CanvasHost(group)
            raw_host.pack(fill="x", pady=(0, 8))
            raw_host.show_figure(
                _build_all_baskets_row_figure(
                    raw_result, epsilon=state.epsilon, heatmap_kind=state.heatmap_kind, row_title="Raw",
                )
            )

            regressed_host = CanvasHost(group)
            regressed_host.pack(fill="x", pady=(0, 8))
            regressed_host.show_figure(
                _build_all_baskets_row_figure(
                    regressed_result, epsilon=state.epsilon, heatmap_kind=state.heatmap_kind,
                    row_title=f"Regressed ({state.regression_target})",
                )
            )


def _cluster_tab_scales(cut_result: dict, labels: list) -> tuple:
    """Global max n_clusters / n_assets across every basket, mode, and the
    per-mode Sum series -- used as the shared y-axis ceiling / histogram
    x-range for every plot of that kind in the Clusters tab."""
    all_clusters: list = []
    all_assets: list = []

    for label in labels:
        for mode in ("raw", "regressed"):
            df = cut_result.get((label, mode))
            if df is not None:
                all_clusters.extend(df["n_clusters"].tolist())
                all_assets.extend(df["n_assets"].tolist())

    for mode in ("raw", "regressed"):
        clusters_wide = build_wide_metric_df(cut_result, labels=labels, mode=mode, metric="n_clusters")
        if "Sum" in clusters_wide.columns:
            all_clusters.extend(clusters_wide["Sum"].tolist())
        assets_wide = build_wide_metric_df(cut_result, labels=labels, mode=mode, metric="n_assets")
        if "Sum" in assets_wide.columns:
            all_assets.extend(assets_wide["Sum"].tolist())

    clusters_max = float(max(all_clusters)) if all_clusters else 1.0
    assets_max = float(max(all_assets)) if all_assets else 1.0
    return clusters_max, assets_max


def _per_basket_metric_df(cut_result: dict, label: str, metric: str) -> pd.DataFrame:
    raw_df = cut_result.get((label, "raw"))
    regressed_df = cut_result.get((label, "regressed"))
    return pd.DataFrame({
        "Raw": raw_df[metric] if raw_df is not None else pd.Series(dtype=float),
        "Regressed": regressed_df[metric] if regressed_df is not None else pd.Series(dtype=float),
    })


def _build_universe_cluster_superplot(cut_result: dict, labels: list, clusters_max: float, assets_max: float):
    """One figure, 3x2 grid: [clusters raw | assets raw] / [clusters regressed
    | assets regressed] / [clusters histogram | assets histogram] -- no
    separate widgets between panels, so there's no visible partitioning."""
    clusters_raw_wide = build_wide_metric_df(cut_result, labels=labels, mode="raw", metric="n_clusters")
    clusters_reg_wide = build_wide_metric_df(cut_result, labels=labels, mode="regressed", metric="n_clusters")
    assets_raw_wide = build_wide_metric_df(cut_result, labels=labels, mode="raw", metric="n_assets")
    assets_reg_wide = build_wide_metric_df(cut_result, labels=labels, mode="regressed", metric="n_assets")

    fig = Figure(figsize=(_CLUSTERS_CELL_FIGSIZE[0] * 2, _CLUSTERS_CELL_FIGSIZE[1] * 3), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, hspace=0.15, wspace=0.2)
    ax_clusters_raw = fig.add_subplot(gs[0, 0])
    ax_assets_raw = fig.add_subplot(gs[0, 1])
    ax_clusters_reg = fig.add_subplot(gs[1, 0])
    ax_assets_reg = fig.add_subplot(gs[1, 1])
    ax_clusters_hist = fig.add_subplot(gs[2, 0])
    ax_assets_hist = fig.add_subplot(gs[2, 1])

    legend_maps = [
        dashboard_plots.draw_universe_over_time_line_chart(
            ax_clusters_raw, clusters_raw_wide, labels=labels, hidden_labels=set(),
            title="Number of Clusters (Raw)", total_label="Sum", ylabel="Number of clusters", y_max=clusters_max,
        ),
        dashboard_plots.draw_universe_over_time_line_chart(
            ax_assets_raw, assets_raw_wide, labels=labels, hidden_labels=set(),
            title="Assets Remaining (Raw)", total_label="Sum", ylabel="Number of assets", y_max=assets_max,
        ),
        dashboard_plots.draw_universe_over_time_line_chart(
            ax_clusters_reg, clusters_reg_wide, labels=labels, hidden_labels=set(),
            title="Number of Clusters (Regressed)", total_label="Sum", ylabel="Number of clusters", y_max=clusters_max,
        ),
        dashboard_plots.draw_universe_over_time_line_chart(
            ax_assets_reg, assets_reg_wide, labels=labels, hidden_labels=set(),
            title="Assets Remaining (Regressed)", total_label="Sum", ylabel="Number of assets", y_max=assets_max,
        ),
    ]

    dashboard_plots.draw_overlaid_histogram(
        ax_clusters_hist, clusters_raw_wide.get("Sum"), clusters_reg_wide.get("Sum"),
        x_max=clusters_max, title="Number of Clusters Distribution (All Time)", xlabel="Number of clusters",
    )
    dashboard_plots.draw_overlaid_histogram(
        ax_assets_hist, assets_raw_wide.get("Sum"), assets_reg_wide.get("Sum"),
        x_max=assets_max, title="Assets Remaining Distribution (All Time)", xlabel="Number of assets",
    )

    return fig, legend_maps


def _build_basket_cluster_superplot(cut_result: dict, label: str, clusters_max: float, assets_max: float):
    """One figure, 2x2 grid: [clusters (raw vs regressed) | assets (raw vs
    regressed)] / [clusters histogram | assets histogram] -- same fixed
    per-panel size as the Universe superplot (see _CLUSTERS_CELL_FIGSIZE)."""
    clusters_df = _per_basket_metric_df(cut_result, label, "n_clusters")
    assets_df = _per_basket_metric_df(cut_result, label, "n_assets")

    fig = Figure(figsize=(_CLUSTERS_CELL_FIGSIZE[0] * 2, _CLUSTERS_CELL_FIGSIZE[1] * 2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.2)
    ax_clusters = fig.add_subplot(gs[0, 0])
    ax_assets = fig.add_subplot(gs[0, 1])
    ax_clusters_hist = fig.add_subplot(gs[1, 0])
    ax_assets_hist = fig.add_subplot(gs[1, 1])

    legend_maps = [
        dashboard_plots.draw_universe_over_time_line_chart(
            ax_clusters, clusters_df, labels=["Raw", "Regressed"], hidden_labels=set(),
            title=f"Number of Clusters -- {label}", total_label=None, ylabel="Number of clusters", y_max=clusters_max,
        ),
        dashboard_plots.draw_universe_over_time_line_chart(
            ax_assets, assets_df, labels=["Raw", "Regressed"], hidden_labels=set(),
            title=f"Assets Remaining -- {label}", total_label=None, ylabel="Number of assets", y_max=assets_max,
        ),
    ]

    dashboard_plots.draw_overlaid_histogram(
        ax_clusters_hist, clusters_df.get("Raw"), clusters_df.get("Regressed"),
        x_max=clusters_max, title=f"Number of Clusters Distribution -- {label}", xlabel="Number of clusters",
    )
    dashboard_plots.draw_overlaid_histogram(
        ax_assets_hist, assets_df.get("Raw"), assets_df.get("Regressed"),
        x_max=assets_max, title=f"Assets Remaining Distribution -- {label}", xlabel="Number of assets",
    )

    return fig, legend_maps


class ClustersSection(ctk.CTkScrollableFrame):
    """Scrollable Clusters tab: a Universe-level overview first (one merged
    superplot), then one merged superplot per basket, separated by a
    horizontal rule like All Baskets. Every panel of a given kind (clusters
    line chart, assets line chart, clusters histogram, assets histogram)
    shares one fixed size and scale so they're directly comparable."""

    def __init__(self, master):
        super().__init__(master, label_text="")
        self._children_to_clear: list = []

    def render(self, state, cut_result: dict | None) -> None:
        for widget in self._children_to_clear:
            widget.destroy()
        self._children_to_clear = []

        if cut_result is None:
            placeholder = ctk.CTkLabel(
                self, text="Not computed yet -- click 'Recompute Clusters' in the sidebar.",
            )
            placeholder.pack(pady=40)
            self._children_to_clear.append(placeholder)
            return

        labels = state.basket_labels
        clusters_max, assets_max = _cluster_tab_scales(cut_result, labels)

        universe_header = ctk.CTkLabel(self, text="Universe", font=ctk.CTkFont(weight="bold", size=14))
        universe_header.pack(anchor="w", pady=(4, 4), padx=4)
        self._children_to_clear.append(universe_header)

        universe_host = CanvasHost(self)
        universe_host.pack(fill="x", pady=(0, 8))
        self._children_to_clear.append(universe_host)
        universe_fig, universe_legend_maps = _build_universe_cluster_superplot(
            cut_result, labels, clusters_max, assets_max
        )
        universe_canvas = universe_host.show_figure(universe_fig)
        for legend_map in universe_legend_maps:
            if legend_map and universe_canvas is not None:
                _connect_legend_picking(universe_canvas, legend_map, set())

        for label in labels:
            divider = ctk.CTkFrame(self, height=2, fg_color=("gray70", "gray30"))
            divider.pack(fill="x", pady=(8, 16))
            self._children_to_clear.append(divider)

            basket_header = ctk.CTkLabel(self, text=label, font=ctk.CTkFont(weight="bold", size=14))
            basket_header.pack(anchor="w", pady=(4, 4), padx=4)
            self._children_to_clear.append(basket_header)

            basket_host = CanvasHost(self)
            basket_host.pack(fill="x", pady=(0, 8))
            self._children_to_clear.append(basket_host)
            basket_fig, basket_legend_maps = _build_basket_cluster_superplot(
                cut_result, label, clusters_max, assets_max
            )
            basket_canvas = basket_host.show_figure(basket_fig)
            for legend_map in basket_legend_maps:
                if legend_map and basket_canvas is not None:
                    _connect_legend_picking(basket_canvas, legend_map, set())


class ScrollableSingleCanvas(ctk.CTkScrollableFrame):
    """A CTkScrollableFrame hosting exactly one CanvasHost -- used where a
    single (possibly tall) figure needs to be scrollable."""

    def __init__(self, master):
        super().__init__(master, label_text="")
        self.host = CanvasHost(self)
        self.host.pack(fill="both", expand=True)


class PlotTabs(ctk.CTkTabview):
    def __init__(self, master, *, command=None):
        super().__init__(master, command=command)
        self.add("Universe & Baskets")
        self.add("Heatmap")
        self.add("Dendrogram")
        self.add("All Baskets")
        self.add("Clusters")

        self.overview_host = CanvasHost(self.tab("Universe & Baskets"))
        self.overview_host.pack(expand=True, fill="both")

        self.heatmap_section = ScrollableSingleCanvas(self.tab("Heatmap"))
        self.heatmap_section.pack(expand=True, fill="both")

        self.dendrogram_host = CanvasHost(self.tab("Dendrogram"))
        self.dendrogram_host.pack(expand=True, fill="both")

        self.all_baskets_section = AllBasketsSection(self.tab("All Baskets"))
        self.all_baskets_section.pack(expand=True, fill="both")

        self.clusters_section = ClustersSection(self.tab("Clusters"))
        self.clusters_section.pack(expand=True, fill="both")

    # ------------------------------------------------------------------ #
    def render_heatmap(self, state, raw_result: dict, regressed_result: dict) -> None:
        fig = _build_stacked_heatmap_figure(
            raw_result, regressed_result, heatmap_kind=state.heatmap_kind, regression_target=state.regression_target,
        )
        self.heatmap_section.host.show_figure(fig)

    # ------------------------------------------------------------------ #
    def render_dendrogram(self, state, raw_result: dict, regressed_result: dict) -> None:
        fig = _build_side_by_side_dendrogram_figure(
            raw_result, regressed_result, epsilon=state.epsilon, regression_target=state.regression_target,
        )
        self.dendrogram_host.show_figure(fig)

    # ------------------------------------------------------------------ #
    def render_universe_and_baskets(self, state, counts: dict, history_df) -> None:
        fig = Figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.35, wspace=0.25)
        ax_bar = fig.add_subplot(gs[0, 0])
        ax_box = fig.add_subplot(gs[0, 1])
        ax_line = fig.add_subplot(gs[1, :])

        max_asset_count = None
        if history_df is not None and not history_df.empty:
            basket_columns = [c for c in history_df.columns if c != "Universe"]
            if basket_columns:
                max_asset_count = int(history_df[basket_columns].to_numpy().max())

        dashboard_plots.draw_basket_size_bar_chart(
            ax_bar, counts, labels=state.basket_labels, selected_date=state.date,
            max_asset_count=max_asset_count,
        )
        dashboard_plots.draw_basket_size_boxplot(ax_box, history_df, labels=state.basket_labels)
        legend_map = dashboard_plots.draw_universe_over_time_line_chart(
            ax_line, history_df, labels=state.history_labels, hidden_labels=state.history_hidden_labels,
        )

        canvas = self.overview_host.show_figure(fig)
        if legend_map and canvas is not None:
            _connect_legend_picking(canvas, legend_map, state.history_hidden_labels)

    # ------------------------------------------------------------------ #
    def render_all_baskets(self, state, per_basket_results: list) -> None:
        self.all_baskets_section.render(state, per_basket_results)

    # ------------------------------------------------------------------ #
    def render_clusters(self, state, cut_result: dict | None) -> None:
        self.clusters_section.render(state, cut_result)
