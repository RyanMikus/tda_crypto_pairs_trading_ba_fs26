"""Matplotlib figure builders for the two dashboard-only plots (heatmap and
dendrogram reuse cloudvisualisation.py directly). Every function draws into a
caller-supplied Axes and never calls plt.show()/plt.subplots -- the caller
(widgets/plot_tabs.py) owns the Figure/Canvas embedded in Tk."""
from __future__ import annotations

import numpy as np
import pandas as pd


def draw_basket_size_bar_chart(
    ax,
    counts: dict,
    *,
    labels: list,
    selected_date: pd.Timestamp,
    max_asset_count: int | None = None,
) -> None:
    ax.clear()
    values = [counts.get(label, 0) for label in labels]
    total_assets = sum(values)

    ax.bar(labels, values)
    ax.set_title(f"Basket Sizes on {selected_date.date()} ({total_assets} assets total)")
    ax.set_xlabel("Basket")
    ax.set_ylabel("Number of assets")
    if max_asset_count:
        ax.set_ylim(0, max_asset_count * 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def draw_basket_size_boxplot(ax, history_df: pd.DataFrame, *, labels: list) -> None:
    """Boxplot of each basket's daily asset count across the full scanned
    history (same history_df as the line chart) -- shows the typical size
    range of each basket over all of time, not just the selected date."""
    ax.clear()
    if history_df is None or history_df.empty:
        ax.text(
            0.5, 0.5, "No data yet.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    valid_labels = [label for label in labels if label in history_df.columns]
    data = [history_df[label].dropna().to_numpy() for label in valid_labels]

    ax.boxplot(data, labels=valid_labels)
    ax.set_title("Basket Size Distribution (All Time)")
    ax.set_xlabel("Basket")
    ax.set_ylabel("Number of assets")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def draw_universe_over_time_line_chart(
    ax, history_df: pd.DataFrame, *, labels: list, hidden_labels: set | None = None,
    title: str = "Assets in Universe Over Time", total_label: str = "Universe",
    ylabel: str = "Number of assets", y_max: float | None = None,
):
    """Draws one line per label plus a clickable legend (click an entry to
    toggle that line's visibility -- see widgets/plot_tabs.py, which wires the
    pick_event). Returns a dict mapping each legend line artist to its
    corresponding plotted line, or None if there was no data to plot.

    All lines are always drawn (never skipped) so the legend always lists
    every label; hidden_labels only controls each line's initial visibility.

    total_label identifies which column gets the distinct dashed/bold total
    styling (the basket-count "Universe" total, or a "Sum" column for the
    Clusters tab). y_max, if given, overrides the auto-computed ceiling --
    used to share one fixed scale across several charts of the same kind.
    """
    ax.clear()
    if history_df is None or history_df.empty:
        ax.text(
            0.5, 0.5, "No data yet.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return None

    hidden_labels = hidden_labels or set()
    lines_by_label = {}
    for label in labels:
        if label not in history_df.columns:
            continue
        if label == total_label:
            (line,) = ax.plot(
                history_df.index, history_df[label], label=label,
                linewidth=1.5, linestyle="--", color="white",
            )
        else:
            (line,) = ax.plot(history_df.index, history_df[label], label=label, linewidth=1)
        line.set_visible(label not in hidden_labels)
        lines_by_label[label] = line

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Fixed y-ceiling (either the caller's shared scale, or this history's own
    # full-range max) so toggling line visibility on/off doesn't rescale the axis.
    if y_max is None:
        y_max = float(history_df.to_numpy().max())
    ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
    ax.grid(alpha=0.25)

    if not lines_by_label:
        return None

    legend = ax.legend(frameon=False, fontsize=8)
    legend_map = {}
    for legend_line, label in zip(legend.get_lines(), lines_by_label.keys()):
        original_line = lines_by_label[label]
        legend_line.set_picker(8)
        legend_line.set_alpha(1.0 if original_line.get_visible() else 0.2)
        legend_map[legend_line] = original_line

    return legend_map


def draw_overlaid_histogram(
    ax, raw_series: pd.Series, regressed_series: pd.Series, *,
    x_max: float, title: str, xlabel: str,
) -> None:
    """Histogram of raw vs. regressed values for the same metric, overlaid in
    the same axes with transparency so both distributions are visible at
    once. x_max is a shared value across every histogram of this "kind"
    (e.g. every clusters-histogram) so they stay visually comparable."""
    ax.clear()

    raw_values = raw_series.dropna().to_numpy() if raw_series is not None else None
    regressed_values = regressed_series.dropna().to_numpy() if regressed_series is not None else None

    if (raw_values is None or raw_values.size == 0) and (regressed_values is None or regressed_values.size == 0):
        ax.text(
            0.5, 0.5, "No data yet.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    bins = np.linspace(0, x_max if x_max > 0 else 1, 21)
    if raw_values is not None and raw_values.size:
        ax.hist(raw_values, bins=bins, alpha=0.55, label="Raw", color="#2a78d6")
    if regressed_values is not None and regressed_values.size:
        ax.hist(regressed_values, bins=bins, alpha=0.55, label="Regressed", color="#eb6834")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False, fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
