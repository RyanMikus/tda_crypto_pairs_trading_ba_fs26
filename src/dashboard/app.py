"""Root window for the TDA basket & clustering dashboard. Wires sidebar
controls to DashboardState, orchestrates recompute-on-change via
CloudComputeService, and hands results to the tab views for rendering.

Rendering is deferred per-tab: a control change marks every affected tab
"dirty" with the render call it would need, but only actually calls it for
whichever tab is currently visible. Switching tabs fires the deferred render
for the tab just switched to. This is what keeps the (relatively expensive,
many-plot) All Baskets tab from being rebuilt on every date/epsilon tick
while the user is looking at a different tab.

Run with: python -m src.dashboard.app
"""
from __future__ import annotations

from pathlib import Path

import customtkinter as ctk

from . import _bootstrap  # noqa: F401
from .state import DashboardState, clamp_date
from .data_loader import load_app_data
from .basket_model import BasketConfig, build_baskets_for_date
from .compute import CloudComputeService, CloudRequest, compute_basket_history, compute_cluster_history, cut_cluster_history
from .workers import BackgroundJobRunner
from .widgets.sidebar import Sidebar
from .widgets.plot_tabs import PlotTabs

DEBOUNCE_MS = 250
UNIVERSE_SCOPE = "Universe"


class DashboardApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TDA Basket & Clustering Dashboard")
        self.geometry("1400x900")

        self._debounce_after_id = None
        self._history_cache: dict = {}
        self._cluster_history_cache: dict = {}
        self._dirty_tabs: dict = {}  # tab_name -> pending render callable

        self._loading_label = ctk.CTkLabel(self, text="Loading data...", font=ctk.CTkFont(size=16))
        self._loading_label.pack(expand=True, fill="both")
        self.after(50, self._load_data)

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #
    def _load_data(self) -> None:
        def _progress(message: str) -> None:
            self._loading_label.configure(text=message)
            self.update_idletasks()

        try:
            self.data = load_app_data(progress_callback=_progress)
        except Exception as exc:  # noqa: BLE001 -- surfaced in the UI, first-run data issues are common
            self._loading_label.configure(text=f"Failed to load data:\n{exc}")
            return

        self._loading_label.destroy()

        self.service = CloudComputeService(self.data)
        self.jobs = BackgroundJobRunner(self)
        self.cluster_jobs = BackgroundJobRunner(self)  # separate from self.jobs -- unrelated background scans
        # NOTE: named dashboard_state, not state -- ctk.CTk (tkinter.Tk) already
        # defines a state() method for window manager state (normal/iconic/zoomed);
        # customtkinter's internal DPI-scaling tracker calls window.state() and
        # would silently break if a plain "state" attribute shadowed it.
        self.dashboard_state = DashboardState(date=self.data.data_max)

        self._build_ui()
        self._request_recompute(debounce=False)
        # Proactively populate the basket-count history so the merged
        # Universe & Baskets tab's boxplot/line chart and the bar chart's
        # fixed y-ceiling are available from the start.
        self.recompute_history()
        # Queues the Clusters tab's "not computed yet" placeholder even if the
        # user opens that tab before touching any other control.
        self._refresh_clusters_view()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = Sidebar(self, app=self, data=self.data, state=self.dashboard_state)
        self.sidebar.grid(row=0, column=0, sticky="ns", padx=8, pady=8)

        self.tabs = PlotTabs(self, command=self._on_tab_changed)
        self.tabs.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

    # ------------------------------------------------------------------ #
    # Lazy per-tab rendering
    # ------------------------------------------------------------------ #
    def _render_if_visible(self, tab_name: str, render_fn) -> None:
        if self.tabs.get() == tab_name:
            render_fn()
        else:
            self._dirty_tabs[tab_name] = render_fn

    def _on_tab_changed(self) -> None:
        current = self.tabs.get()
        render_fn = self._dirty_tabs.pop(current, None)
        if render_fn is not None:
            render_fn()

    # ------------------------------------------------------------------ #
    # State change entrypoints (called by Sidebar)
    # ------------------------------------------------------------------ #
    def on_control_changed(self, *, debounce: bool = True) -> None:
        self.sidebar.refresh_bounds(self.data, self.dashboard_state)
        self._request_recompute(debounce=debounce)
        # Cheap when basket_lookback_days/basket_upper_bounds are unchanged (cache
        # hit, just a redraw); only spawns a background scan on a genuine change.
        self.recompute_history()
        # Cheap when the (lookback/bounds/regression_target) key is already
        # cached -- just re-cuts the cached linkages at the current epsilon and
        # redraws; only a genuine change to that key requires a fresh
        # "Recompute Clusters" click (no scan runs from here).
        self._refresh_clusters_view()

    def _request_recompute(self, *, debounce: bool = True) -> None:
        if self._debounce_after_id is not None:
            self.after_cancel(self._debounce_after_id)
            self._debounce_after_id = None

        if debounce:
            self._debounce_after_id = self.after(DEBOUNCE_MS, self._recompute_now)
        else:
            self._recompute_now()

    def _build_result_pair(self, scope: str):
        """Builds both the raw and regressed result for a scope -- always
        both, since Raw and Regressed are shown together everywhere now."""
        state = self.dashboard_state

        raw_req = CloudRequest(
            date=state.date, basket_lookback_days=state.basket_lookback_days,
            corr_lookback_days=state.corr_lookback_days,
            basket_upper_bounds=state.basket_upper_bounds,
            scope=scope, regression_target=None,
        )
        raw_result = self.service.build(raw_req)

        regressed_req = CloudRequest(
            date=state.date, basket_lookback_days=state.basket_lookback_days,
            corr_lookback_days=state.corr_lookback_days,
            basket_upper_bounds=state.basket_upper_bounds,
            scope=scope, regression_target=state.regression_target,
        )
        regressed_result = self.service.build(regressed_req)

        return raw_result, regressed_result

    def _recompute_now(self) -> None:
        self._debounce_after_id = None
        state = self.dashboard_state

        clamped = clamp_date(
            state.date, self.data.data_min, self.data.data_max,
            state.basket_lookback_days, state.corr_lookback_days,
        )
        if clamped != state.date:
            state.date = clamped
            self.sidebar.sync_date(state.date)

        raw_result, regressed_result = self._build_result_pair(UNIVERSE_SCOPE)
        self._render_if_visible("Heatmap", lambda: self.tabs.render_heatmap(state, raw_result, regressed_result))
        self._render_if_visible(
            "Dendrogram", lambda: self.tabs.render_dendrogram(state, raw_result, regressed_result)
        )

        def _compute_and_render_all_baskets() -> None:
            # Deferred entirely (not just the render) -- building raw+regressed
            # point clouds/correlations for every basket is real compute work,
            # not just a plotting cost. Skipping it whenever this tab isn't the
            # visible one is what actually fixes the lag on other tabs; state
            # is read fresh here, so switching to this tab later still reflects
            # whatever the current controls are at that moment.
            per_basket_results = []
            for label in state.basket_labels:
                label_raw_result, label_regressed_result = self._build_result_pair(label)
                per_basket_results.append(
                    {"label": label, "raw": label_raw_result, "regressed": label_regressed_result}
                )
            self.tabs.render_all_baskets(state, per_basket_results)

        self._render_if_visible("All Baskets", _compute_and_render_all_baskets)

        self._render_universe_and_baskets_tab()

    def _render_universe_and_baskets_tab(self) -> None:
        state = self.dashboard_state
        rolling_volume_df = self.service.rolling_volume(state.basket_lookback_days)
        basket_config = BasketConfig(upper_bounds=state.basket_upper_bounds)
        basket_df = build_baskets_for_date(rolling_volume_df, state.date, basket_config)
        counts = basket_df["basket"].value_counts().to_dict()

        history_df = self._history_cache.get(self._history_key())

        self._render_if_visible(
            "Universe & Baskets", lambda: self.tabs.render_universe_and_baskets(state, counts, history_df)
        )

    # ------------------------------------------------------------------ #
    # Universe-over-time (background scan)
    # ------------------------------------------------------------------ #
    def _history_key(self):
        return (self.dashboard_state.basket_lookback_days, self.dashboard_state.basket_upper_bounds)

    def recompute_history(self) -> None:
        key = self._history_key()
        if key in self._history_cache:
            self._render_universe_and_baskets_tab()
            return

        basket_lookback_days = self.dashboard_state.basket_lookback_days
        basket_upper_bounds = self.dashboard_state.basket_upper_bounds

        def _work(progress_cb, cancel_check):
            return compute_basket_history(
                self.service, basket_lookback_days, basket_upper_bounds,
                progress_callback=progress_cb, cancel_check=cancel_check,
            )

        def _on_done(history_df):
            self._history_cache[key] = history_df
            self._render_universe_and_baskets_tab()

        def _on_error(exc):
            print(f"Universe Over Time recompute failed: {exc}")

        self.jobs.start(_work, on_done=_on_done, on_error=_on_error)

    # ------------------------------------------------------------------ #
    # Clusters (background scan; epsilon changes only re-cut cached results)
    # ------------------------------------------------------------------ #
    def _cluster_history_key(self):
        state = self.dashboard_state
        return (state.basket_lookback_days, state.corr_lookback_days, state.basket_upper_bounds, state.regression_target)

    def _refresh_clusters_view(self) -> None:
        state = self.dashboard_state
        cluster_history = self._cluster_history_cache.get(self._cluster_history_key())
        if cluster_history is None:
            self._render_if_visible("Clusters", lambda: self.tabs.render_clusters(state, None))
            return

        cut_result = cut_cluster_history(cluster_history, epsilon=state.epsilon)
        self._render_if_visible("Clusters", lambda: self.tabs.render_clusters(state, cut_result))

    def recompute_clusters(self) -> None:
        key = self._cluster_history_key()
        if key in self._cluster_history_cache:
            self._refresh_clusters_view()
            return

        state = self.dashboard_state
        basket_lookback_days = state.basket_lookback_days
        corr_lookback_days = state.corr_lookback_days
        basket_upper_bounds = state.basket_upper_bounds
        regression_target = state.regression_target

        def _work(progress_cb, cancel_check):
            return compute_cluster_history(
                self.service, self.data, basket_lookback_days, corr_lookback_days,
                basket_upper_bounds, regression_target,
                progress_callback=progress_cb, cancel_check=cancel_check,
            )

        def _on_progress(done, total):
            self.sidebar.set_clusters_progress(done, total)

        def _on_done(cluster_history):
            self._cluster_history_cache[key] = cluster_history
            # Actual dates scanned, not a hardcoded (1, 1) -- every (label, mode)
            # entry covers the same date range, so any one of them gives the count.
            scanned = len(next(iter(cluster_history.values()))) if cluster_history else 0
            self.sidebar.set_clusters_progress(scanned, scanned)
            self._refresh_clusters_view()

        def _on_error(exc):
            self.sidebar.set_clusters_error(str(exc))

        self.sidebar.set_clusters_progress(0, 1)
        self.cluster_jobs.start(_work, on_progress=_on_progress, on_done=_on_done, on_error=_on_error)


def main() -> None:
    # Dark mode, always -- both the customtkinter chrome and every embedded
    # matplotlib figure. matplotlib style is a global rcParams mutation, set
    # once here (the dashboard's own process) before any Figure is created;
    # it does NOT affect the notebooks, which run in a separate process/kernel
    # and never import this module.
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")

    ctk.set_appearance_mode("dark")
    # Custom black + violet (RGB 128,37,93 = #80255D) theme so the app chrome
    # matches the matplotlib figures' black background instead of ctk's
    # default dark-grey/blue theme.
    theme_path = Path(__file__).with_name("theme.json")
    ctk.set_default_color_theme(str(theme_path))
    app = DashboardApp()
    app.mainloop()


if __name__ == "__main__":
    main()
