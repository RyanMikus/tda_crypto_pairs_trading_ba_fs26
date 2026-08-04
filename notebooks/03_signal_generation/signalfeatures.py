"""Forward-Return-Screening fuer Signal-Features: wiederverwendbarer Baukasten.

Zweck: bevor eine echte Entry/Exit-Regel gebaut wird, schnell pruefen, ob ein
Kandidaten-Feature (Z-Score, vola-adjustierter Spread, ...) ueberhaupt mit dem
zukuenftigen Return zusammenhaengt -- eine Korrelationskurve statt einer
vollen Backtest-Infrastruktur pro Kandidat.

Zwei getrennte Zeitschichten (wichtig fuer das Verstaendnis des ganzen Moduls):

    1. Clique-DISCOVERY: einmal pro Kalendertag, mit den in 02_tda_clustering
       kalibrierten Parametern (BEST_LOOKBACK/THRESHOLD/EPSILON_CAP). Teuer
       (volle Vietoris-Rips/Persistenz-Berechnung), daher NICHT Teil dieses
       Moduls -- das laeuft einmalig in 02_tda_clustering/05_clique_cache.ipynb
       (HelperFunctions.discover_cliques_over_dates) und wird als Parquet-
       Datei gecacht. Das ist auch der Grund, warum der spaetere Structural-
       Break-Monitor (Additional-Info-PDF, "Second Phase") existiert: die
       Clique-MITGLIEDSCHAFT einfrieren statt sie neu zu entdecken.
    2. Signal-LAYER (dieses Modul): liest den Clique-Cache ein und arbeitet
       INNERHALB eines Discovery-Tages auf 5-Min-Kerzen-Basis -- sowohl das
       Feature (z.B. Z-Score) als auch die Forward-Returns. Ein Trade darf
       ueber Mitternacht hinaus laufen (die Forward-Return-Fenster sind nicht
       an den Kalendertag gebunden), nur die Clique-Mitgliedschaft selbst
       bleibt fuer den ganzen Discovery-Tag fix.

Kernidee, angepasst an die geclusterte Struktur aus 02_tda_clustering: es gibt
keine einzelne durchgehende Zeitreihe pro Asset (welche Assets ueberhaupt
zusammen betrachtet werden, aendert sich taeglich), daher wird ein LANGES
PANEL ueber alle (Kerze, Basket, Asset-in-Clique)-Beobachtungen aufgebaut.

Baukasten-Teile:
    - CliqueContext: alles, was ein Feature ueber eine konkrete Clique an
      einer konkreten Kerze wissen darf.
    - Feature-Funktionen (Signatur: CliqueContext -> pd.Series) + FEATURE_REGISTRY,
      um neue Kandidaten einzuhaengen ohne collect_screening_panel anzufassen.
    - collect_screening_panel: baut das lange Panel, vektorisiert ueber ALLE
      Horizonte auf einmal (keine Python-Schleife pro Horizont) -- damit ist
      eine dichte Horizont-Achse (jede einzelne Kerze bis zu einem Maximum)
      bezahlbar, obwohl sie sonst die Zeilenzahl sprengen wuerde.
    - correlation_curve: Punkt-Schaetzer der Korrelation PRO Horizont-Kerze
      (fuer die dichte Kurve) -- bewusst ohne Signifikanztest, Beurteilung
      passiert visuell im Plot.
    - plot_correlation_curves: Linienplot, eine Linie je Feature.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================================
# Kontext, den jede Feature-Funktion sieht
# ============================================================================

@dataclass
class CliqueContext:
    """Alles, was ein Feature ueber eine konkrete Clique an einer Kerze braucht.

    Neue Felder bei Bedarf ergaenzen (z.B. ein Momentum-Fenster) -- bestehende
    Feature-Funktionen bleiben davon unberuehrt, sie lesen nur die Felder,
    die sie kennen.
    """
    timestamp: pd.Timestamp    # exakter 5-Min-Kerzen-Zeitstempel dieser Beobachtung
    discovery_date: pd.Timestamp  # Kalendertag, an dem diese Clique gefunden wurde
    basket: str
    asset_ids: list
    asset_syms: list
    candle_return: pd.Series   # index=asset_syms, Return DIESER Kerze (roh oder residualisiert)
    trailing_vol: pd.Series    # index=asset_syms, rollierende Kerzen-Vola VOR "jetzt" (kausal)
    persistence: float         # Persistenz der Clique (death - birth, gedeckelt bei epsilon_cap)
    lookback_days: int         # topologischer Lookback, mit dem diese Clique gefunden wurde


# ============================================================================
# Feature-Funktionen: Signatur CliqueContext -> pd.Series (index=asset_syms)
# ============================================================================

def feature_zscore(ctx: CliqueContext) -> pd.Series:
    """Cross-sektionaler Z-Score des Kerzen-Returns innerhalb der Clique.

    Entspricht der PDF-Referenzlogik (Haupt-PDF Abschnitt 4.2/5): mu/sigma aus
    den Returns ALLER Clique-Mitglieder an dieser einen Kerze, nicht aus der
    Historie eines einzelnen Assets.
    """
    r = ctx.candle_return
    sigma = r.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.nan, index=r.index)
    return (r - r.mean()) / sigma


FEATURE_REGISTRY: dict = {
    "zscore": feature_zscore,
}
# Weitere Kandidaten hier eintragen, siehe Docstring von feature_zscore fuer
# die erwartete Signatur (CliqueContext -> pd.Series, index=asset_syms).
# ctx.persistence und ctx.trailing_vol stehen bereits zur Verfuegung, falls
# ein Feature die Clique-Persistenz oder die eigene Asset-Vola einbeziehen
# will (z.B. Z-Score * Persistenz als Konfidenz-Gewicht, oder Abweichung
# geteilt durch die eigene rollierende Vola statt der Cross-Sektions-Std der
# Clique -- naeher am Lin-et-al.-Prinzip aus der Additional-Info-PDF).


# ============================================================================
# Tagesreturn-/Vola-Hilfsfunktionen -- jetzt auf nativer 5-Min-Kerzen-Basis
# ============================================================================

def build_returns_matrix(ohlcv_df: pd.DataFrame, *, timestamp_column: str, asset_id_column: str, close_column: str) -> pd.DataFrame:
    """5-Min-Kerzen-Returnmatrix (Index=Zeitstempel, Spalten=asset_id), direkt
    aus den rohen Schlusskursen -- keine Tagesaggregation mehr. Basis fuer
    candle_return/trailing_vol/Forward-Return-Pfade unten."""
    price_matrix = ohlcv_df.pivot_table(
        index=timestamp_column, columns=asset_id_column, values=close_column, aggfunc="last",
    ).sort_index()
    return price_matrix.pct_change()


def build_trailing_vol(returns_df: pd.DataFrame, *, window_candles: int) -> pd.DataFrame:
    """Rollierende Kerzen-Return-Std ueber window_candles Kerzen, min_periods=window_candles
    (kausal: bei Kerze t nur Daten bis einschliesslich t-1 durch den Shift)."""
    return returns_df.shift(1).rolling(window_candles, min_periods=window_candles).std(ddof=0)


# ============================================================================
# Taeglich (bei Entry) eingefrorene Residualisierung, jetzt kerzenbasiert
# ============================================================================

def compute_frozen_betas(
    returns_df: pd.DataFrame,
    *,
    factor_asset_ids: list,
    target_cols: list,
    at_timestamp: pd.Timestamp,
    regression_lookback_candles: int,
) -> dict | None:
    """Schaetzt EINMAL das OLS-Beta jedes Ziel-Assets gegen factor_asset_ids,
    ueber die regression_lookback_candles Kerzen VOR at_timestamp (kausal,
    kein Blick auf at_timestamp selbst). Wird danach fuer den ganzen
    Discovery-Tag (alle Kerzen-Entries UND alle Forward-Horizonte) mit
    DEMSELBEN eingefrorenen Beta angewendet (kein Re-Fit pro Kerze) --
    konsistent mit dem "Beta bei Entry einfrieren"-Prinzip des Structural-
    Break-Monitors (Additional-Info-PDF, Second Phase).

    WICHTIG: factor_asset_ids/target_cols sind Asset-IDs (int), keine Symbole
    -- returns_df ist asset-ID-indiziert (siehe build_returns_matrix). Das
    ist eine andere Konvention als HelperFunctions.residualize_returns (dort
    sind factor_columns Symbol-Strings, weil die Punktwolken dort auf Symbol-
    Spalten arbeiten) -- nicht verwechseln.

    Returns:
        dict asset_col -> np.ndarray (Intercept + Faktor-Betas), oder None
        falls zu wenig Historie oder Faktor-NaNs im Fenster vorhanden sind.
    """
    idx = returns_df.index
    pos = idx.searchsorted(at_timestamp)
    if pos < regression_lookback_candles:
        return None

    window = returns_df.iloc[pos - regression_lookback_candles:pos]
    factors = window[factor_asset_ids].to_numpy()
    if np.isnan(factors).any():
        return None
    design = np.column_stack([np.ones(len(window)), factors])

    betas = {}
    for col in target_cols:
        y = window[col].to_numpy()
        if np.isnan(y).any():
            betas[col] = None
            continue
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        betas[col] = beta
    return betas


def _apply_frozen_betas_matrix(sub_returns: pd.DataFrame, *, factor_asset_ids: list, betas: dict) -> pd.DataFrame:
    """Wendet compute_frozen_betas' Ergebnis auf eine GANZE Kerzen-Spanne auf
    einmal an (vektorisiert ueber alle Zeilen, statt pro Kerze einzeln) --
    Residuum = tatsaechlicher Return minus faktor-erklaerter Return.
    Spalten ohne gueltiges Beta werden komplett NaN."""
    factor_vals = sub_returns[factor_asset_ids].to_numpy()
    design = np.column_stack([np.ones(len(sub_returns)), factor_vals])
    out = {}
    for col, beta in betas.items():
        if beta is None:
            out[col] = np.full(len(sub_returns), np.nan)
        else:
            out[col] = sub_returns[col].to_numpy() - design @ beta
    return pd.DataFrame(out, index=sub_returns.index)


# ============================================================================
# Vektorisierte, dichte Forward-Return-Pfade
# ============================================================================

def _forward_returns_dense(
    sub_matrix: pd.DataFrame,
    entry_positions: np.ndarray,
    horizons: np.ndarray,
    *,
    additive: bool,
) -> np.ndarray:
    """Kern der Vektorisierung: fuer JEDEN Entry gleichzeitig JEDEN Horizont
    berechnen, ohne Python-Schleife ueber Horizonte.

    sub_matrix: (T, n_assets) Returns (roh) oder Residuen (additive=True) --
        deckt entry_positions UND entry_positions.max() + horizons.max() ab.
    entry_positions: Positionen (in sub_matrix) der Entry-Kerzen.
    horizons: Kerzen-Offsets (>=1).
    additive=False (roh): kumulatives Produkt, Forward-Return = Verhaeltnis.
    additive=True (residualisiert): kumulative Summe, Forward-Return = Differenz
        (Approximation fuer kleine Kerzen-Returns -- fuers Screening per
        Korrelation ausreichend, keine praezise P&L-Rechnung).

    Returns:
        np.ndarray (n_entries, n_horizons, n_assets) -- Forward-Return relativ
        zur Clique (Cross-Sektions-Mittelwert je (Entry, Horizont) bereits
        abgezogen).
    """
    values = sub_matrix.to_numpy()
    cum = np.cumsum(values, axis=0) if additive else np.cumprod(1.0 + values, axis=0)

    idx_matrix = entry_positions[:, None] + horizons[None, :]  # (n_entries, n_horizons)
    future_vals = cum[idx_matrix]                              # (n_entries, n_horizons, n_assets)
    entry_vals = cum[entry_positions][:, None, :]               # (n_entries, 1, n_assets)

    fwd = (future_vals - entry_vals) if additive else (future_vals / entry_vals - 1.0)
    fwd_relative = fwd - np.nanmean(fwd, axis=-1, keepdims=True)
    return fwd_relative


# ============================================================================
# Panel-Sammlung
# ============================================================================

def collect_screening_panel(
    *,
    cliques_df: pd.DataFrame,
    id_to_symbol: dict,
    returns_df: pd.DataFrame,
    trailing_vol_df: pd.DataFrame,
    entry_stride_candles: int,
    horizons: np.ndarray,
    feature_fns: dict,
    residualize: bool = False,
    factor_asset_ids: list | None = None,
    regression_lookback_candles: int | None = None,
) -> pd.DataFrame:
    """Baut das lange Screening-Panel: eine Zeile pro (Entry-Kerze, Basket, Asset-in-Clique).

    Nutzt eine BEREITS BERECHNETE Clique-Discovery (siehe
    02_tda_clustering/05_clique_cache.ipynb + HelperFunctions.discover_cliques_over_dates)
    statt sie selbst zu wiederholen -- die teure Persistenzberechnung passiert
    dort einmalig, hier nur noch die vergleichsweise billige, vektorisierte
    Kerzen-/Feature-/Forward-Return-Berechnung (siehe _forward_returns_dense),
    was eine dichte Horizont-Achse (jede Kerze statt weniger Stuetzpunkte)
    ueberhaupt bezahlbar macht -- das macht diese Funktion schnell genug, um
    bei jeder neuen Feature-Idee neu zu laufen, ohne auf die Discovery zu warten.

    Args:
        cliques_df: DataFrame wie von HelperFunctions.discover_cliques_over_dates
            zurueckgegeben (discovery_date, basket, asset_ids, persistence,
            lookback_days) -- z.B. per pd.read_parquet aus dem Cache geladen
            und ggf. schon auf den gewuenschten Zeitraum gefiltert.
        returns_df / trailing_vol_df: 5-Min-Kerzen, Spalten=asset_id (siehe
            build_returns_matrix/build_trailing_vol) -- bei residualize=True
            sind das die ROHEN (nicht vorab residualisierten) Returns, die
            Residualisierung passiert hier pro Discovery-Tag mit eingefrorenem
            Beta.
        entry_stride_candles: nur jede N-te Kerze des Discovery-Tages wird als
            Entry verwendet (1 = jede Kerze -- Vorsicht, multipliziert die
            Zeilenzahl).
        horizons: Kerzen-Offsets (>=1), z.B. np.arange(1, 289) fuer einen
            ganzen Tag dicht. Forward-Fenster duerfen ueber Mitternacht
            hinausreichen (Tage, an denen der Horizont ueber das Ende von
            returns_df hinausreichen wuerde, werden stillschweigend uebersprungen).
    """
    if residualize and (factor_asset_ids is None or regression_lookback_candles is None):
        raise ValueError("residualize=True braucht factor_asset_ids und regression_lookback_candles")

    horizons = np.asarray(horizons, dtype=np.int64)
    max_horizon = int(horizons.max())
    ts_index = returns_df.index
    rows = []

    for clique_row in cliques_df.itertuples():
        label = clique_row.basket
        lb = clique_row.lookback_days
        clique_ids = list(clique_row.asset_ids)
        clique_syms = [id_to_symbol.get(a, f"ID:{a}") for a in clique_ids]

        day_start = pd.Timestamp(clique_row.discovery_date).normalize()
        day_end = day_start + pd.Timedelta(days=1)
        entry_ts = ts_index[(ts_index >= day_start) & (ts_index < day_end)][::entry_stride_candles]
        if len(entry_ts) == 0:
            continue

        first_entry_pos = ts_index.searchsorted(entry_ts[0])
        last_entry_pos = ts_index.searchsorted(entry_ts[-1])
        span_end_pos = last_entry_pos + max_horizon
        if span_end_pos >= len(ts_index):
            continue  # Horizont wuerde ueber das Ende der verfuegbaren Daten hinausreichen

        span_index = ts_index[first_entry_pos:span_end_pos + 1]
        entry_positions_in_span = ts_index.searchsorted(entry_ts) - first_entry_pos

        betas = None
        if residualize:
            betas = compute_frozen_betas(
                returns_df, factor_asset_ids=factor_asset_ids, target_cols=clique_ids,
                at_timestamp=day_start, regression_lookback_candles=regression_lookback_candles,
            )
            if betas is None:
                continue
            sub_processed = _apply_frozen_betas_matrix(
                returns_df.loc[span_index, list(set(clique_ids) | set(factor_asset_ids))],
                factor_asset_ids=factor_asset_ids, betas=betas,
            )[clique_ids]
            if sub_processed.isna().any().any():
                continue
        else:
            sub_processed = returns_df.loc[span_index, clique_ids]
            if sub_processed.isna().any().any():
                continue

        fwd_relative = _forward_returns_dense(
            sub_processed, entry_positions_in_span, horizons, additive=residualize,
        )  # (n_entries, n_horizons, n_assets)

        trailing_vol_at_entries = trailing_vol_df.loc[entry_ts, clique_ids]

        for i, ts in enumerate(entry_ts):
            candle_ret = sub_processed.iloc[entry_positions_in_span[i]]
            candle_ret = pd.Series(candle_ret.to_numpy(), index=clique_syms)
            if candle_ret.isna().any():
                continue

            tv = pd.Series(trailing_vol_at_entries.iloc[i].to_numpy(), index=clique_syms)

            ctx = CliqueContext(
                timestamp=ts, discovery_date=day_start, basket=label,
                asset_ids=clique_ids, asset_syms=clique_syms,
                candle_return=candle_ret, trailing_vol=tv,
                persistence=clique_row.persistence, lookback_days=lb,
            )
            feature_values = {name: fn(ctx) for name, fn in feature_fns.items()}

            for j, sym in enumerate(clique_syms):
                row = {
                    "timestamp": ts, "discovery_date": day_start, "basket": label,
                    "asset_id": clique_ids[j], "asset_sym": sym,
                    "persistence": clique_row.persistence, "clique_size": len(clique_ids),
                }
                for name in feature_fns:
                    row[name] = feature_values[name].get(sym, np.nan)
                for k, h in enumerate(horizons):
                    row[f"fwd_ret_{h}c"] = fwd_relative[i, k, j]
                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Vergleich
# ============================================================================

def correlation_curve(
    panel_df: pd.DataFrame,
    feature_col: str,
    horizon_cols: list,
    *,
    method: str = "spearman",
    min_obs: int = 10,
) -> pd.DataFrame:
    """Punkt-Schaetzer der Korrelation PRO Horizont -- gedacht fuer eine
    DICHTE Horizont-Achse (viele Werte), um die ganze Kurve auf einen Blick
    zu sehen. Bewusst ohne Konfidenzband: die Kurvenform (faellt sie schnell
    auf 0 ab oder bleibt sie ueber viele Horizonte stabil negativ?) soll
    direkt im Plot beurteilt werden.

    Returns:
        DataFrame, Index=Horizont (aus den Spaltennamen extrahiert, z.B.
        "fwd_ret_12c" -> 12), Spalten point_estimate/n_obs.
    """
    rows = []
    for h_col in horizon_cols:
        sub = panel_df[[feature_col, h_col]].dropna()
        n = len(sub)
        h_value = int("".join(ch for ch in h_col if ch.isdigit()))
        r = sub[feature_col].corr(sub[h_col], method=method) if n >= min_obs else np.nan
        rows.append({"horizon": h_value, "point_estimate": r, "n_obs": n})
    return pd.DataFrame(rows).sort_values("horizon").set_index("horizon")


# ============================================================================
# Visualisierung
# ============================================================================

def plot_correlation_curves(
    curves: dict,
    *,
    candle_minutes: int = 5,
    title: str = "Korrelation mit Forward-Return, je Horizont",
    ax=None,
    show: bool = True,
):
    """Linienplot: x=Horizont (in Stunden umgerechnet), y=Korrelation, eine
    Linie je Feature -- bewusst ohne Konfidenzband, die Kurvenform soll
    direkt im Plot beurteilt werden (siehe correlation_curve).

    Args:
        curves: dict feature_name -> DataFrame wie von correlation_curve
            zurueckgegeben (Index=Horizont in Kerzen, Spalte point_estimate).
        candle_minutes: Kerzengroesse in Minuten, fuer die x-Achsen-Beschriftung
            (Standard 5, siehe config.OHLCV_DIR = asset_ohlcv_5m).
    """
    import matplotlib.pyplot as plt

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    for name, curve in curves.items():
        hours = curve.index.to_numpy() * candle_minutes / 60.0
        ax.plot(hours, curve["point_estimate"], label=name)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizont (Stunden)")
    ax.set_ylabel(f"Korrelation mit Forward-Return")
    ax.set_title(title)
    ax.legend()

    if owns_figure:
        fig.tight_layout()
        if show:
            plt.show()
            plt.close(fig)
    return ax


def plot_correlation_curves_interactive(
    curves_by_variant: dict,
    *,
    candle_minutes: int = 5,
    title: str = "Korrelation mit Forward-Return, je Horizont",
):
    """Interaktive (Plotly) Version von plot_correlation_curves: alle Varianten
    (z.B. roh/residualisiert) UND alle Features gleichzeitig in EINER Figur,
    statt getrennter Subplots -- klickbare Legende zum gezielten Ein-/Ausblenden
    einzelner Linien, Hover zeigt Korrelation + n_obs. Wird nuetzlicher, je mehr
    Eintraege FEATURE_REGISTRY bekommt (aktuell nur "zscore", daher noch eine
    Linie pro Variante).

    Args:
        curves_by_variant: dict variant_name -> (dict feature_name -> DataFrame
            wie von correlation_curve zurueckgegeben, z.B.
            {"roh": curves_raw, "residualisiert": curves_residual}).
    """
    import plotly.graph_objects as go

    dash_options = ["solid", "dash", "dot", "dashdot"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig = go.Figure()
    for i, (variant, curves) in enumerate(curves_by_variant.items()):
        dash = dash_options[i % len(dash_options)]
        for j, (feat, curve) in enumerate(curves.items()):
            hours = curve.index.to_numpy() * candle_minutes / 60.0
            fig.add_trace(go.Scatter(
                x=hours, y=curve["point_estimate"],
                mode="lines", name=f"{feat} ({variant})",
                line=dict(color=colors[j % len(colors)], dash=dash),
                customdata=curve["n_obs"].to_numpy(),
                hovertemplate=(
                    "Horizont=%{x:.1f}h<br>Korrelation=%{y:.3f}"
                    "<br>n_obs=%{customdata}<extra>%{fullData.name}</extra>"
                ),
            ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Horizont (Stunden)",
        yaxis_title="Korrelation mit Forward-Return",
        legend_title="Feature (Variante) -- klicken zum Ein-/Ausblenden",
        template="plotly_white",
        height=550,
    )
    fig.show()
    return fig
