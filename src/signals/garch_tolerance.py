"""GARCH(1,1)-Toleranzintervall-Signalkette fuer Pairtrading -- Pendant zu
src.signals.tolerance (dort: EWMA-Vola nach Lin, Chen & Syu 2021).

Ablauf (Steps 1-4), identisch zu tolerance.py bis auf Schritt 2:
    1. Return-Spread pro Paar (tolerance.build_pairwise_return_spreads,
       wiederverwendet -- reine Datenaufbereitung, Signal-agnostisch).
    2. GARCH(1,1)-Vola normieren: Xt = Rt / sigma_t (src.signals.garch.
       compute_garch_volatility, siehe dortigen Modul-Docstring fuer die
       Begruendung der eigenen Implementierung statt des arch-Pakets).
    3. Xmax_t = max(|Xt|) je Zeitpunkt (tolerance.compute_max_adjusted_spread,
       wiederverwendet -- arbeitet nur auf dem normierten DataFrame, kennt
       die Normierungsmethode nicht).
    4. Wilks-Toleranzintervall + 1-Schritt-Vergleich (tolerance.
       wilks_tolerance_limits/generate_tolerance_signals, wiederverwendet).

Exit-Mitte (Aufgabenstellung-Zusatz, siehe garch_pairtrading_engine.py):
    Anders als beim Preis-Spread-Exit der EWMA-/Bollinger-Varianten
    (compute_exit_mean auf dem Preis-Level-Spread) wird hier die Mitte
    ZWISCHEN den beiden bereits vom Entry-Schritt berechneten Wilks-
    Toleranzgrenzen als Exit-Referenz verwendet:
        exit_mid_t = (lower_limit_t + upper_limit_t) / 2
    auf |Xt| DES GEHANDELTEN PAARES (nicht Xmax der ganzen Clique) -- der
    Trade schliesst, sobald |Xt| diese Mitte erreicht/unterschreitet. Kein
    zusaetzliches Lookback-Fenster noetig, die Mitte ergibt sich direkt aus
    den ohnehin fuer den Entry berechneten Grenzen.

Trade-Simulation/Orchestrierung (Steps 5-6) lebt in
notebooks/03_Tradeing/garch_pairtrading_engine.py, das dieses Modul
importiert -- analog zur Aufteilung tolerance.py/pairtrading_engine.py.
"""

import pandas as pd

from src.signals.garch import compute_garch_volatility


def compute_garch_adjusted_spreads(
    spread_df: pd.DataFrame, *, dist: str = "t", max_iter: int = 500,
) -> tuple[pd.DataFrame, dict]:
    """Xt = Rt / sigma_t fuer jedes Paar einzeln, mit GARCH(1,1)- statt
    EWMA-Volatilitaet (siehe compute_garch_volatility) -- Pendant zu
    tolerance.compute_adjusted_spreads.

    sigma_t wird UNABHAENGIG pro Spalte (Paar) gefittet, jedes Paar hat
    seine eigene GARCH-Parametrisierung (omega/alpha/beta/nu).

    Args:
        spread_df: DataFrame (timestamp x pair), siehe
            tolerance.build_pairwise_return_spreads.
        dist, max_iter: siehe compute_garch_volatility.

    Returns:
        (adjusted_df, fit_info_by_pair): adjusted_df wie
        tolerance.compute_adjusted_spreads (normierte Spreads Xt).
        fit_info_by_pair ist ein dict {pair: fit_info} (siehe
        compute_garch_volatility) -- fuer Diagnose/Logging, ob der Fit pro
        Paar konvergiert ist.
    """
    import numpy as np

    adjusted = {}
    fit_info_by_pair = {}
    for pair in spread_df.columns:
        sigma, fit_info = compute_garch_volatility(spread_df[pair], dist=dist, max_iter=max_iter)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = spread_df[pair] / sigma.replace(0.0, np.nan)
        adjusted[pair] = x
        fit_info_by_pair[pair] = fit_info
    return pd.DataFrame(adjusted, index=spread_df.index), fit_info_by_pair
