"""Spread-Diagnostik: 1:1-Extraktion aus notebooks/03_signal_generation/TradeData.ipynb.

Prüft, ob ein Preis-Level-Spread zwischen Tickern (paarweise, CADF-Stil) oder
über eine ganze Cluster-Gruppe (Johansen-Kointegration) mean-revertierend ist.
Reine Signalgenerierungs-Logik ohne Kenntnis von Position/Kapital/Zeit-
Simulation -- wird sowohl vom explorativen TradeData.ipynb als auch von den
Backtest-Strategien (src/backtest/strategy.py) verwendet.

WICHTIG (siehe TradeData.ipynb, Zelle "Schritt 1"): die Renditedifferenz
r1_t - r2_t ist für Stationaritätstests UNGEEIGNET (trivial stationär, da
bereits log-differenziert). ADF/KPSS/Hurst/Half-Life werden stattdessen auf
dem PREIS-Level-Spread angewendet: spread_t = log(P1_t) - hedge_ratio*log(P2_t).
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

_JOHANSEN_SIG_COLUMNS = {0.10: 0, 0.05: 1, 0.01: 2}


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Log-Returns pro Ticker: r_t = log(P_t / P_t-1). Erste Zeile ist NaN -> dropna."""
    return np.log(price_df / price_df.shift(1)).dropna(how="any")


def compute_return_spread(returns_df: pd.DataFrame, ticker_1: str, ticker_2: str) -> pd.Series:
    """Rt = r_Ai,t - r_Aj,t -- Intra-Cluster-Spread-Zeitreihe für die
    Lin-et-al.-Signalgenerierung. NICHT für ADF/KPSS/Hurst/Half-Life verwenden
    (siehe compute_price_spread)."""
    return returns_df[ticker_1] - returns_df[ticker_2]


def estimate_hedge_ratio(price_df: pd.DataFrame, ticker_1: str, ticker_2: str) -> float:
    """Schätzt die Hedge-Ratio per OLS (CADF-Stil, Chan 2013):
    log(P1) = alpha + beta * log(P2) + eps
    beta ist die Hedge-Ratio, mit der ticker_2 gewichtet wird, um den
    stationärsten Preis-Spread gegen ticker_1 zu bilden.

    Zeilen, an denen ticker_1 oder ticker_2 NaN ist (z.B. weil ein Asset im
    price_df-Fenster noch nicht gelistet war), werden vor der Regression
    entfernt -- sonst bricht OLS mit "exog contains inf or nans" ab, sobald
    ein Clique-Mitglied eine Datenlücke hat (siehe TestHedgeRatioAndSpread).
    """
    log_p1 = np.log(price_df[ticker_1]).to_numpy()
    log_p2 = np.log(price_df[ticker_2]).to_numpy()

    valid = np.isfinite(log_p1) & np.isfinite(log_p2)#gibt zurück, ob die Werte in log_p1 und log_p2 endlich sind (nicht NaN oder unendlich)
    n_valid = int(valid.sum())
    if n_valid < 3:
        raise ValueError(
            f"estimate_hedge_ratio({ticker_1!r}, {ticker_2!r}): nur {n_valid} gültige "
            f"(nicht-NaN) Beobachtungen im übergebenen Fenster -- mindestens 3 nötig."
        )

    x = add_constant(log_p2[valid])
    model = OLS(log_p1[valid], x).fit()
    return float(model.params[1])


def compute_price_spread(price_df: pd.DataFrame, ticker_1: str, ticker_2: str, hedge_ratio: float = 1.0) -> pd.Series:
    """Spread auf Preis-Ebene (Level): spread_t = log(P1_t) - hedge_ratio * log(P2_t).
    Das ist die Reihe, auf die ADF/KPSS/Hurst/Half-Life angewendet werden müssen."""
    log_p1 = np.log(price_df[ticker_1])
    log_p2 = np.log(price_df[ticker_2])
    return log_p1 - hedge_ratio * log_p2


def hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """Hurst-Exponent via Rescaled-Range-Analyse (R/S), wie in der Chan-Toolbox.

    H < 0.5  -> mean-revertierend (anti-persistent)
    H = 0.5  -> Random Walk
    H > 0.5  -> trendfolgend (persistent)
    """
    series = np.asarray(series)
    lags = range(2, min(max_lag, len(series) // 2))

    tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
    tau = np.array(tau)
    lags = np.array(list(lags))

    valid = tau > 0
    poly = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
    return float(poly[0])


def compute_half_life(series: pd.Series) -> float:
    """Half-Life eines Ornstein-Uhlenbeck-Prozesses: Δy(t) = λ*y(t-1) + μ + ε
    -> Half-Life = -ln(2)/λ. λ >= 0 ergibt keine sinnvolle Half-Life -> inf."""
    y = series.to_numpy()
    y_lag = y[:-1]
    y_diff = np.diff(y)

    x = add_constant(y_lag)
    model = OLS(y_diff, x).fit()
    lam = model.params[1]

    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / lam)


def analyze_spread_stationarity(price_df: pd.DataFrame, ticker_1: str, ticker_2: str, alpha: float = 0.05) -> dict:
    """Führt ADF/KPSS/Hurst/Half-Life auf dem PREIS-Level-Spread eines Tickerpaars aus.

    Returns:
        Dict mit hedge_ratio, adf_pvalue, adf_stationary, kpss_pvalue,
        kpss_stationary, kpss_capped, hurst, half_life_periods,
        mean_reverting (ADF und KPSS stimmen überein UND Hurst < 0.5).
    """
    hedge_ratio = estimate_hedge_ratio(price_df, ticker_1, ticker_2)
    spread = compute_price_spread(price_df, ticker_1, ticker_2, hedge_ratio)
    diag = evaluate_spread_quality(spread, alpha=alpha)
    return {"ticker_1": ticker_1, "ticker_2": ticker_2, "hedge_ratio": hedge_ratio, **diag}


def evaluate_spread_quality(spread: pd.Series, *, alpha: float = 0.05) -> dict:
    """ADF/KPSS/Hurst/Half-Life auf einer FERTIGEN Spread-Reihe (statt auf einem
    Ticker-Paar wie analyze_spread_stationarity). Nötig, weil der Cluster-Spread
    aus N Assets kein Paar ist. Rückgabe-Schlüssel identisch zu
    analyze_spread_stationarity, minus ticker_1/ticker_2/hedge_ratio.
    """
    adf_pvalue = adfuller(spread)[1]
    adf_stationary = adf_pvalue < alpha  # ADF: H0 = Unit Root -> p < alpha heisst stationär

    # KPSS: H0 = stationär -> p < alpha heisst NICHT stationär (umgekehrte Logik zu ADF).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InterpolationWarning)
        _, kpss_pvalue, _, _ = kpss(spread, regression="c", nlags="auto")
    kpss_capped = any(issubclass(w.category, InterpolationWarning) for w in caught)
    kpss_stationary = kpss_pvalue >= alpha

    hurst = hurst_exponent(spread.to_numpy())
    half_life = compute_half_life(spread)

    mean_reverting = adf_stationary and kpss_stationary and hurst < 0.5

    return {
        "adf_pvalue": adf_pvalue,
        "adf_stationary": adf_stationary,
        "kpss_pvalue": kpss_pvalue,
        "kpss_stationary": kpss_stationary,
        "kpss_capped": kpss_capped,
        "hurst": hurst,
        "half_life_periods": half_life,
        "mean_reverting": mean_reverting,
    }


def compute_all_spread_diagnostics(price_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Wendet analyze_spread_stationarity auf alle Tickerpaare an (alle mit allen).

    Returns:
        Lange Tabelle, ein Zeile pro Paar, sortiert nach Half-Life aufsteigend;
        mean_reverting=False Paare am Ende.
    """
    tickers = price_df.columns.tolist()

    rows = []
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j <= i:
                continue
            rows.append(analyze_spread_stationarity(price_df, t1, t2, alpha=alpha))

    diagnostics_df = pd.DataFrame(rows)
    diagnostics_df = diagnostics_df.sort_values(
        ["mean_reverting", "half_life_periods"], ascending=[False, True]
    ).reset_index(drop=True)

    n_capped = int(diagnostics_df["kpss_capped"].sum())
    if n_capped > 0:
        print(f"ℹ️  Bei {n_capped}/{len(diagnostics_df)} Paaren lag die KPSS-Teststatistik ausserhalb "
              f"der Nachschlagetabelle — der angezeigte kpss_pvalue ist der nächstgelegene "
              f"Tabellen-Grenzwert, nicht der exakte Wert. Die Stationaritäts-Einstufung "
              f"(kpss_stationary) bleibt davon unberührt.")

    return diagnostics_df


def compute_johansen_test(price_df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Johansen-Kointegrationstest über alle Ticker gleichzeitig (Log-Preise).

    Prüft, ob es unter der GESAMTEN Gruppe von Tickern eine oder mehrere
    stationäre Linearkombinationen gibt, statt nur einzelne Paare zu betrachten.

    Returns:
        Dict mit n_coint_relations, trace_stats, trace_crit_95, eigenvectors
        (DataFrame, Spalten=relation_i, Zeilen=Ticker), result (rohes
        statsmodels-Objekt).
    """
    tickers = price_df.columns.tolist()
    log_prices = np.log(price_df)

    result = coint_johansen(log_prices.to_numpy(), det_order, k_ar_diff)

    trace_stats = result.lr1
    trace_crit_95 = result.cvt[:, _JOHANSEN_SIG_COLUMNS[0.05]]

    n_coint_relations = int(np.sum(trace_stats > trace_crit_95))

    eigenvectors = pd.DataFrame(
        result.evec[:, :len(tickers)],
        index=tickers,
        columns=[f"relation_{i+1}" for i in range(len(tickers))],
    )

    return {
        "n_coint_relations": n_coint_relations,
        "trace_stats": trace_stats,
        "trace_crit_95": trace_crit_95,
        "eigenvectors": eigenvectors,
        "result": result,
    }


def normalize_cointegration_vector(weights: pd.Series) -> pd.Series:
    """Normiert einen Johansen-Eigenvektor deterministisch.

    Johansen-Eigenvektoren haben beliebiges Vorzeichen UND beliebige Skalierung
    (statsmodels normiert nicht deterministisch). Ohne diese Normierung dreht
    derselbe Cluster zwischen zwei Snapshots die Spread-Richtung -- der Z-Score
    springt dann von +2 auf -2, obwohl sich ökonomisch nichts geändert hat.

    Konvention: Element mit grösstem |w| wird positiv, danach sum(|w|) == 1.
    """
    idx_max = weights.abs().idxmax()
    signed = weights if weights[idx_max] > 0 else -weights
    return signed / signed.abs().sum()


def build_cluster_spread(price_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Baut den Cluster-Spread als gewichtete Summe der Log-Preise.

    weights kommt typischerweise aus normalize_cointegration_vector(
    compute_johansen_test(...)["eigenvectors"]["relation_1"]).
    """
    log_prices = np.log(price_df[weights.index])
    return (log_prices * weights).sum(axis=1)


def spread_zscore_params(spread: pd.Series) -> tuple[float, float]:
    """(mu, sigma) des Formationsfensters -- werden bei Entry eingefroren."""
    return float(spread.mean()), float(spread.std(ddof=0))
