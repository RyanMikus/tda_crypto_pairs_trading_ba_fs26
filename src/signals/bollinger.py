"""Bollinger-Band-Signalkette fuer Pairtrading auf dem Preis-Level-Spread.

Ablauf (Steps 1-3), Pendant zu tolerance.py's EWMA-Toleranzintervall-Kette,
aber mit dem klassischen Bollinger-Band-Ansatz statt Wilks-Toleranzintervall:
    1. Preis-Level-Spread pro Paar in einer Clique (build_pairwise_price_spreads)
       -- spread_t = log(P_i,t) - hedge_ratio * log(P_j,t), NICHT die Rendite-
       differenz (siehe src.signals.spread-Modul-Docstring: die Renditedifferenz
       ist fuer Stationaritaetstests/Level-Spreads ungeeignet, weil bereits
       log-differenziert). Klassischer Bollinger-Ansatz arbeitet auf dem
       Preis-Level-Spread selbst.
    2. Gleitender Mittelwert + gleitende Standardabweichung ueber ein
       Lookback-Fenster (compute_bollinger_bands) -> obere/untere Bandgrenze
       = Mittelwert +/- n_std * Std.
    3. Z_t = (spread_t - Mittelwert_t) / Std_t je Paar, Zmax_t = max(|Z_t|)
       ueber alle Paare (compute_max_zscore) -- Pendant zu Xmax_t in
       tolerance.py, macht Paare untereinander vergleichbar.

Exit-Signal (compute_exit_mean): EIGENER rollierender Mittelwert des
Preis-Spreads mit eigenem, tunbaren Lookback-Fenster (exit_lookback_candles,
unabhaengig vom Entry-lookback_candles) -- der Trade schliesst, sobald der
Spread diesen Mittelwert erreicht/kreuzt (direkter Preis-Vergleich, kein
Z-Score noetig fuer den Exit).

Entry/Exit-Logik (Steps 4-5) lebt in
notebooks/03_Tradeing/bollinger_pairtrading_engine.py, das dieses Modul
importiert -- analog zur Aufteilung tolerance.py/pairtrading_engine.py.
"""

import numpy as np
import pandas as pd

from src.signals.spread import compute_price_spread, estimate_hedge_ratio


# ============================================================================
# 1. Intra-Clique-Preis-Level-Spreads
# ============================================================================

def build_pairwise_price_spreads(
    price_df: pd.DataFrame, *, hedge_lookback_candles: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """spread_t^(Ai,Aj) = log(P_Ai,t) - hedge_ratio * log(P_Aj,t) fuer ALLE
    Paare innerhalb einer Clique -- Preis-LEVEL-Spread, nicht Renditedifferenz
    (siehe Modul-Docstring).

    Args:
        price_df: DataFrame (timestamp x asset_id) mit Preisen, bereits auf
            die Mitglieder EINER Clique gefiltert.
        hedge_lookback_candles: falls gesetzt, wird die Hedge-Ratio NUR auf
            den letzten hedge_lookback_candles Kerzen geschaetzt (vermeidet
            Lookahead auf zukuenftige Kerzen innerhalb des uebergebenen
            price_df-Fensters). Default: gesamtes price_df (Aufrufer ist dafuer
            verantwortlich, dass price_df keine zukuenftigen Kerzen enthaelt).

    Returns:
        (spread_df, hedge_ratios): spread_df wie build_pairwise_return_spreads
        (timestamp x pair, pair als Tuple (asset_id_i, asset_id_j) mit i < j),
        hedge_ratios ist ein dict {pair: hedge_ratio} fuer Diagnose/Reuse.
        Paare, fuer die im hedge_window zu wenig gueltige (nicht-NaN)
        Beobachtungen vorliegen (z.B. ein Clique-Mitglied war zu diesem
        Zeitpunkt noch nicht gelistet), werden STILLSCHWEIGEND uebersprungen
        -- tauchen also weder in spread_df noch in hedge_ratios auf, statt
        den gesamten Aufruf abzubrechen (siehe estimate_hedge_ratio).
    """
    asset_ids = list(price_df.columns)
    hedge_window = price_df.iloc[-hedge_lookback_candles:] if hedge_lookback_candles else price_df

    spreads = {}
    hedge_ratios = {}
    for idx, a_i in enumerate(asset_ids):
        for a_j in asset_ids[idx + 1:]:
            try:
                hedge_ratio = estimate_hedge_ratio(hedge_window, a_i, a_j)
            except ValueError:
                continue
            spreads[(a_i, a_j)] = compute_price_spread(price_df, a_i, a_j, hedge_ratio)
            hedge_ratios[(a_i, a_j)] = hedge_ratio

    return pd.DataFrame(spreads, index=price_df.index), hedge_ratios


# ============================================================================
# 2. Bollinger-Baender: gleitender Mittelwert + gleitende Standardabweichung
# ============================================================================

def compute_bollinger_bands(
    spread_df: pd.DataFrame, *, lookback_candles: int, n_std: float,
) -> dict[tuple, pd.DataFrame]:
    """Rollierender Mittelwert/Std je Paar-Spalte, klassische Bollinger-Baender:
        mid_t   = mean(spread_{t-lookback_candles+1 .. t})
        std_t   = std(spread_{t-lookback_candles+1 .. t}, ddof=1)
        upper_t = mid_t + n_std * std_t
        lower_t = mid_t - n_std * std_t

    Args:
        spread_df: DataFrame (timestamp x pair), siehe build_pairwise_price_spreads.
        lookback_candles: Fenstergroesse fuer rolling mean/std (min_periods=
            lookback_candles, also NaN bis genug Historie vorhanden ist --
            kein Teil-Fenster-Bias am Anfang der Serie).
        n_std: Bandbreite in Standardabweichungen (klassisch 2.0).

    Returns:
        dict {pair: DataFrame mit Spalten spread/mid/std/upper/lower/zscore},
        ein DataFrame je Paar (gleicher Index wie spread_df).
    """
    if lookback_candles < 2:
        raise ValueError(f"lookback_candles muss >= 2 sein, erhalten: {lookback_candles}")
    if n_std <= 0:
        raise ValueError(f"n_std muss > 0 sein, erhalten: {n_std}")

    bands = {}
    for pair in spread_df.columns:
        spread = spread_df[pair]
        mid = spread.rolling(lookback_candles, min_periods=lookback_candles).mean()
        std = spread.rolling(lookback_candles, min_periods=lookback_candles).std(ddof=1)
        upper = mid + n_std * std
        lower = mid - n_std * std
        with np.errstate(invalid="ignore", divide="ignore"):
            zscore = (spread - mid) / std.replace(0.0, np.nan)
        bands[pair] = pd.DataFrame({
            "spread": spread, "mid": mid, "std": std,
            "upper": upper, "lower": lower, "zscore": zscore,
        }, index=spread_df.index)
    return bands


def compute_exit_mean(spread_df: pd.DataFrame, *, exit_lookback_candles: int) -> pd.DataFrame:
    """Rollierender Mittelwert des Preis-Spreads je Paar-Spalte, UNABHAENGIG
    vom Entry-Lookback (compute_bollinger_bands) -- eigenes, tunbares
    Zeitfenster fuer die Exit-Bedingung.

    Anders als die Entry-Baender (Mittelwert + n_std * Std, verglichen als
    Z-Score) ist der Exit hier ein direkter Preis-Spread-Vergleich: der Trade
    schliesst, sobald spread_t diesen Mittelwert erreicht/kreuzt (siehe
    bollinger_pairtrading_engine.simulate_bollinger_trade) -- kein Z-Score,
    keine Standardabweichung noetig fuer den Exit.

    Args:
        spread_df: DataFrame (timestamp x pair), siehe build_pairwise_price_spreads.
        exit_lookback_candles: Fenstergroesse fuer den rollierenden Mittelwert
            (min_periods=exit_lookback_candles, NaN bis genug Historie vorhanden ist).

    Returns:
        DataFrame gleicher Form wie spread_df, mit dem rollierenden Mittelwert
        je Paar (Spaltennamen wie spread_df).
    """
    if exit_lookback_candles < 2:
        raise ValueError(f"exit_lookback_candles muss >= 2 sein, erhalten: {exit_lookback_candles}")

    return spread_df.rolling(exit_lookback_candles, min_periods=exit_lookback_candles).mean()


def compute_max_zscore(bands: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
    """Zmax_t = max(|Z_t| ueber alle Paare) je Zeitpunkt, plus welches Paar
    und in welche Richtung -- Pendant zu tolerance.compute_max_adjusted_spread.

    Args:
        bands: wie von compute_bollinger_bands zurueckgegeben.

    Returns:
        DataFrame (Index=timestamp) mit Spalten:
            - zmax (float): max(|Z_t|) ueber alle Paare an diesem Zeitpunkt.
            - pair (object): das Tuple (asset_id_i, asset_id_j) des Extremwerts.
            - sign (float): Vorzeichen des zugrundeliegenden Z_t. sign=+1
              bedeutet spread_t liegt UEBER dem gleitenden Mittelwert (Asset i
              ist relativ zu Asset j teuer geworden) -> Handelsrichtung
              "Short Asset i / Long Asset j" (Reversion-Erwartung: Spread
              faellt zurueck zur Mitte). sign=-1 entsprechend umgekehrt.
    """
    zscore_df = pd.DataFrame({pair: df["zscore"] for pair, df in bands.items()})
    abs_df = zscore_df.abs()

    rows = []
    for ts, row in abs_df.iterrows():
        if row.isna().all():
            rows.append({"zmax": np.nan, "pair": None, "sign": np.nan})
            continue
        best_pair = row.idxmax()
        zmax = row[best_pair]
        raw_z = zscore_df.at[ts, best_pair]
        sign = np.sign(raw_z) if pd.notna(raw_z) else np.nan
        rows.append({"zmax": zmax, "pair": best_pair, "sign": sign})
    return pd.DataFrame(rows, index=zscore_df.index)
