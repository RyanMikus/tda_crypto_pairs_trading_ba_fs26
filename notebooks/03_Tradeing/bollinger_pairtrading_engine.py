"""Bollinger-Band-Pairtrading-Engine fuer Clique-Mitglieder.

Dritte, PARALLELE Strategie-Engine neben pairtrading_engine.py (EWMA-
Toleranzintervall nach Lin, Chen & Syu 2021) und der RSI/Bollinger-
Einzelasset-Engine (src/backtest/). Baut auf derselben Infrastruktur auf
(Cliquen-Cache, ohlcv_df), wiederholt sie nicht.

Ablauf (Steps 1-6):
    1.-3. Signalkette (Preis-Level-Spread, gleitender Mittelwert/Std,
       Bollinger-Baender, Zmax) lebt in src.signals.bollinger, siehe
       dortigen Modul-Docstring.
    4. Entry bei Bandausbruch (|Zt| > n_std), Exit sobald der Preis-Spread
       seinen EIGENEN rollierenden Mittelwert (exit_lookback_candles,
       compute_exit_mean) erreicht/kreuzt, oder Holding-Limit
       (simulate_bollinger_trade).
    5. Orchestrierung ueber alle Cliquen/Baskets (run_bollinger_backtest).
    6. Zweistufiges Hyperparameter-Tuning (Stufe-1-Screen + Walk-Forward,
       siehe tune_bollinger.py).

WICHTIGE ABGRENZUNG zu pairtrading_engine.py: dort ist die Signalgroesse die
EWMA-vola-normierte RENDITEDIFFERENZ (Xt = Rt/sigma_t), verglichen gegen ein
nichtparametrisches Wilks-Toleranzintervall. HIER ist die Signalgroesse der
klassische Bollinger-Z-Score des PREIS-LEVEL-Spreads (Zt = (spread_t -
gleitender_Mittelwert_t) / gleitende_Std_t), verglichen gegen feste
Bandbreiten in Standardabweichungen (n_std). Beide sind Mean-Reversion-
Strategien auf denselben TDA-Cliquen, aber mit unterschiedlicher Signal-
Definition -- direkt vergleichbar ueber plot_strategy_comparison
(pairtrading_engine.py).

PairTrade nutzt dieselben Spalten wie pairtrading_engine.PairTrade, damit
compute_backtest_stats/compute_equity_curve (pairtrading_engine.py)
unveraendert weiterverwendet werden koennen.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Src.signals.bollinger import (
    build_pairwise_price_spreads,
    compute_bollinger_bands,
    compute_exit_mean,
    compute_max_zscore,
)


@dataclass
class BollingerTrade:
    """Zwei-Leg-Trade (dollar-neutral: gleiche Notional-Groesse long/short).

    Stellt dieselben Spalten (net_return, holding_candles, entry_time,
    exit_time) wie pairtrading_engine.PairTrade bereit, damit
    pairtrading_engine.compute_backtest_stats/compute_equity_curve
    unveraendert auf Bollinger-Ergebnissen weiterverwendet werden koennen.
    """
    basket: str
    discovery_date: pd.Timestamp
    pair: tuple
    asset_long: int
    asset_short: int
    asset_long_sym: str
    asset_short_sym: str
    direction: str  # "long_j_short_i" | "long_i_short_j" (siehe simulate_bollinger_trade)
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price_long: float
    entry_price_short: float
    exit_price_long: float
    exit_price_short: float
    exit_reason: str  # "signal" | "holding_limit" | "day_end"
    gross_return: float  # Summe beider Leg-Returns (dollar-neutral, je 50% Notional)
    net_return: float    # gross_return - 4*fee_bps/10000 (Entry+Exit je Leg)
    holding_candles: int


def simulate_bollinger_trade(
    price_df: pd.DataFrame,
    entry_time: pd.Timestamp,
    exit_deadline: pd.Timestamp,
    pair: tuple,
    direction: str,
    *,
    signal_series: dict,
    basket: str,
    discovery_date: pd.Timestamp,
    max_holding_candles: int,
    fee_bps: float,
) -> "BollingerTrade | None":
    """Simuliert EINEN Paar-Trade ab entry_time (zwei Legs, dollar-neutral).

    Exit-Bedingungen (erste erfuellte gewinnt): der Preis-Spread erreicht/
    kreuzt seinen EIGENEN rollierenden Mittelwert (exit_mean, siehe
    compute_exit_mean -- unabhaengiges Lookback-Fenster vom Entry-Band),
    max_holding_candles erreicht, oder exit_deadline erreicht.

    Kreuzung statt Schwellenwert: verglichen wird das Vorzeichen von
    (spread_t - exit_mean_t) bei Entry gegen jeden Folgezeitpunkt -- sobald
    sich das Vorzeichen aendert (oder exakt 0 erreicht), hat der Spread den
    Mittelwert erreicht/ueberquert und der Trade wird geschlossen. Kein
    fester exit_z-Schwellenwert mehr noetig.

    Args:
        price_df: DataFrame (timestamp x asset_id), deckt entry_time bis
            mindestens exit_deadline ab.
        entry_time: Zeitpunkt des Entry-Signals.
        exit_deadline: spaetester moeglicher Exit-Zeitpunkt.
        pair: (asset_id_i, asset_id_j).
        direction: "long_i_short_j" oder "long_j_short_i", siehe
            compute_max_zscore.
        signal_series: dict mit "spread" (Preis-Spread-Verlauf dieses Paares)
            und "exit_mean" (rollierender Mittelwert, siehe compute_exit_mean).
        max_holding_candles, fee_bps: siehe Modul-Docstring.

    Returns:
        BollingerTrade, oder None falls entry_time fehlt oder Preise fehlen.
    """
    a_i, a_j = pair
    if a_i not in price_df.columns or a_j not in price_df.columns:
        return None

    idx = price_df.index
    if entry_time not in idx:
        return None
    entry_pos = idx.get_loc(entry_time)

    price_i_entry = price_df[a_i].iloc[entry_pos]
    price_j_entry = price_df[a_j].iloc[entry_pos]
    if pd.isna(price_i_entry) or pd.isna(price_j_entry):
        return None

    if direction == "long_i_short_j":
        asset_long, asset_short = a_i, a_j
        entry_price_long, entry_price_short = price_i_entry, price_j_entry
    elif direction == "long_j_short_i":
        asset_long, asset_short = a_j, a_i
        entry_price_long, entry_price_short = price_j_entry, price_i_entry
    else:
        raise ValueError(f"direction muss 'long_i_short_j' oder 'long_j_short_i' sein, erhalten: {direction}")

    spread_series = signal_series["spread"]
    exit_mean_series = signal_series["exit_mean"]

    entry_diff = spread_series.iloc[entry_pos] - exit_mean_series.iloc[entry_pos]
    entry_sign = np.sign(entry_diff) if pd.notna(entry_diff) else np.nan

    exit_pos = None
    exit_reason = None
    max_pos = min(len(idx) - 1, entry_pos + max_holding_candles)

    for i in range(entry_pos + 1, max_pos + 1):
        ts = idx[i]
        if ts > exit_deadline or i == len(idx) - 1:
            exit_pos, exit_reason = i, "day_end"
            break
        if i - entry_pos >= max_holding_candles:
            exit_pos, exit_reason = i, "holding_limit"
            break
        cur_diff = spread_series.iloc[i] - exit_mean_series.iloc[i]
        if pd.notna(cur_diff) and pd.notna(entry_sign):
            cur_sign = np.sign(cur_diff)
            if cur_sign == 0 or cur_sign != entry_sign:
                exit_pos, exit_reason = i, "signal"
                break

    if exit_pos is None:
        exit_pos, exit_reason = max_pos, "holding_limit"

    exit_time = idx[exit_pos]
    exit_price_long = price_df[asset_long].iloc[exit_pos]
    exit_price_short = price_df[asset_short].iloc[exit_pos]
    if pd.isna(exit_price_long) or pd.isna(exit_price_short):
        return None

    ret_long = exit_price_long / entry_price_long - 1.0
    ret_short = -(exit_price_short / entry_price_short - 1.0)
    # Dollar-neutral: je 50% Notional pro Leg -> gross_return ist der Mittelwert,
    # nicht die Summe, der beiden Leg-Returns.
    gross_return = 0.5 * (ret_long + ret_short)
    net_return = gross_return - 4.0 * fee_bps / 10_000.0

    id_to_symbol_fallback = lambda a: f"ID:{a}"  # noqa: E731 -- Symbol-Aufloesung passiert im Aufrufer

    return BollingerTrade(
        basket=basket, discovery_date=discovery_date, pair=pair,
        asset_long=asset_long, asset_short=asset_short,
        asset_long_sym=id_to_symbol_fallback(asset_long), asset_short_sym=id_to_symbol_fallback(asset_short),
        direction=direction, entry_time=entry_time, exit_time=exit_time,
        entry_price_long=float(entry_price_long), entry_price_short=float(entry_price_short),
        exit_price_long=float(exit_price_long), exit_price_short=float(exit_price_short),
        exit_reason=exit_reason, gross_return=float(gross_return), net_return=float(net_return),
        holding_candles=int(exit_pos - entry_pos),
    )


def run_bollinger_backtest(
    *,
    cliques_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    id_to_symbol: dict,
    timestamp_column: str,
    asset_id_column: str,
    close_column: str,
    start_date,
    end_date,
    lookback_candles: int,
    n_std: float,
    exit_lookback_candles: int,
    max_holding_candles: int,
    fee_bps: float,
    trade_mode: str = "max_only",
    hedge_lookback_candles: int | None = None,
    warmup_candles: int | None = None,
) -> pd.DataFrame:
    """Orchestriert den Bollinger-Band-Pairtrading-Backtest ueber alle
    Cliquen/Baskets im gewaehlten Zeitraum.

    trade_mode (wie pairtrading_engine.run_pairtrading_backtest):
        - "max_only": pro Zeitpunkt nur das Paar mit dem groessten |Zt| als
          Handelskandidat.
        - "all_significant": handelt gleichzeitig ALLE Paare, deren |Zt| das
          Band verletzt.

    Args:
        cliques_df: aus dem Cache, bereits auf Baskets/Variante gefiltert.
        ohlcv_df: 5-Min-Kerzen-OHLCV mit timestamp/asset_id/close.
        lookback_candles: Fenster fuer gleitenden Mittelwert/Std des ENTRY-
            Bands (siehe compute_bollinger_bands) UND fuer die Hedge-Ratio-
            Schaetzung, falls hedge_lookback_candles nicht separat gesetzt ist.
        n_std: Bandbreite in Standardabweichungen fuer den Entry.
        exit_lookback_candles: EIGENES, von lookback_candles unabhaengiges
            Fenster fuer den rollierenden Mittelwert, gegen den der Exit
            verglichen wird (siehe compute_exit_mean/simulate_bollinger_trade)
            -- der Trade schliesst, sobald der Spread diesen Mittelwert
            erreicht/kreuzt.
        max_holding_candles, fee_bps: siehe simulate_bollinger_trade.
        hedge_lookback_candles: siehe build_pairwise_price_spreads. Default:
            lookback_candles (gleiches Fenster wie die Entry-Baender).
        warmup_candles: Vorlauf vor jedem Discovery-Tag fuer Hedge-Ratio-
            Schaetzung sowie Entry-/Exit-Lookback. Default: das groessere
            von lookback_candles/exit_lookback_candles + 30.

    Returns:
        DataFrame, eine Zeile pro Trade (BollingerTrade-Felder), leer falls
        keine Trades zustande kamen.
    """
    if trade_mode not in ("max_only", "all_significant"):
        raise ValueError(f"trade_mode muss 'max_only' oder 'all_significant' sein, erhalten: {trade_mode}")

    warmup_candles = warmup_candles or (max(lookback_candles, exit_lookback_candles) + 30)
    hedge_lookback_candles = hedge_lookback_candles or lookback_candles

    price_matrix = ohlcv_df.pivot_table(
        index=timestamp_column, columns=asset_id_column, values=close_column, aggfunc="last",
    ).sort_index()
    ts_index = price_matrix.index

    start_ts = pd.Timestamp(start_date)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(end_date)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    end_ts = end_ts + pd.Timedelta(days=1)
    day_mask = (cliques_df["discovery_date"] >= start_ts) & (cliques_df["discovery_date"] < end_ts)
    target_cliques = cliques_df.loc[day_mask]

    all_trades: list[BollingerTrade] = []

    for clique_row in target_cliques.itertuples():
        day_start = pd.Timestamp(clique_row.discovery_date).normalize()
        day_end = day_start + pd.Timedelta(days=1)

        day_start_pos = ts_index.searchsorted(day_start)
        day_end_pos = ts_index.searchsorted(day_end)
        window_start_pos = max(0, day_start_pos - warmup_candles)
        if day_end_pos <= day_start_pos or day_start_pos - warmup_candles < 0:
            continue

        clique_ids = [a for a in clique_row.asset_ids if a in price_matrix.columns]
        if len(clique_ids) < 2:
            continue

        window_prices = price_matrix.iloc[window_start_pos:day_end_pos][clique_ids]
        if window_prices.empty:
            continue

        spread_df, _hedge_ratios = build_pairwise_price_spreads(
            window_prices, hedge_lookback_candles=hedge_lookback_candles,
        )
        bands = compute_bollinger_bands(spread_df, lookback_candles=lookback_candles, n_std=n_std)
        max_z_df = compute_max_zscore(bands)
        exit_mean_df = compute_exit_mean(spread_df, exit_lookback_candles=exit_lookback_candles)

        day_slice = window_prices.index[window_prices.index >= day_start]
        exit_deadline = day_slice[-1] if len(day_slice) else window_prices.index[-1]

        if trade_mode == "max_only":
            entry_rows = max_z_df.loc[day_slice]
            entry_rows = entry_rows.loc[entry_rows["zmax"].abs() > n_std]
            for ts in entry_rows.index:
                row = entry_rows.loc[ts]
                pair = row["pair"]
                if pair is None:
                    continue
                direction = "long_j_short_i" if row["sign"] > 0 else "long_i_short_j"
                signal_series = {"spread": spread_df[pair], "exit_mean": exit_mean_df[pair]}
                trade = simulate_bollinger_trade(
                    window_prices, ts, exit_deadline, pair, direction,
                    signal_series=signal_series,
                    basket=clique_row.basket, discovery_date=day_start,
                    max_holding_candles=max_holding_candles, fee_bps=fee_bps,
                )
                if trade is not None:
                    trade.asset_long_sym = id_to_symbol.get(trade.asset_long, f"ID:{trade.asset_long}")
                    trade.asset_short_sym = id_to_symbol.get(trade.asset_short, f"ID:{trade.asset_short}")
                    all_trades.append(trade)
        else:  # all_significant
            for pair, band_df in bands.items():
                zt = band_df["zscore"]
                pair_abs = zt.abs()
                pair_entry = day_slice[(pair_abs.loc[day_slice] > n_std).to_numpy()]
                for ts in pair_entry:
                    sign = np.sign(zt.loc[ts])
                    if pd.isna(sign) or sign == 0:
                        continue
                    direction = "long_j_short_i" if sign > 0 else "long_i_short_j"
                    signal_series = {"spread": spread_df[pair], "exit_mean": exit_mean_df[pair]}
                    trade = simulate_bollinger_trade(
                        window_prices, ts, exit_deadline, pair, direction,
                        signal_series=signal_series,
                        basket=clique_row.basket, discovery_date=day_start,
                        max_holding_candles=max_holding_candles, fee_bps=fee_bps,
                    )
                    if trade is not None:
                        trade.asset_long_sym = id_to_symbol.get(trade.asset_long, f"ID:{trade.asset_long}")
                        trade.asset_short_sym = id_to_symbol.get(trade.asset_short, f"ID:{trade.asset_short}")
                        all_trades.append(trade)

    if not all_trades:
        return pd.DataFrame(columns=[
            "basket", "discovery_date", "pair", "asset_long", "asset_short",
            "asset_long_sym", "asset_short_sym", "direction", "entry_time", "exit_time",
            "entry_price_long", "entry_price_short", "exit_price_long", "exit_price_short",
            "exit_reason", "gross_return", "net_return", "holding_candles",
        ])

    return pd.DataFrame([vars(t) for t in all_trades]).sort_values("entry_time").reset_index(drop=True)


# ============================================================================
# Visualisierung
# ============================================================================

def plot_spread_with_bollinger_bands(
    band_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    ax=None,
    show: bool = True,
):
    """Plottet Preis-Level-Spread, gleitenden Mittelwert und Bollinger-Baender
    fuer EIN Paar, mit Entry-/Exit-Markern -- Pendant zu
    pairtrading_engine.plot_xmax_with_tolerance_bands.

    Args:
        band_df: EIN Eintrag aus compute_bollinger_bands's Rueckgabe-dict
            (Spalten spread/mid/upper/lower).
        trades_df: wie von run_bollinger_backtest zurueckgegeben, EINGESCHRAENKT
            auf das hier geplottete Paar/den Zeitraum.
    """
    import matplotlib.pyplot as plt

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    idx = band_df.index
    ax.plot(idx, band_df["spread"], color="#2a2a28", linewidth=1.1, label="Preis-Spread")
    ax.plot(idx, band_df["mid"], color="#898781", linewidth=1, linestyle="--", label="Gleitender Mittelwert")
    ax.plot(idx, band_df["upper"], color="#c1443f", linewidth=1, linestyle=":", label="Oberes Band")
    ax.plot(idx, band_df["lower"], color="#256abf", linewidth=1, linestyle=":", label="Unteres Band")

    if not trades_df.empty:
        for _, row in trades_df.iterrows():
            if row["entry_time"] in idx:
                ax.axvline(row["entry_time"], color="firebrick", linewidth=1, alpha=0.7)
                ax.scatter([row["entry_time"]], [band_df["spread"].loc[row["entry_time"]]], color="firebrick", zorder=5, s=40)
            if row["exit_time"] in idx:
                ax.axvline(row["exit_time"], color="seagreen", linewidth=1, alpha=0.7)
                ax.scatter([row["exit_time"]], [band_df["spread"].loc[row["exit_time"]]], color="seagreen", zorder=5, s=40, marker="s")

    ax.set_title("Preis-Spread mit Bollinger-Baendern")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Spread (Log-Preis-Differenz)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)

    if owns_figure:
        fig.autofmt_xdate()
        fig.tight_layout()
        if show:
            plt.show()
            plt.close(fig)
    return ax


# ============================================================================
# Zweistufiges Hyperparameter-Tuning: Stufe 1 (guenstiger Signal-Screen) +
# Walk-Forward-Zeitfenster fuer Stufe 2 (Optuna, siehe tune_bollinger.py).
# ============================================================================

def screen_bollinger_signal_quality(
    cliques_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    *,
    timestamp_column: str,
    asset_id_column: str,
    close_column: str,
    start_date,
    end_date,
    param_combinations: list[dict],
    forward_k_candles: int = 12,
    hedge_lookback_candles: int | None = None,
    warmup_candles: int | None = None,
) -> pd.DataFrame:
    """Stufe-1-Screen: reine Signalqualitaet (Information Coefficient) OHNE
    Trade-Simulation, als billiger Vorfilter vor dem vollen Backtest --
    Pendant zu pairtrading_engine.screen_signal_quality.

    Information Coefficient (IC): Signalstaerke
        signal_t = (|Zmax_t| - n_std) / n_std
    (relative Bandverletzung, <=0 falls kein Ausbruch) wird mit der
    nachfolgenden Rueckkehr verglichen:
        reversion_t = -sign_t * (Z_{t+k} - Z_t)
    IC = Pearson-Korrelation zwischen signal_t und reversion_t ueber alle
    Kerzen mit Band-Ausbruch, gepoolt ueber alle Cliquen im Zeitraum.

    Args:
        cliques_df: wie in run_bollinger_backtest, bereits auf Basket(s)/
            Variante gefiltert.
        ohlcv_df: 5-Min-Kerzen-OHLCV mit mindestens timestamp/asset_id/close.
        start_date, end_date: Zeitfenster, in dem gescreent wird (i.d.R. NUR
            das Trainings-/Validierungsfenster eines Walk-Forward-Blocks --
            NIE der finale Testzeitraum).
        param_combinations: Liste von dicts mit mind. den Keys
            "lookback_candles" (int) und "n_std" (float). Jede Kombination
            wird unabhaengig gescreent.
        forward_k_candles: wie viele Kerzen nach dem Signal die Rueckkehr
            gemessen wird.
        hedge_lookback_candles, warmup_candles: siehe run_bollinger_backtest.

    Returns:
        DataFrame, eine Zeile pro Parameterkombination, mit Spalten:
            - lookback_candles, n_std: durchgereichte Eingabeparameter.
            - ic (float): Information Coefficient, NaN falls zu wenige
              Entry-Beobachtungen (< 10) fuer eine stabile Schaetzung.
            - n_signals (int): Anzahl der Band-Ausbrueche, auf denen der IC
              beruht.
    """
    if not param_combinations:
        raise ValueError("param_combinations darf nicht leer sein.")

    max_lookback = max(int(p["lookback_candles"]) for p in param_combinations)
    warmup_candles = warmup_candles or (max_lookback + 30)
    hedge_lookback_candles = hedge_lookback_candles or max_lookback

    price_matrix = ohlcv_df.pivot_table(
        index=timestamp_column, columns=asset_id_column, values=close_column, aggfunc="last",
    ).sort_index()
    ts_index = price_matrix.index

    start_ts = pd.Timestamp(start_date)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(end_date)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    end_ts = end_ts + pd.Timedelta(days=1)
    day_mask = (cliques_df["discovery_date"] >= start_ts) & (cliques_df["discovery_date"] < end_ts)
    target_cliques = cliques_df.loc[day_mask]

    # Preis-Spreads sind UNABHAENGIG von lookback_candles/n_std (nur von der
    # Hedge-Ratio-Schaetzung) -- einmal pro Clique-Tag berechnen und fuer alle
    # Kombinationen wiederverwenden.
    per_day_spreads: list[tuple[pd.DataFrame, pd.DatetimeIndex]] = []
    for clique_row in target_cliques.itertuples():
        day_start = pd.Timestamp(clique_row.discovery_date).normalize()
        day_end = day_start + pd.Timedelta(days=1)

        day_start_pos = ts_index.searchsorted(day_start)
        day_end_pos = ts_index.searchsorted(day_end)
        window_start_pos = max(0, day_start_pos - warmup_candles)
        if day_end_pos <= day_start_pos or day_start_pos - warmup_candles < 0:
            continue

        clique_ids = [a for a in clique_row.asset_ids if a in price_matrix.columns]
        if len(clique_ids) < 2:
            continue

        window_prices = price_matrix.iloc[window_start_pos:day_end_pos][clique_ids]
        if window_prices.empty:
            continue

        spread_df, _hedge_ratios = build_pairwise_price_spreads(
            window_prices, hedge_lookback_candles=hedge_lookback_candles,
        )
        day_start_in_window = spread_df.index[spread_df.index >= day_start]
        per_day_spreads.append((spread_df, day_start_in_window))

    rows = []
    for params in param_combinations:
        lookback_candles = int(params["lookback_candles"])
        n_std = float(params["n_std"])

        signals: list[float] = []
        reversions: list[float] = []

        for spread_df, day_slice in per_day_spreads:
            bands = compute_bollinger_bands(spread_df, lookback_candles=lookback_candles, n_std=n_std)
            max_z_df = compute_max_zscore(bands)
            day_max_z = max_z_df.loc[day_slice]
            entry_rows = day_max_z.loc[day_max_z["zmax"].abs() > n_std]

            for ts in entry_rows.index:
                row = entry_rows.loc[ts]
                pair = row["pair"]
                if pair is None or pair not in bands:
                    continue

                zt = bands[pair]["zscore"]
                pos = zt.index.get_loc(ts)
                target_pos = pos + forward_k_candles
                if target_pos >= len(zt):
                    continue

                z_now = zt.iloc[pos]
                z_future = zt.iloc[target_pos]
                if not (np.isfinite(z_now) and np.isfinite(z_future)):
                    continue

                sign = np.sign(z_now) if z_now != 0 else row["sign"]
                signal_strength = (abs(z_now) - n_std) / n_std
                reversion = -sign * (z_future - z_now)

                signals.append(signal_strength)
                reversions.append(reversion)

        n_signals = len(signals)
        if n_signals >= 10:
            ic = float(np.corrcoef(signals, reversions)[0, 1])
        else:
            ic = np.nan

        rows.append({
            "lookback_candles": lookback_candles, "n_std": n_std,
            "ic": ic, "n_signals": n_signals,
        })

    return pd.DataFrame(rows)
