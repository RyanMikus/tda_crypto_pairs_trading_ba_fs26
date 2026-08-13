"""Pairtrading-Engine fuer Clique-Mitglieder (Lin, Chen & Syu 2021).

Ablauf (Steps 1-6):
    1.-4. Signalkette (Return-Spread, EWMA-Normierung, Xmax, Wilks-
       Toleranzintervall) lebt in src.signals.tolerance, siehe dortigen
       Modul-Docstring.
    5. Entry bei Grenzdurchbruch, Exit bei Rueckkehr/Holding-Limit
       (simulate_pair_trade).
    6. Orchestrierung ueber alle Cliquen/Baskets (run_pairtrading_backtest).

PairTrade nutzt dieselben Spalten wie backtest_engine.Trade, damit
compute_backtest_stats/compute_equity_curve unveraendert weiterverwendet
werden koennen.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

#!############################
from Src.signals.bollinger import build_pairwise_price_spreads, compute_exit_mean
from Src.signals.tolerance import (
    build_pairwise_return_spreads,
    compute_adjusted_spreads,
    compute_ewma_volatility,
    compute_max_adjusted_spread,
    generate_tolerance_signals,
    wilks_tolerance_limits,
)
#!############################

CANDLES_PER_DAY = 24 * 60 // 5  # 288, siehe config.OHLCV_DIR = asset_ohlcv_5m


@dataclass
class PairTrade:
    """Zwei-Leg-Trade (dollar-neutral: gleiche Notional-Groesse long/short).

    Stellt dieselben Spalten (net_return, holding_candles, entry_time,
    exit_time) wie backtest_engine.Trade bereit, damit
    backtest_engine.compute_backtest_stats/compute_equity_curve unveraendert
    auf pairtrading-Ergebnissen weiterverwendet werden koennen.
    """
    basket: str
    discovery_date: pd.Timestamp
    pair: tuple
    asset_long: int
    asset_short: int
    asset_long_sym: str
    asset_short_sym: str
    direction: str  # "long_j_short_i" | "long_i_short_j" (siehe simulate_pair_trade)
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


def simulate_pair_trade(
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
) -> "PairTrade | None":
    """Simuliert EINEN Paar-Trade ab entry_time (zwei Legs, dollar-neutral).

    Exit-Bedingungen (erste erfuellte gewinnt): der PREIS-Level-Spread
    (nicht die Renditedifferenz -- siehe Modul-Docstring "WICHTIGE
    ABGRENZUNG") erreicht/kreuzt seinen EIGENEN rollierenden Mittelwert
    (exit_mean, siehe src.signals.bollinger.compute_exit_mean -- unabhaengiges
    Lookback-Fenster vom Entry-Toleranzintervall), max_holding_candles
    erreicht, oder exit_deadline erreicht. Analog zum Exit-Mechanismus der
    Bollinger-Band-Strategie (bollinger_pairtrading_engine.simulate_bollinger_trade).

    Kreuzung statt Schwellenwert: verglichen wird das Vorzeichen von
    (price_spread_t - exit_mean_t) bei Entry gegen jeden Folgezeitpunkt --
    sobald sich das Vorzeichen aendert (oder exakt 0 erreicht), hat der
    Preis-Spread den Mittelwert erreicht/ueberquert und der Trade wird
    geschlossen.

    Args:
        price_df: DataFrame (timestamp x asset_id), deckt entry_time bis
            mindestens exit_deadline ab.
        entry_time: Zeitpunkt des Entry-Signals.
        exit_deadline: spaetester moeglicher Exit-Zeitpunkt.
        pair: (asset_id_i, asset_id_j).
        direction: "long_i_short_j" oder "long_j_short_i", siehe
            compute_max_adjusted_spread.
        signal_series: dict mit "price_spread" (Preis-Level-Spread dieses
            Paares) und "exit_mean" (rollierender Mittelwert, siehe
            compute_exit_mean) fuer den Rueckkehr-Exit.
        max_holding_candles, fee_bps: siehe Modul-Docstring.

    Returns:
        PairTrade, oder None falls entry_time fehlt oder Preise fehlen.
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

    #!############################
    price_spread_series = signal_series["price_spread"]
    exit_mean_series = signal_series["exit_mean"]

    entry_diff = price_spread_series.iloc[entry_pos] - exit_mean_series.iloc[entry_pos]
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
        cur_diff = price_spread_series.iloc[i] - exit_mean_series.iloc[i]
        if pd.notna(cur_diff) and pd.notna(entry_sign):
            cur_sign = np.sign(cur_diff)
            if cur_sign == 0 or cur_sign != entry_sign:
                exit_pos, exit_reason = i, "signal"
                break
    #!############################

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

    return PairTrade(
        basket=basket, discovery_date=discovery_date, pair=pair,
        asset_long=asset_long, asset_short=asset_short,
        asset_long_sym=id_to_symbol_fallback(asset_long), asset_short_sym=id_to_symbol_fallback(asset_short),
        direction=direction, entry_time=entry_time, exit_time=exit_time,
        entry_price_long=float(entry_price_long), entry_price_short=float(entry_price_short),
        exit_price_long=float(exit_price_long), exit_price_short=float(exit_price_short),
        exit_reason=exit_reason, gross_return=float(gross_return), net_return=float(net_return),
        holding_candles=int(exit_pos - entry_pos),
    )


def run_pairtrading_backtest(
    *,
    cliques_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    id_to_symbol: dict,
    timestamp_column: str,
    asset_id_column: str,
    close_column: str,
    start_date,
    end_date,
    decay: float,
    lookback_candles: int,
    p_content: float,
    confidence: float,
    exit_lookback_candles: int,
    max_holding_candles: int,
    fee_bps: float,
    trade_mode: str = "max_only",
    recalib_every: int = 1,
    hedge_lookback_candles: int | None = None,
    warmup_candles: int | None = None,
) -> pd.DataFrame:
    """Orchestriert den Pairtrading-Backtest ueber alle Cliquen/Baskets im
    gewaehlten Zeitraum.

    trade_mode (Erweiterung ggue. dem Originalpaper):
        - "max_only": repliziert Lin et al. exakt -- pro Zeitpunkt nur das
          Paar mit dem groessten |Xt| als Handelskandidat.
        - "all_significant": handelt gleichzeitig ALLE Paare, deren |Xt| die
          Toleranzgrenze verletzt. Vertretbar, weil TDA-Cliquen bereits
          kleiner/reiner als Lin et al.'s volles Universum sind -- siehe
          Thesis-Diskussion.

    Args:
        cliques_df: aus dem Cache, bereits auf Baskets/Variante gefiltert.
        ohlcv_df: 5-Min-Kerzen-OHLCV mit timestamp/asset_id/close.
        decay, lookback_candles, p_content, confidence, recalib_every: siehe
            compute_ewma_volatility / generate_tolerance_signals -- steuern
            NUR den Entry (Renditedifferenz-Signalkette).
        exit_lookback_candles: EIGENES, von lookback_candles unabhaengiges
            Fenster fuer den rollierenden Mittelwert des PREIS-Level-Spreads,
            gegen den der Exit verglichen wird (siehe
            src.signals.bollinger.compute_exit_mean/simulate_pair_trade) --
            der Trade schliesst, sobald der Preis-Spread diesen Mittelwert
            erreicht/kreuzt. Analog zur Bollinger-Band-Strategie.
        max_holding_candles, fee_bps: siehe simulate_pair_trade.
        hedge_lookback_candles: Fenster fuer die OLS-Hedge-Ratio-Schaetzung
            des Preis-Spreads (siehe build_pairwise_price_spreads). Default:
            exit_lookback_candles.
        warmup_candles: Vorlauf vor jedem Discovery-Tag fuer EWMA-Warmup,
            Toleranzgrenzen-Lookback UND Exit-Mittelwert-Lookback. Default:
            das groessere von lookback_candles/exit_lookback_candles + 30.

    Returns:
        DataFrame, eine Zeile pro Trade (PairTrade-Felder), leer falls keine
        Trades zustande kamen.
    """
    if trade_mode not in ("max_only", "all_significant"):
        raise ValueError(f"trade_mode muss 'max_only' oder 'all_significant' sein, erhalten: {trade_mode}")

    warmup_candles = warmup_candles or (max(lookback_candles, exit_lookback_candles) + 30)
    hedge_lookback_candles = hedge_lookback_candles or exit_lookback_candles

    price_matrix = ohlcv_df.pivot_table(
        index=timestamp_column, columns=asset_id_column, values=close_column, aggfunc="last",
    ).sort_index()
    ts_index = price_matrix.index

    # tz_localize statt tz="UTC"-Konstruktorargument: start_date/end_date koennen
    # bereits tz-aware sein (z.B. fold["is_start"] aus hyperparameter_tuner.py),
    # was pd.Timestamp(..., tz="UTC") mit einem ValueError ablehnt.
    start_ts = pd.Timestamp(start_date)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(end_date)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    end_ts = end_ts + pd.Timedelta(days=1)
    day_mask = (cliques_df["discovery_date"] >= start_ts) & (cliques_df["discovery_date"] < end_ts)
    target_cliques = cliques_df.loc[day_mask]

    all_trades: list[PairTrade] = []

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

        #!############################
        spread_df = build_pairwise_return_spreads(window_prices)
        adjusted_df = compute_adjusted_spreads(spread_df, decay=decay)
        max_spread_df = compute_max_adjusted_spread(adjusted_df)
        tolerance_df = generate_tolerance_signals(
            max_spread_df, lookback_candles=lookback_candles,
            p_content=p_content, confidence=confidence, recalib_every=recalib_every,
        )
        #!############################

        # Exit-Signal (unabhaengig vom Entry): Preis-LEVEL-Spread + eigener
        # rollierender Mittelwert (siehe simulate_pair_trade-Docstring und
        # src.signals.bollinger.compute_exit_mean). Renditedifferenz (oben)
        # bleibt ausschliesslich fuer den Entry zustaendig.
        price_spread_df, _hedge_ratios = build_pairwise_price_spreads(
            window_prices, hedge_lookback_candles=hedge_lookback_candles,
        )
        exit_mean_df = compute_exit_mean(price_spread_df, exit_lookback_candles=exit_lookback_candles)

        day_slice = window_prices.index[window_prices.index >= day_start]
        exit_deadline = day_slice[-1] if len(day_slice) else window_prices.index[-1]

        #!############################
        if trade_mode == "max_only":
            entry_rows = tolerance_df.loc[day_slice]
            entry_rows = entry_rows.loc[entry_rows["entry"]]
            for ts in entry_rows.index:
                row = entry_rows.loc[ts]
                pair = row["pair"]
                if pair is None or pair not in price_spread_df.columns:
                    continue
                direction = "long_j_short_i" if row["sign"] > 0 else "long_i_short_j"
                signal_series = {"price_spread": price_spread_df[pair], "exit_mean": exit_mean_df[pair]}
                trade = simulate_pair_trade(
                    window_prices, ts, exit_deadline, pair, direction,
                    signal_series=signal_series, basket=clique_row.basket, discovery_date=day_start,
                    max_holding_candles=max_holding_candles, fee_bps=fee_bps,
                )
                if trade is not None:
                    trade.asset_long_sym = id_to_symbol.get(trade.asset_long, f"ID:{trade.asset_long}")
                    trade.asset_short_sym = id_to_symbol.get(trade.asset_short, f"ID:{trade.asset_short}")
                    all_trades.append(trade)
        else:  # all_significant
            for pair in adjusted_df.columns:
                if pair not in price_spread_df.columns:
                    continue
                xt_pair = adjusted_df[pair]
                pair_abs = xt_pair.abs()
                # Toleranzgrenzen werden weiterhin auf der CLIQUE-WEITEN Xmax-Historie
                # geschaetzt (Paper-Konvention), aber jedes Paar wird individuell
                # gegen dieselbe upper_limit-Serie geprueft.
                pair_entry = day_slice[(pair_abs.loc[day_slice] > tolerance_df["upper_limit"].loc[day_slice]).to_numpy()]
                for ts in pair_entry:
                    sign = np.sign(xt_pair.loc[ts])
                    if pd.isna(sign) or sign == 0:
                        continue
                    direction = "long_j_short_i" if sign > 0 else "long_i_short_j"
                    signal_series = {"price_spread": price_spread_df[pair], "exit_mean": exit_mean_df[pair]}
                    trade = simulate_pair_trade(
                        window_prices, ts, exit_deadline, pair, direction,
                        signal_series=signal_series, basket=clique_row.basket, discovery_date=day_start,
                        max_holding_candles=max_holding_candles, fee_bps=fee_bps,
                    )
                    if trade is not None:
                        trade.asset_long_sym = id_to_symbol.get(trade.asset_long, f"ID:{trade.asset_long}")
                        trade.asset_short_sym = id_to_symbol.get(trade.asset_short, f"ID:{trade.asset_short}")
                        all_trades.append(trade)
        #!############################

    if not all_trades:
        return pd.DataFrame(columns=[
            "basket", "discovery_date", "pair", "asset_long", "asset_short",
            "asset_long_sym", "asset_short_sym", "direction", "entry_time", "exit_time",
            "entry_price_long", "entry_price_short", "exit_price_long", "exit_price_short",
            "exit_reason", "gross_return", "net_return", "holding_candles",
        ])

    return pd.DataFrame([vars(t) for t in all_trades]).sort_values("entry_time").reset_index(drop=True)


# ============================================================================
# 5. Sensitivitaets-Gitter
# ============================================================================

def run_sensitivity_grid(
    *,
    cliques_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    id_to_symbol: dict,
    timestamp_column: str,
    asset_id_column: str,
    close_column: str,
    start_date,
    end_date,
    decay_values: list[float],
    lookback_candles_values: list[int],
    p_content: float,
    confidence: float,
    exit_lookback_candles: int,
    max_holding_candles: int,
    fee_bps: float,
    trade_mode: str = "max_only",
    initial_capital: float = 10_000.0,
    max_concurrent_positions: int = 10,
) -> pd.DataFrame:
    """Wiederholt run_pairtrading_backtest ueber ein Gitter aus
    (decay, lookback_candles), sammelt Kennzahlen fuer plot_sensitivity_heatmap.

    Lazy-Import von compute_equity_curve/compute_backtest_stats, um einen
    zirkulaeren Modul-Import zu vermeiden.

    Returns:
        Eine Zeile pro (decay, lookback_candles)-Kombination mit Stats-Feldern
        plus den Gitter-Parametern.
    """
    from backtest_engine import compute_backtest_stats, compute_equity_curve

    rows = []
    for decay in decay_values:
        for lookback_candles in lookback_candles_values:
            trades_df = run_pairtrading_backtest(
                cliques_df=cliques_df, ohlcv_df=ohlcv_df, id_to_symbol=id_to_symbol,
                timestamp_column=timestamp_column, asset_id_column=asset_id_column, close_column=close_column,
                start_date=start_date, end_date=end_date,
                decay=decay, lookback_candles=lookback_candles, p_content=p_content, confidence=confidence,
                exit_lookback_candles=exit_lookback_candles,
                max_holding_candles=max_holding_candles, fee_bps=fee_bps, trade_mode=trade_mode,
            )
            equity_curve = compute_equity_curve(
                trades_df, initial_capital=initial_capital, max_concurrent_positions=max_concurrent_positions,
            )
            stats = compute_backtest_stats(trades_df, equity_curve)
            rows.append({"decay": decay, "lookback_candles": lookback_candles, **stats})

    return pd.DataFrame(rows)


# ============================================================================
# 6. Visualisierung
# ============================================================================

def plot_xmax_with_tolerance_bands(
    max_spread_df: pd.DataFrame,
    tolerance_bands_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    ax=None,
    show: bool = True,
):
    """Reproduziert Fig. 3 aus Lin et al. (2021): Xmax_t als Linie, obere/
    untere Toleranzgrenze als gestrichelte Linien, Entry-/Exit-Punkte als
    farbige Marker (Farbe/Form je beteiligtem Paar, Legende mit Paar-Labels).

    Args:
        max_spread_df: wie von compute_max_adjusted_spread zurueckgegeben.
        tolerance_bands_df: wie von generate_tolerance_signals zurueckgegeben
            (Spalten lower_limit/upper_limit), gleicher Index wie max_spread_df.
        trades_df: wie von run_pairtrading_backtest zurueckgegeben,
            EINGESCHRAENKT auf den hier geplotteten Zeitraum/Clique (sonst
            werden Marker aus anderen Cliquen mit eingezeichnet).
    """
    import matplotlib.pyplot as plt

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    idx = max_spread_df.index
    ax.plot(idx, max_spread_df["xmax"], color="#2a2a28", linewidth=1.1, label="Xmax")
    ax.plot(idx, tolerance_bands_df["upper_limit"].reindex(idx), color="#c1443f", linewidth=1, linestyle="--", label="Obere Toleranzgrenze")
    ax.plot(idx, tolerance_bands_df["lower_limit"].reindex(idx), color="#256abf", linewidth=1, linestyle="--", label="Untere Toleranzgrenze")

    if not trades_df.empty:
        pairs = sorted(trades_df["pair"].astype(str).unique())
        markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
        colors = ["#1baf7a", "#eb6834", "#4a3aa7", "#e34948", "#eda100", "#008300", "#e87ba4", "#2a78d6"]
        pair_style = {p: (markers[i % len(markers)], colors[i % len(colors)]) for i, p in enumerate(pairs)}

        for _, row in trades_df.iterrows():
            marker, color = pair_style[str(row["pair"])]
            xmax_at_entry = max_spread_df["xmax"].reindex([row["entry_time"]]).iloc[0]
            if pd.notna(xmax_at_entry):
                ax.scatter([row["entry_time"]], [xmax_at_entry], color=color, marker=marker, s=70, zorder=5,
                           label=f"Entry {row['pair']}" if f"Entry {row['pair']}" not in ax.get_legend_handles_labels()[1] else None)

    ax.set_title("Xmax mit Wilks-Toleranzgrenzen")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Xmax (vola-adjustierter Spread)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    if owns_figure:
        fig.autofmt_xdate()
        fig.tight_layout()
        if show:
            plt.show()
            plt.close(fig)
    return ax


def plot_roi_by_basket_and_clique(trades_df: pd.DataFrame, *, ax=None, show: bool = True):
    """Balkendiagramm: mittlerer net_return je Basket (analog Fig. 4/5 im
    Paper) -- zeigt, welche TDA-Cliquen fuer diese Strategie am
    profitabelsten sind, direkt vergleichbar mit den RSI/Bollinger-Ergebnissen
    je Clique."""
    import matplotlib.pyplot as plt

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    if trades_df.empty:
        ax.text(0.5, 0.5, "Keine Trades", ha="center", va="center", transform=ax.transAxes)
    else:
        by_basket = trades_df.groupby("basket")["net_return"].mean().sort_index() * 100
        ax.bar(by_basket.index.astype(str), by_basket.values, color="#2a78d6", alpha=0.85)
        ax.axhline(0, color="#898781", linewidth=0.8)

    ax.set_title("Ø Net Return je Basket")
    ax.set_xlabel("Basket")
    ax.set_ylabel("Ø Net Return (%)")
    ax.grid(True, axis="y", alpha=0.25)

    if owns_figure:
        fig.tight_layout()
        if show:
            plt.show()
            plt.close(fig)
    return ax


def plot_strategy_comparison(
    equity_curves: dict,
    *,
    initial_capital: float,
    ax=None,
    show: bool = True,
):
    """Ueberlagerte Equity-Kurven mehrerer Strategien/Varianten im selben
    Zeitraum -- z.B. RSI/Bollinger (backtest_engine) vs. EWMA-Toleranz-
    intervall (max_only) vs. EWMA-Toleranzintervall (all_significant).

    Args:
        equity_curves: dict label -> pd.Series (wie von
            backtest_engine.compute_equity_curve zurueckgegeben).
    """
    import matplotlib.pyplot as plt

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7"]
    for i, (label, curve) in enumerate(equity_curves.items()):
        if curve.empty:
            continue
        ax.plot(curve.index, curve.values, color=colors[i % len(colors)], linewidth=1.3, label=label)

    ax.axhline(initial_capital, color="#898781", linewidth=1, linestyle="--", label="Startkapital")
    ax.set_title("Strategie-Vergleich: Equity-Kurven")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Kapital")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    if owns_figure:
        fig.autofmt_xdate()
        fig.tight_layout()
        if show:
            plt.show()
            plt.close(fig)
    return ax


def plot_sensitivity_heatmap(results_grid: pd.DataFrame, *, metric: str = "sharpe", p_content_col: str | None = None, show: bool = True):
    """Heatmap(s): x-Achse=lookback_candles, y-Achse=decay, Zellwert=metric.

    Falls p_content_col gesetzt ist (Spaltenname in results_grid, der den
    verwendeten p_content-Wert je Zeile markiert -- z.B. wenn run_sensitivity_grid
    fuer mehrere p_content-Werte nacheinander aufgerufen und die Ergebnisse
    konkateniert wurden), werden getrennte Subplots je p_content-Wert erzeugt
    (analog Tabelle 3/4 im Paper, p=90% und p=95% nebeneinander).

    Args:
        results_grid: wie von run_sensitivity_grid zurueckgegeben (ggf.
            mehrfach aufgerufen und mit pd.concat zusammengefuehrt, plus
            optional einer p_content_col-Spalte).
        metric: Spaltenname in results_grid, der als Zellwert dient
            (z.B. "sharpe" oder "total_return").
    """
    import matplotlib.pyplot as plt

    groups = [(None, results_grid)] if p_content_col is None else list(results_grid.groupby(p_content_col))
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), squeeze=False)
    axes = axes[0]

    for ax, (group_key, group_df) in zip(axes, groups):
        pivot = group_df.pivot_table(index="decay", columns="lookback_candles", values=metric, aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                if np.isfinite(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7)
        title = f"{metric}" if group_key is None else f"{metric} (p_content={group_key})"
        ax.set_title(title)
        ax.set_xlabel("Lookback (Kerzen)")
        ax.set_ylabel("EWMA-Decay")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
    return axes


# ============================================================================
# Equity-Kurve & Kennzahlen (ehemals backtest_engine.py -- diese Datei wurde
# geloescht; die drei Funktionen unten sind hier nachgebaut, weil sie
# ausschliesslich generisch auf den PairTrade-Spalten (net_return,
# holding_candles, entry_time, exit_time) arbeiten -- siehe PairTrade-
# Docstring weiter oben ("Stellt dieselben Spalten ... bereit, damit
# backtest_engine.compute_backtest_stats/compute_equity_curve unveraendert
# ... weiterverwendet werden koennen"). Fuer diese Pairtrading-Engine
# reicht ein einziger Satz dieser drei Funktionen hier im selben Modul.
# ============================================================================

def compute_equity_curve(
    trades_df: pd.DataFrame, *, initial_capital: float, max_concurrent_positions: int,
) -> pd.Series:
    """Baut eine Equity-Kurve aus abgeschlossenen Trades.

    Kapital wird gleichgewichtet auf bis zu `max_concurrent_positions`
    gleichzeitig offene Positionen aufgeteilt: pro Trade wird
    `capital_at_entry / n_open_at_entry` eingesetzt, `net_return` darauf
    angewendet, der P&L bei Exit dem Gesamtkapital gutgeschrieben. Die Kurve
    wird an jedem Entry- UND Exit-Zeitpunkt ausgewertet (Events sortiert),
    sodass sie sowohl Einstiegs- als auch Austiegs-Sprünge zeigt.

    Args:
        trades_df: DataFrame mit mind. den Spalten entry_time, exit_time,
            net_return (siehe PairTrade).
        initial_capital: Startkapital.
        max_concurrent_positions: Kappt, wie viele gleichzeitig offene
            Positionen bei der Kapitalaufteilung pro Trade berücksichtigt
            werden (Sicherheitsgrenze gegen Überallokation bei vielen
            gleichzeitigen Signalen).

    Returns:
        pd.Series, Index = Zeitpunkte (entry_time/exit_time-Events, sortiert,
        dedupliziert), Werte = Kapitalstand nach dem jeweiligen Event.
    """
    if trades_df.empty:
        return pd.Series([initial_capital], index=[pd.Timestamp.now(tz="UTC")], name="equity")

    trades = trades_df.sort_values("entry_time").reset_index(drop=True)

    # Events: (zeit, ist_entry, trade_index) -- bei Gleichstand zuerst Exits
    # abarbeiten, damit ein Trade, der genau dann endet, wenn ein neuer
    # beginnt, das Kapital fuer den neuen Trade schon wieder freigibt.
    events = []
    for i, row in trades.iterrows():
        events.append((row["entry_time"], 1, i))
        events.append((row["exit_time"], 0, i))
    events.sort(key=lambda e: (e[0], e[1]))

    capital = initial_capital
    open_trades: set[int] = set()
    entry_capital_share: dict[int, float] = {}
    equity_index = []
    equity_values = []

    for time_, is_entry, i in events:
        if is_entry:
            n_open_at_entry = min(len(open_trades) + 1, max_concurrent_positions)
            share = capital / max(n_open_at_entry, 1)
            entry_capital_share[i] = share
            open_trades.add(i)
        else:
            share = entry_capital_share.pop(i, capital / max_concurrent_positions)
            capital += share * trades.loc[i, "net_return"]
            open_trades.discard(i)
        equity_index.append(time_)
        equity_values.append(capital)

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_index), name="equity")
    equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")].sort_index()
    return equity_curve


def compute_backtest_stats(
    trades_df: pd.DataFrame, equity_curve: pd.Series, *, candle_minutes: int = 5,
) -> dict:
    """Kennzahlen aus Trades + Equity-Kurve.

    Sharpe wird trade-basiert berechnet (Mittelwert/Std der net_return je
    Trade) und mit sqrt(Trades pro Jahr) annualisiert, approximiert über die
    durchschnittliche Haltedauer der Trades -- ohne Kalenderauflösung der
    Equity-Kurve, robust auch bei wenigen/unregelmässigen Trades.

    Returns: dict mit n_trades, win_rate, avg_net_return, sharpe,
        max_drawdown, total_return, avg_holding_hours.
    """
    if trades_df.empty:
        return {
            "n_trades": 0, "win_rate": np.nan, "avg_net_return": np.nan,
            "sharpe": np.nan, "max_drawdown": np.nan, "total_return": np.nan,
            "avg_holding_hours": np.nan,
        }

    net_returns = trades_df["net_return"]
    n_trades = len(trades_df)
    win_rate = float((net_returns > 0).mean())
    avg_net_return = float(net_returns.mean())

    avg_holding_candles = float(trades_df["holding_candles"].mean())
    avg_holding_hours = avg_holding_candles * candle_minutes / 60.0

    std_return = net_returns.std(ddof=1)
    if std_return > 0 and avg_holding_hours > 0:
        trades_per_year = (365.0 * 24.0) / avg_holding_hours
        sharpe = float(avg_net_return / std_return * np.sqrt(trades_per_year))
    else:
        sharpe = np.nan

    if len(equity_curve) > 0:
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0
        max_drawdown = float(drawdown.min())
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
    else:
        max_drawdown = np.nan
        total_return = np.nan

    return {
        "n_trades": n_trades, "win_rate": win_rate, "avg_net_return": avg_net_return,
        "sharpe": sharpe, "max_drawdown": max_drawdown, "total_return": total_return,
        "avg_holding_hours": avg_holding_hours,
    }


def plot_equity_curve(equity_curve: pd.Series, *, initial_capital: float, show: bool = True):
    """Plottet die Equity-Kurve gegen das Startkapital als Referenzlinie."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_curve.index, equity_curve.values, color="steelblue", linewidth=1.2)
    ax.axhline(initial_capital, color="grey", linestyle="--", linewidth=0.8, label="Startkapital")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Kapital")
    ax.set_title("Equity-Kurve")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
    return ax


# ============================================================================
# Zweistufiges Hyperparameter-Tuning: Stufe 1 (guenstiger Signal-Screen) +
# Walk-Forward-Zeitfenster fuer Stufe 2 (Optuna, siehe tune_backtest.py).
# ============================================================================

def screen_signal_quality(
    cliques_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    *,
    timestamp_column: str,
    asset_id_column: str,
    close_column: str,
    start_date,
    end_date,
    param_combinations: list[dict],
    ewma_decay: float = 0.94,
    confidence: float = 0.95,
    forward_k_candles: int = 12,
    warmup_candles: int | None = None,
) -> pd.DataFrame:
    """Stufe-1-Screen: reine Signalqualitaet (Information Coefficient) OHNE
    Trade-Simulation, als billiger Vorfilter vor dem vollen Backtest.

    Nutzt dieselbe Signal-Kette wie run_pairtrading_backtest (Schritte 1-4),
    ohne simulate_pair_trade/Positionsverwaltung -- daher deutlich schneller.

    Information Coefficient (IC): Signalstaerke
        signal_t = (|Xmax_t| - upper_limit_t) / upper_limit_t
    (relative Grenzverletzung, <=0 falls kein Durchbruch) wird mit der
    nachfolgenden Rueckkehr verglichen:
        reversion_t = -sign_t * (Xt_{t+k} - Xt_t)
    IC = Pearson-Korrelation zwischen signal_t und reversion_t ueber alle
    Kerzen mit Entry-Durchbruch, gepoolt ueber alle Cliquen im Zeitraum.
    Hoher IC = Indiz fuer handelbare Signale (kein Beweis).

    Args:
        cliques_df: wie in run_pairtrading_backtest, bereits auf Basket(s)/
            Variante gefiltert.
        ohlcv_df: 5-Min-Kerzen-OHLCV mit mindestens timestamp/asset_id/close.
        start_date, end_date: Zeitfenster, in dem gescreent wird (i.d.R. NUR
            das Trainings-/Validierungsfenster eines Walk-Forward-Blocks,
            siehe generate_walkforward_windows -- NIE der finale Testzeitraum).
        param_combinations: Liste von dicts mit mind. den Keys
            "lookback_candles" (int) und "p_content" (float). Jede
            Kombination wird unabhaengig gescreent.
        ewma_decay, confidence: fest gehaltene Parameter (siehe Aufgaben-
            stellung -- nicht Teil des Screens/der Optuna-Suche).
        forward_k_candles: wie viele Kerzen nach dem Signal die Rueckkehr
            gemessen wird (Analogon zu einer kurzen Haltedauer).
        warmup_candles: Vorlauf pro Clique-Discovery-Tag, wie in
            run_pairtrading_backtest. Default: groesster lookback_candles-
            Wert aus param_combinations + 30.

    Returns:
        DataFrame, eine Zeile pro Parameterkombination, mit Spalten:
            - lookback_candles, p_content: durchgereichte Eingabeparameter.
            - ic (float): Information Coefficient (Pearson-Korrelation),
              NaN falls zu wenige Entry-Beobachtungen (< 10) fuer eine
              stabile Schaetzung vorliegen.
            - n_signals (int): Anzahl der Entry-Durchbrueche, auf denen der
              IC beruht (fuer Optuna als Mindestqualitaetskriterium nutzbar).
    """
    if not param_combinations:
        raise ValueError("param_combinations darf nicht leer sein.")

    max_lookback = max(int(p["lookback_candles"]) for p in param_combinations)
    warmup_candles = warmup_candles or (max_lookback + 30)

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

    # Spread/Xmax sind UNABHAENGIG von lookback_candles/p_content (nur von
    # ewma_decay) -- einmal pro Clique-Tag berechnen und fuer alle
    # Kombinationen wiederverwenden, statt sie pro Kombination neu zu bauen.
    per_day_max_spread: list[tuple[pd.DataFrame, pd.DataFrame]] = []
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

        #!############################
        spread_df = build_pairwise_return_spreads(window_prices)
        adjusted_df = compute_adjusted_spreads(spread_df, decay=ewma_decay)
        max_spread_df = compute_max_adjusted_spread(adjusted_df)
        #!############################
        day_start_in_window = spread_df.index[spread_df.index >= day_start]
        per_day_max_spread.append((max_spread_df, adjusted_df, day_start_in_window))

    rows = []
    for params in param_combinations:
        lookback_candles = int(params["lookback_candles"])
        p_content = float(params["p_content"])

        signals: list[float] = []
        reversions: list[float] = []

        for max_spread_df, adjusted_df, day_slice in per_day_max_spread:
            #!############################
            tolerance_df = generate_tolerance_signals(
                max_spread_df, lookback_candles=lookback_candles,
                p_content=p_content, confidence=confidence, recalib_every=1,
            )
            day_tolerance = tolerance_df.loc[day_slice]
            entry_rows = day_tolerance.loc[day_tolerance["entry"]]
            #!############################

            for ts in entry_rows.index:
                row = entry_rows.loc[ts]
                pair = row["pair"]
                if pair is None or pair not in adjusted_df.columns:
                    continue
                upper = row["upper_limit"]
                if not np.isfinite(upper) or upper == 0:
                    continue

                xt_pair = adjusted_df[pair]
                pos = xt_pair.index.get_loc(ts)
                target_pos = pos + forward_k_candles
                if target_pos >= len(xt_pair):
                    continue

                xt_now = xt_pair.iloc[pos]
                xt_future = xt_pair.iloc[target_pos]
                if not (np.isfinite(xt_now) and np.isfinite(xt_future)):
                    continue

                sign = np.sign(xt_now) if xt_now != 0 else row["sign"]
                # xt_now ist der adjustierte Spread DES Entry-Paares zum
                # Entry-Zeitpunkt -- fuer dieses Paar identisch zu xmax_t,
                # da genau dieses Paar den Grenzdurchbruch ausgeloest hat
                # (siehe compute_max_adjusted_spread/generate_tolerance_signals).
                signal_strength = (abs(xt_now) - upper) / upper
                reversion = -sign * (xt_future - xt_now)

                signals.append(signal_strength)
                reversions.append(reversion)

        n_signals = len(signals)
        if n_signals >= 10:
            ic = float(np.corrcoef(signals, reversions)[0, 1])
        else:
            ic = np.nan

        rows.append({
            "lookback_candles": lookback_candles, "p_content": p_content,
            "ic": ic, "n_signals": n_signals,
        })

    return pd.DataFrame(rows)


def generate_walkforward_windows(
    start, end, *, train_days: int, val_days: int, step_days: int, embargo_days: int = 0,
) -> list[dict]:
    """Baut eine Liste von Walk-Forward-Zeitfenstern fuer das Tuning.

    Zerlegt [start, end) in (train, val)-Bloecke, die je Iteration um
    step_days vorruecken. Optionaler Embargo-Puffer zwischen train_end und
    val_start vermeidet Leakage durch ueberlappende Lookback-Historie.

    [start, end) ist nur fuer Tuning -- der finale Testzeitraum liegt
    ausserhalb und wird hier nie beruehrt (siehe assert_no_test_period_leakage).

    Args:
        start, end: Trainings-/Validierungszeitraum (nicht der Testzeitraum).
        train_days, val_days: Fensterlaengen in Tagen.
        step_days: Schrittweite je Iteration.
        embargo_days: Puffer zwischen train_end und val_start.

    Returns:
        Liste von dicts (train_start, train_end, val_start, val_end,
        train_end/val_end exklusiv). Leer, falls kein Fenster passt.
    """
    if train_days <= 0 or val_days <= 0 or step_days <= 0:
        raise ValueError("train_days, val_days, step_days muessen > 0 sein.")
    if embargo_days < 0:
        raise ValueError("embargo_days darf nicht negativ sein.")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    windows = []
    train_start = start_ts
    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        val_start = train_end + pd.Timedelta(days=embargo_days)
        val_end = val_start + pd.Timedelta(days=val_days)
        if val_end > end_ts:
            break
        windows.append({
            "train_start": train_start, "train_end": train_end,
            "val_start": val_start, "val_end": val_end,
        })
        train_start = train_start + pd.Timedelta(days=step_days)

    return windows


def assert_no_test_period_leakage(windows: list[dict], test_start, test_end) -> None:
    """Sicherung gegen Datenlecks: bricht hart ab, falls IRGENDEIN
    Walk-Forward-Fenster (Training ODER Validierung) in den finalen
    Testzeitraum hineinreicht.

    Wird von tune_backtest.py vor JEDEM Optuna-Trial aufgerufen (nicht nur
    einmal), damit ein versehentlich falsch konfigurierter Suchraum nicht
    still und leise den finalen Testzeitraum kontaminiert -- genau das
    Risiko, das die Aufgabenstellung mit "Sicherung einbauen" adressiert.

    Args:
        windows: wie von generate_walkforward_windows zurueckgegeben.
        test_start, test_end: finaler, waehrend des gesamten Tunings
            unberuehrter Testzeitraum.

    Raises:
        AssertionError: falls ein Fenster train_start/train_end/val_start/
            val_end in [test_start, test_end) liegt oder es ueberlappt.
    """
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    for w in windows:
        for key in ("train_start", "train_end", "val_start", "val_end"):
            ts = w[key]
            assert not (test_start_ts <= ts < test_end_ts), (
                f"Walk-Forward-Fenster {w} verletzt die Zeittrennung: "
                f"{key}={ts} liegt im finalen Testzeitraum "
                f"[{test_start_ts}, {test_end_ts})."
            )
        # Zusaetzlich: kompletter Ueberlapp der Intervalle pruefen, nicht nur
        # der Randpunkte (falls test-Fenster vollstaendig INNERHALB eines
        # train/val-Fensters liegt).
        assert not (w["train_start"] < test_end_ts and test_start_ts < w["train_end"]), (
            f"Walk-Forward-Trainingsfenster {w} ueberlappt den finalen Testzeitraum."
        )
        assert not (w["val_start"] < test_end_ts and test_start_ts < w["val_end"]), (
            f"Walk-Forward-Validierungsfenster {w} ueberlappt den finalen Testzeitraum."
        )
