"""GARCH-Toleranzintervall-Pairtrading-Engine fuer Clique-Mitglieder.

Vierte, PARALLELE Strategie-Engine neben pairtrading_engine.py (EWMA-
Toleranzintervall), bollinger_pairtrading_engine.py (Bollinger-Baender) und
der RSI/Bollinger-Einzelasset-Engine (src/backtest/). Baut auf derselben
Infrastruktur auf (Cliquen-Cache, ohlcv_df), wiederholt sie nicht.

Ablauf (Steps 1-6):
    1.-4. Signalkette (Return-Spread, GARCH(1,1)-Normierung, Xmax, Wilks-
       Toleranzintervall) lebt in src.signals.garch_tolerance/
       src.signals.garch, siehe dortige Modul-Docstrings.
    5. Entry bei Grenzdurchbruch (wie pairtrading_engine.py), Exit sobald
       |Xt| DES GEHANDELTEN PAARES die MITTE zwischen der beim Entry
       geltenden unteren und oberen Wilks-Toleranzgrenze erreicht/
       unterschreitet (simulate_garch_trade) -- siehe Modul-Docstring
       von src.signals.garch_tolerance, Abschnitt "Exit-Mitte".
    6. Orchestrierung ueber alle Cliquen/Baskets (run_garch_backtest).

WICHTIGE ABGRENZUNG zu pairtrading_engine.py: dort schaetzt EWMA
(compute_ewma_volatility, fester decay-Parameter) die Volatilitaet der
Renditedifferenz. HIER wird stattdessen pro Paar ein GARCH(1,1)-Modell
(Student-t-Innovationen, siehe Aufgabenstellung-Zusatz-PDF: "Student-t GARCH
besser geeignet fuer Crypto-Fat-Tails als Standard-GARCH") per Maximum-
Likelihood gefittet -- kein fester decay, sondern datengetriebene
omega/alpha/beta/nu je Paar und Rekalibrierungszeitpunkt.

WICHTIGE ABGRENZUNG beim Exit: pairtrading_engine.py und
bollinger_pairtrading_engine.py exiten, sobald der PREIS-Level-Spread einen
separat berechneten rollierenden Mittelwert (exit_lookback_candles) kreuzt.
HIER exitet der Trade, sobald |Xt| die MITTE zwischen den beiden Wilks-
Toleranzgrenzen erreicht -- kein zusaetzliches Lookback-Fenster, die
Referenzlinie ergibt sich direkt aus den fuer den Entry ohnehin berechneten
Grenzen (siehe src.signals.garch_tolerance-Modul-Docstring).

PairTrade nutzt dieselben Spalten wie pairtrading_engine.PairTrade, damit
pairtrading_engine.compute_backtest_stats/compute_equity_curve/
plot_equity_curve unveraendert auf GARCH-Ergebnissen weiterverwendet werden
koennen -- diese Funktionen werden hier importiert, nicht dupliziert.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Src.signals.garch_tolerance import compute_garch_adjusted_spreads
from Src.signals.tolerance import (
    build_pairwise_return_spreads,
    compute_max_adjusted_spread,
    generate_tolerance_signals,
)

import pairtrading_engine as pte  # compute_equity_curve/compute_backtest_stats/plot_equity_curve wiederverwendet


@dataclass
class GarchTrade:
    """Zwei-Leg-Trade (dollar-neutral: gleiche Notional-Groesse long/short).

    Stellt dieselben Spalten wie pairtrading_engine.PairTrade bereit, damit
    pairtrading_engine.compute_backtest_stats/compute_equity_curve
    unveraendert auf GARCH-Ergebnissen weiterverwendet werden koennen.
    """
    basket: str
    discovery_date: pd.Timestamp
    pair: tuple
    asset_long: int
    asset_short: int
    asset_long_sym: str
    asset_short_sym: str
    direction: str  # "long_j_short_i" | "long_i_short_j" (siehe simulate_garch_trade)
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


def simulate_garch_trade(
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
) -> "GarchTrade | None":
    """Simuliert EINEN Paar-Trade ab entry_time (zwei Legs, dollar-neutral).

    Exit-Bedingungen (erste erfuellte gewinnt): |Xt| DES GEHANDELTEN PAARES
    erreicht/unterschreitet die Mitte zwischen unterer und oberer Wilks-
    Toleranzgrenze (siehe src.signals.garch_tolerance-Modul-Docstring),
    max_holding_candles erreicht, oder exit_deadline erreicht.

    Args:
        price_df: DataFrame (timestamp x asset_id), deckt entry_time bis
            mindestens exit_deadline ab.
        entry_time: Zeitpunkt des Entry-Signals.
        exit_deadline: spaetester moeglicher Exit-Zeitpunkt.
        pair: (asset_id_i, asset_id_j).
        direction: "long_i_short_j" oder "long_j_short_i", siehe
            tolerance.compute_max_adjusted_spread.
        signal_series: dict mit "xt" (GARCH-normierter Spread DIESES Paares)
            und "exit_mid" (Mitte zwischen lower_limit/upper_limit,
            (lower_limit + upper_limit) / 2, aus generate_tolerance_signals).
        max_holding_candles, fee_bps: siehe Modul-Docstring.

    Returns:
        GarchTrade, oder None falls entry_time fehlt oder Preise fehlen.
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

    xt = signal_series["xt"]
    exit_mid = signal_series["exit_mid"]

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
        cur_xt = xt.iloc[i]
        cur_mid = exit_mid.iloc[i]
        if pd.notna(cur_xt) and pd.notna(cur_mid) and abs(cur_xt) <= cur_mid:
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

    return GarchTrade(
        basket=basket, discovery_date=discovery_date, pair=pair,
        asset_long=asset_long, asset_short=asset_short,
        asset_long_sym=id_to_symbol_fallback(asset_long), asset_short_sym=id_to_symbol_fallback(asset_short),
        direction=direction, entry_time=entry_time, exit_time=exit_time,
        entry_price_long=float(entry_price_long), entry_price_short=float(entry_price_short),
        exit_price_long=float(exit_price_long), exit_price_short=float(exit_price_short),
        exit_reason=exit_reason, gross_return=float(gross_return), net_return=float(net_return),
        holding_candles=int(exit_pos - entry_pos),
    )


def run_garch_backtest(
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
    p_content: float,
    confidence: float,
    max_holding_candles: int,
    fee_bps: float,
    trade_mode: str = "max_only",
    recalib_every: int = 1,
    garch_dist: str = "t",
    garch_max_iter: int = 500,
    warmup_candles: int | None = None,
) -> pd.DataFrame:
    """Orchestriert den GARCH-Toleranzintervall-Pairtrading-Backtest ueber
    alle Cliquen/Baskets im gewaehlten Zeitraum.

    trade_mode (wie pairtrading_engine.run_pairtrading_backtest):
        - "max_only": pro Zeitpunkt nur das Paar mit dem groessten |Xt| als
          Handelskandidat.
        - "all_significant": handelt gleichzeitig ALLE Paare, deren |Xt| die
          Toleranzgrenze verletzt.

    Args:
        cliques_df: aus dem Cache, bereits auf Baskets/Variante gefiltert.
        ohlcv_df: 5-Min-Kerzen-OHLCV mit timestamp/asset_id/close.
        lookback_candles, p_content, confidence, recalib_every: siehe
            src.signals.tolerance.generate_tolerance_signals -- steuern den
            Entry UND liefern die Grenzen, deren Mitte den Exit definiert
            (siehe Modul-Docstring).
        max_holding_candles, fee_bps: siehe simulate_garch_trade.
        garch_dist, garch_max_iter: siehe
            src.signals.garch.compute_garch_volatility. GARCH wird PRO
            CLIQUE-TAG neu gefittet (auf dem Warmup-Fenster), da sich die
            Clique-Zusammensetzung von Tag zu Tag aendern kann.
        warmup_candles: Vorlauf vor jedem Discovery-Tag fuer GARCH-Fit und
            Toleranzgrenzen-Lookback. Default: lookback_candles + 30. GARCH
            braucht ausreichend Historie fuer einen stabilen Fit (siehe
            compute_garch_volatility -- < 30 gueltige Beobachtungen liefert
            NaN-Vola), ein zu kurzes Warmup-Fenster fuehrt zu leeren Trades,
            nicht zu einem Fehler.

    Returns:
        DataFrame, eine Zeile pro Trade (GarchTrade-Felder), leer falls
        keine Trades zustande kamen.
    """
    if trade_mode not in ("max_only", "all_significant"):
        raise ValueError(f"trade_mode muss 'max_only' oder 'all_significant' sein, erhalten: {trade_mode}")

    warmup_candles = warmup_candles or (lookback_candles + 30)

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

    all_trades: list[GarchTrade] = []

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

        spread_df = build_pairwise_return_spreads(window_prices)
        adjusted_df, _fit_info = compute_garch_adjusted_spreads(
            spread_df, dist=garch_dist, max_iter=garch_max_iter,
        )
        max_spread_df = compute_max_adjusted_spread(adjusted_df)
        tolerance_df = generate_tolerance_signals(
            max_spread_df, lookback_candles=lookback_candles,
            p_content=p_content, confidence=confidence, recalib_every=recalib_every,
        )
        # Exit-Mitte: (lower_limit + upper_limit) / 2, siehe Modul-Docstring
        # "Exit-Mitte" -- eine Serie, gilt fuer ALLE Paare gleich (die
        # Toleranzgrenzen sind auf Xmax der Clique geschaetzt, nicht pro Paar).
        exit_mid = (tolerance_df["lower_limit"] + tolerance_df["upper_limit"]) / 2.0

        # Gegen tolerance_df.index schneiden, NICHT gegen window_prices.index:
        # build_pairwise_return_spreads -> compute_log_returns dropna't Zeilen,
        # an denen irgendein Clique-Mitglied NaN ist (z.B. ein neu gelistetes
        # Asset mit Datenluecke) -- tolerance_df hat denselben (kuerzeren)
        # Index wie spread_df/adjusted_df. Ohne diesen Schnitt wirft
        # tolerance_df.loc[day_slice] weiter unten einen KeyError, sobald
        # der Discovery-Tag von so einer Datenluecke betroffen ist.
        day_slice_full = window_prices.index[window_prices.index >= day_start]
        day_slice = day_slice_full.intersection(tolerance_df.index)
        exit_deadline = day_slice[-1] if len(day_slice) else window_prices.index[-1]

        if trade_mode == "max_only":
            entry_rows = tolerance_df.loc[day_slice]
            entry_rows = entry_rows.loc[entry_rows["entry"]]
            for ts in entry_rows.index:
                row = entry_rows.loc[ts]
                pair = row["pair"]
                if pair is None or pair not in adjusted_df.columns:
                    continue
                direction = "long_j_short_i" if row["sign"] > 0 else "long_i_short_j"
                xt_pair = adjusted_df[pair]
                signal_series = {"xt": xt_pair, "exit_mid": exit_mid}
                trade = simulate_garch_trade(
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
                xt_pair = adjusted_df[pair]
                pair_abs = xt_pair.abs()
                pair_entry = day_slice[(pair_abs.loc[day_slice] > tolerance_df["upper_limit"].loc[day_slice]).to_numpy()]
                for ts in pair_entry:
                    sign = np.sign(xt_pair.loc[ts])
                    if pd.isna(sign) or sign == 0:
                        continue
                    direction = "long_j_short_i" if sign > 0 else "long_i_short_j"
                    signal_series = {"xt": xt_pair, "exit_mid": exit_mid}
                    trade = simulate_garch_trade(
                        window_prices, ts, exit_deadline, pair, direction,
                        signal_series=signal_series, basket=clique_row.basket, discovery_date=day_start,
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

def plot_xmax_with_garch_bands(
    max_spread_df: pd.DataFrame,
    tolerance_bands_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    ax=None,
    show: bool = True,
):
    """Reproduziert pairtrading_engine.plot_xmax_with_tolerance_bands, ZUSAETZLICH
    mit der Exit-Mitte (lower_limit + upper_limit) / 2 als weiterer Linie --
    zeigt, WARUM ein Trade an genau diesem Punkt geschlossen wurde.

    Args:
        max_spread_df: wie von tolerance.compute_max_adjusted_spread zurueckgegeben.
        tolerance_bands_df: wie von tolerance.generate_tolerance_signals
            zurueckgegeben (Spalten lower_limit/upper_limit), gleicher Index
            wie max_spread_df.
        trades_df: wie von run_garch_backtest zurueckgegeben, EINGESCHRAENKT
            auf den hier geplotteten Zeitraum/Clique.
    """
    import matplotlib.pyplot as plt

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    idx = max_spread_df.index
    exit_mid = (tolerance_bands_df["lower_limit"] + tolerance_bands_df["upper_limit"]) / 2.0

    ax.plot(idx, max_spread_df["xmax"], color="#2a2a28", linewidth=1.1, label="Xmax")
    ax.plot(idx, tolerance_bands_df["upper_limit"].reindex(idx), color="#c1443f", linewidth=1, linestyle="--", label="Obere Toleranzgrenze")
    ax.plot(idx, tolerance_bands_df["lower_limit"].reindex(idx), color="#256abf", linewidth=1, linestyle="--", label="Untere Toleranzgrenze")
    ax.plot(idx, exit_mid.reindex(idx), color="#1baf7a", linewidth=1.2, linestyle="-.", label="Exit-Mitte")

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

    ax.set_title("Xmax mit Wilks-Toleranzgrenzen und Exit-Mitte (GARCH)")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Xmax (GARCH-adjustierter Spread)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    if owns_figure:
        # fig.autofmt_xdate()/fig.tight_layout() vermeiden: auf diesem
        # Windows/conda-Setup fuehrt fig.tight_layout() -- nach vorangegangener
        # scipy.optimize-Nutzung im selben Prozess (GARCH-Fit) -- reproduzierbar
        # zu einem nativen Crash (0xC0000409), isoliert per Debug-Bisektion.
        # plt.setp(...rotation=...) erreicht dieselbe schraege Tick-Beschriftung
        # ohne den betroffenen Layout-Engine-Codepfad.
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        if show:
            plt.show()
            plt.close(fig)
    return ax


def plot_price_spread(
    price_spread: pd.Series,
    trades_df: pd.DataFrame,
    *,
    ax=None,
    show: bool = True,
):
    """Plottet den Preis-Level-Spread EINES Paares mit Entry-/Exit-Markern
    -- Pendant zu bollinger_pairtrading_engine.plot_spread_with_bollinger_bands
    / pairtrading_engine.plot_price_spread_with_exit, hier fuer die GARCH-
    Toleranzintervall-Strategie.

    OHNE Referenzlinie fuer den Exit: anders als bei EWMA/Bollinger ist der
    GARCH-Exit NICHT auf dem Preis-Level-Spread definiert, sondern auf |Xt|
    (dem GARCH-normierten Return-Spread) gegen die Mitte der Wilks-
    Toleranzgrenzen (siehe simulate_garch_trade/Modul-Docstring "Exit-
    Mitte") -- eine Referenzlinie auf DIESEM Preis-Spread-Plot waere
    irrefuehrend, da sie nicht die tatsaechliche Exit-Bedingung abbildet
    (siehe plot_xmax_with_garch_bands fuer den Plot MIT Exit-Mitte, dort
    aber auf |Xmax| statt dem Preis-Spread).

    Args:
        price_spread: EINE Spalte aus build_pairwise_price_spreads's
            spread_df (Preis-Level-Spread des geplotteten Paares) --
            NICHT der GARCH-normierte Return-Spread, der fuer Entry/Exit
            tatsaechlich verwendet wird.
        trades_df: wie von run_garch_backtest zurueckgegeben, EINGESCHRAENKT
            auf das hier geplottete Paar/den Zeitraum.
    """
    import matplotlib.pyplot as plt

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    idx = price_spread.index
    ax.plot(idx, price_spread, color="#2a2a28", linewidth=1.1, label="Preis-Spread")

    if not trades_df.empty:
        for _, row in trades_df.iterrows():
            if row["entry_time"] in idx:
                ax.axvline(row["entry_time"], color="firebrick", linewidth=1, alpha=0.7)
                ax.scatter([row["entry_time"]], [price_spread.loc[row["entry_time"]]], color="firebrick", zorder=5, s=40)
            if row["exit_time"] in idx:
                ax.axvline(row["exit_time"], color="seagreen", linewidth=1, alpha=0.7)
                ax.scatter([row["exit_time"]], [price_spread.loc[row["exit_time"]]], color="seagreen", zorder=5, s=40, marker="s")

    ax.set_title("Preis-Spread (GARCH) -- Exit-Kriterium wirkt auf |Xt|, nicht auf diesen Spread")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Spread (Log-Preis-Differenz)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)

    if owns_figure:
        # siehe plot_xmax_with_garch_bands: fig.tight_layout() nach GARCH-
        # Fits (scipy.optimize) crasht nativ auf diesem Windows/conda-Setup.
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        if show:
            plt.show()
            plt.close(fig)
    return ax
