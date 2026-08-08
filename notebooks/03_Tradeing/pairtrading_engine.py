"""EWMA-Toleranzintervall-Pairtrading-Engine fuer Clique-Mitglieder (Lin, Chen &
Syu 2021, "Multi-asset pair-trading strategy: A statistical learning approach").

Zweite, PARALLELE Strategie-Engine zu backtest_engine.py (RSI/Bollinger pro
Einzelasset). Baut auf derselben Infrastruktur auf, wiederholt sie nicht:
    - Cliquen kommen aus dem Cache in data/processed/cliques/
      (02_tda_clustering/05_clique_cache.ipynb). Keine eigene Clique-Discovery.
    - 5-Min-Kerzen-Preise/Returns kommen wie in backtest_engine.py/
      signalfeatures.py direkt aus ohlcv_df (Pivot auf asset_id).

WICHTIGE ABGRENZUNG (nicht mit dem Preis-Level-Spread aus der ADF/KPSS/Hurst-
Stationaritaetsdiagnostik in 03_Tradeing/TradeData.ipynb verwechseln): dort
wurde die Renditedifferenz durch einen Preis-Level-Spread ersetzt, weil DAS
der richtige Prozess fuer klassische Stationaritaetstests ist. HIER, fuer die
Lin-et-al.-Signalgenerierung, ist die Renditedifferenz

    Rt^(Ai,Aj) = r_Ai,t - r_Aj,t

selbst die korrekte, vom Paper vorgeschriebene Groesse (Log-Return-Differenz,
KEIN Hedge-Ratio, KEIN Preis-Level-Spread) -- sie wird unveraendert so
verwendet, auf sie wendet das Paper seine EWMA-Volatilitaetsbereinigung und
Toleranzintervalle an.

Ablauf (Lin et al. 2021, Steps 1-6):
    1. Fuer jedes Paar innerhalb einer Clique: Rt = r_Ai,t - r_Aj,t
       (build_pairwise_return_spreads).
    2. EWMA-Volatilitaet pro Paar schaetzen, Rt darauf normieren:
       Xt = Rt / sigma_t (compute_ewma_volatility, compute_adjusted_spreads).
    3. Ueber alle Paare einer Clique: Xmax_t = max(|Xt|) je Zeitpunkt, plus
       welches Paar/welche Richtung (compute_max_adjusted_spread).
    4. Rollierendes Fenster von Xmax-Werten -> nichtparametrisches Wilks-
       Toleranzintervall (wilks_tolerance_limits) -> 1-Schritt-Prognose-
       Vergleich (generate_tolerance_signals).
    5. Durchbruch der Toleranzgrenze = Entry (dollar-neutrales Paar-Long/Short),
       Rueckkehr zur Mitte oder Holding-Limit = Exit (simulate_pair_trade).
    6. Orchestrierung ueber alle Cliquen/Baskets (run_pairtrading_backtest).

Bausteine:
    - compute_ewma_volatility, build_pairwise_return_spreads,
      compute_adjusted_spreads, compute_max_adjusted_spread: Schritte 1-3.
    - wilks_tolerance_limits: nichtparametrisches Toleranzintervall (Kern-
      Baustein, siehe eigener Docstring -- durch test_pairtrading_engine.py
      gegen Referenzwerte abgesichert).
    - generate_tolerance_signals, simulate_pair_trade, run_pairtrading_backtest:
      Schritte 4-6, Signal- und Trade-Ebene.
    - run_sensitivity_grid: Parameter-Gitter-Orchestrierung fuer die
      Sensitivitaets-Heatmap.
    - Plot-Funktionen: Xmax+Toleranzband, ROI je Basket/Clique,
      Strategie-Vergleich (vs. backtest_engine), Sensitivitaets-Heatmap.

PairTrade (dataclass) stellt dieselben Spalten (net_return, holding_candles,
entry_time, exit_time) wie backtest_engine.Trade bereit, damit
backtest_engine.compute_backtest_stats/compute_equity_curve unveraendert auf
den Ergebnissen dieser Engine weiterverwendet werden koennen.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

CANDLES_PER_DAY = 24 * 60 // 5  # 288, siehe config.OHLCV_DIR = asset_ohlcv_5m


# ============================================================================
# 1. EWMA-Volatilitaetsschaetzung
# ============================================================================

def compute_ewma_volatility(returns: pd.Series, *, decay: float) -> pd.Series:
    """RiskMetrics-EWMA nach Lin et al. (2021), Gl. 4A:

        sigma_t^2 = (1 - decay) * a_{t-1}^2 + decay * sigma_{t-1}^2

    a_t = returns - mean(returns) (demeaned, ueber die GESAMTE uebergebene
    Serie -- das Paper demeant vor der EWMA-Rekursion, nicht rollierend).
    sigma_0^2 wird als Mittelwert der quadrierten Anfangsbeobachtungen
    initialisiert (Paper, Abschnitt 2, "Implementation of the EWMA..."),
    konkret: Mittelwert von a_t^2 ueber die ersten min(30, len(returns))
    Beobachtungen (RiskMetrics-Standardpraxis fuer die Anlaufphase, da das
    Paper selbst keine exakte Fenstergroesse fuer sigma_0 nennt).

    WICHTIG -- Kerzengranularitaet: Der im Paper genannte Standard-Decay
    delta=0.94 ist fuer TAEGLICHE Daten kalibriert (RiskMetrics-Konvention).
    Dieses Projekt arbeitet mit 5-Minuten-Kerzen (288 Kerzen/Tag) -- ein
    Decay von 0.94 impliziert auf Kerzenbasis eine sehr viel kuerzere
    effektive Halbwertszeit als auf Tagesbasis. `decay` bleibt deshalb ein
    PFLICHT-Parameter (kein Default in dieser Funktion) und sollte ueber
    run_sensitivity_grid/plot_sensitivity_heatmap empirisch geprueft werden,
    welcher Wert fuer 5-Min-Granularitaet sinnvoll ist -- 0.94 ist hier nur
    ein Startwert, keine uebernommene Kalibrierung.

    Args:
        returns: 1D-Return-Serie (z.B. eine Spalte aus
            build_pairwise_return_spreads), darf NaN am Anfang enthalten
            (werden uebersprungen, sigma bleibt dort NaN).
        decay: EWMA-Decay-Faktor in (0, 1), naeher an 1 = laengeres Gedaechtnis.

    Returns:
        Series gleicher Laenge/Index wie returns, sigma_t (nicht sigma_t^2).
        Erste gueltige Werte vor Erreichen der Init-Fensterlaenge sind NaN.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError(f"decay muss in (0, 1) liegen, erhalten: {decay}")

    values = returns.to_numpy(dtype=float)
    n = len(values)
    sigma2 = np.full(n, np.nan)

    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 2:
        return pd.Series(sigma2, index=returns.index)

    mean_val = np.nanmean(values)
    a = values - mean_val  # demeaned, siehe Docstring

    first_valid_idx = np.flatnonzero(valid_mask)[0]
    init_window = min(30, valid_mask.sum())
    init_indices = np.flatnonzero(valid_mask)[:init_window]
    sigma2_prev = float(np.mean(a[init_indices] ** 2))

    # Erster Punkt, ab dem eine EWMA-Schaetzung existiert: direkt nach dem
    # Init-Fenster. Fuer Punkte davor bleibt sigma2 NaN (nicht genug Historie).
    last_init_idx = init_indices[-1]
    sigma2[last_init_idx] = sigma2_prev

    prev_a = a[last_init_idx]
    for i in range(last_init_idx + 1, n):
        if not valid_mask[i - 1] or not valid_mask[i]:
            sigma2[i] = np.nan
            continue
        sigma2_prev = (1.0 - decay) * prev_a ** 2 + decay * sigma2_prev
        sigma2[i] = sigma2_prev
        prev_a = a[i]

    return pd.Series(np.sqrt(sigma2), index=returns.index)


# ============================================================================
# 2. Intra-Clique-Spreads (Lin et al. Steps 1-3)
# ============================================================================

def build_pairwise_return_spreads(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Rt^(Ai,Aj) = r_Ai,t - r_Aj,t fuer ALLE Paare innerhalb einer Clique.

    Args:
        returns_df: DataFrame (timestamp x asset_id), bereits auf die
            Mitglieder EINER Clique gefiltert (Log-Returns, siehe
            signalfeatures.build_returns_matrix -- dort wird pct_change()
            verwendet; fuer die kleinen 5-Min-Returns hier ist die Differenz
            zu log-Returns vernachlaessigbar, konsistent mit der bestehenden
            Codebasis wird dieselbe Returns-Matrix wiederverwendet).

    Returns:
        DataFrame (timestamp x pair), pair als Tuple (asset_id_i, asset_id_j)
        mit i < j (per Konvention, Vorzeichen wird erst in
        compute_max_adjusted_spread interpretiert). Eine Spalte pro
        Kombination aus 2 der len(returns_df.columns) Assets.
    """
    asset_ids = list(returns_df.columns)
    spreads = {}
    for a_i, a_j in combinations(asset_ids, 2):
        spreads[(a_i, a_j)] = returns_df[a_i] - returns_df[a_j]
    return pd.DataFrame(spreads, index=returns_df.index)


def compute_adjusted_spreads(spread_df: pd.DataFrame, *, decay: float) -> pd.DataFrame:
    """Xt = Rt / sigma_t fuer jedes Paar einzeln (Gl. 7 im Paper).

    sigma_t kommt aus compute_ewma_volatility, UNABHAENGIG pro Spalte
    (Paar) geschaetzt -- jedes Paar hat seine eigene EWMA-Volatilitaets-
    historie, kein gemeinsames sigma ueber die Clique.

    Args:
        spread_df: DataFrame (timestamp x pair), siehe
            build_pairwise_return_spreads.
        decay: siehe compute_ewma_volatility.

    Returns:
        DataFrame gleicher Form wie spread_df, normierte Spreads Xt.
        sigma_t == 0 oder NaN ergibt NaN (kein Divide-by-zero).
    """
    adjusted = {}
    for pair in spread_df.columns:
        sigma = compute_ewma_volatility(spread_df[pair], decay=decay)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = spread_df[pair] / sigma.replace(0.0, np.nan)
        adjusted[pair] = x
    return pd.DataFrame(adjusted, index=spread_df.index)


def compute_max_adjusted_spread(adjusted_df: pd.DataFrame) -> pd.DataFrame:
    """Xmax_t = max(|Xt| ueber alle Paare) je Zeitpunkt, plus welches Paar
    und in welche Richtung (Vorzeichen) den Extremwert erzeugt hat.

    Args:
        adjusted_df: DataFrame (timestamp x pair) normierter Spreads,
            siehe compute_adjusted_spreads.

    Returns:
        DataFrame (Index=timestamp) mit Spalten:
            - xmax (float): max(|Xt|) ueber alle Paare an diesem Zeitpunkt,
              NaN falls an diesem Zeitpunkt kein Paar einen gueltigen Wert hat.
            - pair (object): das Tuple (asset_id_i, asset_id_j) des Extremwerts.
            - sign (float): Vorzeichen des zugrundeliegenden Xt (nicht von
              |Xt|) -- sign=+1 bedeutet R_t = r_i - r_j war ueberdurchschnittlich
              POSITIV (Asset i outperformt Asset j relativ zur normalen
              Vola-bereinigten Streuung) -> Handelsrichtung "Short Asset i /
              Long Asset j" (Reversion-Erwartung: i faellt relativ, j steigt
              relativ). sign=-1 entsprechend umgekehrt: Long Asset i / Short
              Asset j.
    """
    abs_df = adjusted_df.abs()
    rows = []
    for ts, row in abs_df.iterrows():
        if row.isna().all():
            rows.append({"xmax": np.nan, "pair": None, "sign": np.nan})
            continue
        best_pair = row.idxmax()
        xmax = row[best_pair]
        raw_x = adjusted_df.at[ts, best_pair]
        sign = np.sign(raw_x) if pd.notna(raw_x) else np.nan
        rows.append({"xmax": xmax, "pair": best_pair, "sign": sign})
    return pd.DataFrame(rows, index=adjusted_df.index)


# ============================================================================
# 3. Wilks nichtparametrisches Toleranzintervall
# ============================================================================

def wilks_tolerance_limits(sample: np.ndarray, *, p_content: float, confidence: float) -> tuple[float, float]:
    """Nichtparametrisches (p, 1-alpha) zweiseitiges Toleranzintervall nach
    Wilks (1941), wie in Lin et al. (2021) Gl. (1)-(3) verwendet.

    Fuer eine Stichprobe der Groesse n mit Ordnungsstatistiken
    X_(1) <= ... <= X_(n) gilt: F(X_(k)) folgt (verteilungsfrei) einer
    Beta(k, n-k+1)-Verteilung. Ein zweiseitiges Toleranzintervall
    [X_(r), X_(s)] mit r < s deckt mit Konfidenz (1-alpha) mindestens den
    Anteil p_content der Grundgesamtheit ab, wenn

        P(F(X_(s)) - F(X_(r)) >= p_content) >= confidence.

    Symmetrische Wahl (wie im R-Paket "tolerance" von Young 2010, das das
    Paper referenziert): fuer eine gegebene Breite w = s - r wird das kleinste
    w gesucht, sodass ein zentriertes (oder so nah wie moeglich zentriertes)
    Paar (r, s) = (floor((n-w)/2)+1, floor((n-w)/2)+w) die Konfidenz-
    Bedingung erfuellt. Die Konfidenz-Bedingung selbst wird EXAKT ueber die
    Verteilung von F(X_(s)) - F(X_(r)) berechnet, welche (fuer r < s) einer
    Beta(s-r, n-s+r+1)-Verteilung folgt (Standardresultat fuer Ordnungs-
    statistik-Differenzen, siehe z.B. David & Nagaraja, "Order Statistics"):

        P(F(X_(s)) - F(X_(r)) >= p_content) = 1 - Beta.cdf(p_content; s-r, n-s+r+1)

    Es wird das kleinste w >= 1 gesucht (kleinstes Intervall, das noch die
    Konfidenz erreicht), beginnend bei w=n-1 (r=1, s=n, volle Stichprobe) und
    absteigend geprueft -- iteriert wird über w von n-1 bis 1 und beim ersten
    w gestoppt, das NICHT mehr die Bedingung erfuellt; das zuletzt gueltige
    (r, s) wird zurueckgegeben. Existiert selbst fuer w=n-1 keine gueltige
    Loesung, werden die Stichproben-Extremwerte (min, max) zurueckgegeben
    und deren tatsaechlich erreichte Konfidenz ist NICHT garantiert
    (Stichprobe zu klein fuer die geforderte (p_content, confidence)-Kombination).

    Args:
        sample: 1D-Array historischer Beobachtungen (z.B. Xmax-Werte im
            Rueckblickfenster). NaN-Werte werden vor der Berechnung entfernt.
        p_content: geforderter Anteil der Grundgesamtheit, der abgedeckt
            werden soll (z.B. 0.95).
        confidence: Konfidenzniveau (1-alpha), mit dem die p_content-Abdeckung
            gilt (z.B. 0.95).

    Returns:
        (lower_limit, upper_limit) als Werte AUS `sample` (Ordnungsstatistiken
        X_(r), X_(s)), keine interpolierten Werte.

    Raises:
        ValueError: falls nach Entfernen von NaN weniger als 3 Beobachtungen
            uebrig bleiben (kein sinnvolles Toleranzintervall moeglich).
    """
    clean = np.sort(np.asarray(sample, dtype=float))
    clean = clean[~np.isnan(clean)]
    n = len(clean)
    if n < 3:
        raise ValueError(f"wilks_tolerance_limits braucht mindestens 3 gueltige Beobachtungen, erhalten: {n}")

    best_r, best_s = 1, n  # 1-indiziert, wie in der Ordnungsstatistik-Konvention

    for w in range(n - 1, 0, -1):
        r = (n - w) // 2 + 1
        s = r + w
        if s > n:
            continue
        # P(F(X_(s)) - F(X_(r)) >= p_content) = 1 - Beta_{s-r, n-s+r+1}.cdf(p_content)
        prob_meets_content = 1.0 - beta_dist.cdf(p_content, s - r, n - s + r + 1)
        if prob_meets_content >= confidence:
            best_r, best_s = r, s
        else:
            break

    lower_limit = float(clean[best_r - 1])
    upper_limit = float(clean[best_s - 1])
    return lower_limit, upper_limit


# ============================================================================
# 4. Signal-Generierung & Trade-Simulation (Steps 4-6 im Paper)
# ============================================================================

def generate_tolerance_signals(
    max_spread_df: pd.DataFrame,
    *,
    lookback_candles: int,
    p_content: float,
    confidence: float,
    recalib_every: int = 1,
) -> pd.DataFrame:
    """Rollierendes Fenster -> Wilks-Toleranzgrenzen -> 1-Schritt-Vergleich.

    Fuer jeden Rekalibrierungszeitpunkt t (alle recalib_every Kerzen) werden
    die Toleranzgrenzen aus dem zurueckliegenden Fenster
    max_spread_df["xmax"][t-lookback_candles : t] geschaetzt (KEIN Blick auf
    xmax[t] selbst -- kausal). Diese Grenzen gelten dann fuer alle Kerzen bis
    zur naechsten Rekalibrierung; an jeder einzelnen Kerze wird ihr xmax-Wert
    gegen die zuletzt geschaetzten Grenzen verglichen (das entspricht der
    "1-Schritt-Prognose" aus dem Paper bei recalib_every=1, und einer
    guenstigeren Approximation bei recalib_every>1).

    Args:
        max_spread_df: DataFrame wie von compute_max_adjusted_spread
            zurueckgegeben (Spalten xmax, pair, sign).
        lookback_candles: Fenstergroesse fuer die Toleranzgrenzen-Schaetzung,
            in Anzahl Kerzen (Aequivalent zu den 60/80/100 Tagesbars im Paper
            -- siehe Docstring von compute_ewma_volatility zur Kerzen-vs-Tage-
            Kalibrierungsfrage).
        p_content, confidence: siehe wilks_tolerance_limits.
        recalib_every: alle wie viele Kerzen die Grenzen neu geschaetzt
            werden (1 = jede Kerze neu, wie im Paper Step 6; hoehere Werte
            sind eine Performance-Abkuerzung, keine Paper-Vorgabe).

    Returns:
        DataFrame gleicher Index wie max_spread_df, mit Spalten:
            - lower_limit, upper_limit (float): zuletzt geschaetzte
              Toleranzgrenzen (NaN, solange kein volles Lookback-Fenster
              verfuegbar ist).
            - entry (bool): True, wenn xmax an dieser Kerze upper_limit
              ueberschreitet (Betrags-Durchbruch, siehe compute_max_adjusted_spread).
            - pair, sign: durchgereicht aus max_spread_df, fuer den
              Entry-Zeitpunkt relevant.
    """
    xmax = max_spread_df["xmax"].to_numpy()
    n = len(xmax)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)

    last_lower, last_upper = np.nan, np.nan
    for i in range(n):
        if i >= lookback_candles and (i - lookback_candles) % recalib_every == 0:
            window = xmax[i - lookback_candles:i]
            window = window[~np.isnan(window)]
            if len(window) >= 3:
                try:
                    last_lower, last_upper = wilks_tolerance_limits(
                        window, p_content=p_content, confidence=confidence,
                    )
                except ValueError:
                    last_lower, last_upper = np.nan, np.nan
        lower[i] = last_lower
        upper[i] = last_upper

    entry = (xmax > upper) & ~np.isnan(xmax) & ~np.isnan(upper)

    out = pd.DataFrame({
        "lower_limit": lower, "upper_limit": upper, "entry": entry,
        "pair": max_spread_df["pair"].to_numpy(), "sign": max_spread_df["sign"].to_numpy(),
    }, index=max_spread_df.index)
    return out


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
    """Simuliert EINEN Paar-Trade ab entry_time, State-Machine-Stil analog zu
    backtest_engine.simulate_trades, aber mit zwei Legs.

    Exit-Bedingungen (erste erfuellte gewinnt): Rueckkehr des adjustierten
    Spreads Xt in Richtung 0 (Vorzeichenwechsel oder |Xt| unterschreitet die
    untere Toleranzgrenze, analog Ernie-Chan-Stil "Exit wenn Spread zur Mitte
    zurueckkehrt"), max_holding_candles erreicht, oder exit_deadline (Ende
    der verfuegbaren Preisreihe) erreicht.

    Args:
        price_df: DataFrame (timestamp x asset_id) mit Preisen, muss
            entry_time bis mindestens exit_deadline abdecken.
        entry_time: Zeitpunkt des Entry-Signals.
        exit_deadline: spaetester moeglicher Exit-Zeitpunkt (z.B. Ende des
            verfuegbaren Preisfensters).
        pair: (asset_id_i, asset_id_j), wie in build_pairwise_return_spreads.
        direction: "long_i_short_j" (sign war -1: r_i - r_j unterdurchschnittlich
            negativ -> i erwartet zu steigen relativ, j zu fallen) oder
            "long_j_short_i" (sign war +1, siehe compute_max_adjusted_spread).
        signal_series: dict mit "xt" (pd.Series, normierter Spread Xt fuer
            GENAU dieses Paar, gleicher Index wie price_df) und "lower_limit"
            (pd.Series, zuletzt geschaetzte untere Betragsgrenze) fuer den
            Rueckkehr-Exit.
        max_holding_candles, fee_bps: siehe backtest_engine.simulate_trades.

    Returns:
        PairTrade, oder None falls entry_time nicht im Preisindex liegt oder
        keine gueltigen Preise fuer beide Assets zum Entry vorliegen.
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
    lower_limit = signal_series["lower_limit"]

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
        cur_lower = lower_limit.iloc[i]
        if pd.notna(cur_xt) and pd.notna(cur_lower) and abs(cur_xt) <= cur_lower:
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
    max_holding_candles: int,
    fee_bps: float,
    trade_mode: str = "max_only",
    recalib_every: int = 1,
    warmup_candles: int | None = None,
) -> pd.DataFrame:
    """Orchestriert den vollstaendigen Pairtrading-Backtest ueber alle
    Cliquen/Baskets im gewaehlten Zeitraum (analog backtest_engine.run_backtest).

    trade_mode (methodische Erweiterung ggue. dem Originalpaper):
        - "max_only": repliziert Lin et al. exakt -- pro Zeitpunkt wird NUR
          das Paar mit dem groessten |Xt| (Xmax) als Handelskandidat betrachtet.
        - "all_significant": erlaubt zusaetzlich das GLEICHZEITIGE Handeln
          ALLER Paare, deren |Xt| ihre jeweilige (auf Xmax-Historie
          geschaetzte) Toleranzgrenze verletzt, nicht nur des Extremwerts.
          Methodisch vertretbar, WEIL die hier verwendeten Cliquen bereits
          TDA-gefiltert und dadurch kleiner/reiner als Lin et al.'s volles
          Asset-Universum sind (mehr Kandidaten pro Clique realistisch
          handelbar, ohne die Signalqualitaet durch Streuung ueber ein
          riesiges Universum zu verwaessern) -- siehe Thesis-Diskussion.

    Args:
        cliques_df: wie aus dem Cache geladen (discovery_date, basket,
            asset_ids, ...), bereits auf gewuenschte Baskets/Variante gefiltert.
        ohlcv_df: 5-Min-Kerzen-OHLCV mit mindestens timestamp/asset_id/close.
        decay, lookback_candles, p_content, confidence, recalib_every: siehe
            compute_ewma_volatility / generate_tolerance_signals.
        max_holding_candles, fee_bps: siehe simulate_pair_trade.
        warmup_candles: Vorlauf vor jedem Discovery-Tag fuer EWMA-Warmup UND
            das Toleranzgrenzen-Lookback-Fenster. Default:
            lookback_candles + 30 (EWMA-Init-Fenster, siehe
            compute_ewma_volatility) mit grosszuegigem Puffer.

    Returns:
        DataFrame, eine Zeile pro Trade (PairTrade-Felder als Spalten,
        inkl. net_return/holding_candles/entry_time/exit_time fuer
        Kompatibilitaet mit backtest_engine.compute_backtest_stats/
        compute_equity_curve), leer falls keine Trades zustande kamen.
    """
    if trade_mode not in ("max_only", "all_significant"):
        raise ValueError(f"trade_mode muss 'max_only' oder 'all_significant' sein, erhalten: {trade_mode}")

    warmup_candles = warmup_candles or (lookback_candles + 30)

    price_matrix = ohlcv_df.pivot_table(
        index=timestamp_column, columns=asset_id_column, values=close_column, aggfunc="last",
    ).sort_index()
    returns_matrix = price_matrix.pct_change(fill_method=None)
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
        window_returns = returns_matrix.iloc[window_start_pos:day_end_pos][clique_ids]
        if window_prices.empty:
            continue

        spread_df = build_pairwise_return_spreads(window_returns)
        adjusted_df = compute_adjusted_spreads(spread_df, decay=decay)
        max_spread_df = compute_max_adjusted_spread(adjusted_df)
        tolerance_df = generate_tolerance_signals(
            max_spread_df, lookback_candles=lookback_candles,
            p_content=p_content, confidence=confidence, recalib_every=recalib_every,
        )

        day_slice = window_prices.index[window_prices.index >= day_start]
        exit_deadline = day_slice[-1] if len(day_slice) else window_prices.index[-1]

        if trade_mode == "max_only":
            entry_rows = tolerance_df.loc[day_slice]
            entry_rows = entry_rows.loc[entry_rows["entry"]]
            for ts in entry_rows.index:
                row = entry_rows.loc[ts]
                pair = row["pair"]
                if pair is None:
                    continue
                direction = "long_j_short_i" if row["sign"] > 0 else "long_i_short_j"
                xt_pair = adjusted_df[pair]
                signal_series = {"xt": xt_pair, "lower_limit": tolerance_df["lower_limit"]}
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
                    signal_series = {"xt": xt_pair, "lower_limit": tolerance_df["lower_limit"]}
                    trade = simulate_pair_trade(
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
    max_holding_candles: int,
    fee_bps: float,
    trade_mode: str = "max_only",
    initial_capital: float = 10_000.0,
    max_concurrent_positions: int = 10,
) -> pd.DataFrame:
    """Wiederholt run_pairtrading_backtest ueber ein Gitter aus
    (decay, lookback_candles), sammelt Kennzahlen fuer plot_sensitivity_heatmap.

    Braucht backtest_engine.compute_equity_curve/compute_backtest_stats zur
    Kennzahlen-Berechnung -- wird per Lazy-Import geholt, um einen zirkulaeren
    Modul-Import auf oberster Ebene zu vermeiden (backtest_engine importiert
    dieses Modul nicht, aber so bleibt die Abhaengigkeitsrichtung explizit
    lokal und offensichtlich).

    Returns:
        Lange Tabelle, eine Zeile pro (decay, lookback_candles)-Kombination,
        mit allen Feldern aus backtest_engine.compute_backtest_stats plus
        den Gitter-Parametern selbst als Spalten.
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

    Nutzt exakt dieselbe Signal-Kette wie run_pairtrading_backtest (Schritte
    1-4 in Lin et al. 2021: build_pairwise_return_spreads ->
    compute_adjusted_spreads -> compute_max_adjusted_spread ->
    generate_tolerance_signals), OHNE simulate_pair_trade/Positions-
    verwaltung -- deshalb um Groessenordnungen schneller als ein voller
    Backtest (kein Trade-Loop, kein Equity-Tracking).

    Information Coefficient (IC): fuer jede Kerze t wird die Signalstaerke
        signal_t = (|Xmax_t| - upper_limit_t) / upper_limit_t
    (relative Ueberschreitung der Toleranzgrenze; <=0 falls kein Durchbruch)
    mit der nachfolgenden Rueckkehr des adjustierten Spreads verglichen:
        reversion_t = -sign_t * (Xt_{t+k} - Xt_t)
    (sign_t * Xt_t ist der "ausgeschlagene" Wert; reversion_t ist positiv,
    wenn sich der Spread in Richtung Mittelwert zurueckbewegt -- genau das
    Verhalten, auf das die Strategie wettet). Der IC ist die Pearson-
    Korrelation zwischen signal_t und reversion_t ueber alle Kerzen mit
    gueltigem Entry-Durchbruch (signal_t > 0), gepoolt ueber alle Cliquen
    im Zeitraum. Ein hoher IC bedeutet: staerkere Grenzverletzung geht mit
    staerkerer Rueckkehr einher -- ein Indiz (kein Beweis), dass eine
    Parameterkombination handelbare Signale erzeugt.

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
    returns_matrix = price_matrix.pct_change(fill_method=None)
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

        window_returns = returns_matrix.iloc[window_start_pos:day_end_pos][clique_ids]
        if window_returns.empty:
            continue

        spread_df = build_pairwise_return_spreads(window_returns)
        adjusted_df = compute_adjusted_spreads(spread_df, decay=ewma_decay)
        max_spread_df = compute_max_adjusted_spread(adjusted_df)
        day_start_in_window = window_returns.index[window_returns.index >= day_start]
        per_day_max_spread.append((max_spread_df, adjusted_df, day_start_in_window))

    rows = []
    for params in param_combinations:
        lookback_candles = int(params["lookback_candles"])
        p_content = float(params["p_content"])

        signals: list[float] = []
        reversions: list[float] = []

        for max_spread_df, adjusted_df, day_slice in per_day_max_spread:
            tolerance_df = generate_tolerance_signals(
                max_spread_df, lookback_candles=lookback_candles,
                p_content=p_content, confidence=confidence, recalib_every=1,
            )
            day_tolerance = tolerance_df.loc[day_slice]
            entry_rows = day_tolerance.loc[day_tolerance["entry"]]

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

    Zerlegt [start, end) in aufeinanderfolgende (train, val)-Bloecke:
    Trainingsfenster der Laenge train_days, direkt gefolgt von einem
    Validierungsfenster der Laenge val_days, danach ruecken beide Fenster
    um step_days weiter. Ein optionaler Embargo-Puffer wird ZWISCHEN
    train_end und val_start eingefuegt (Tage, die weder trainiert noch
    validiert werden), um Leakage durch ueberlappende Rekalibrierungs-
    fenster/Lookback-Historie zu vermeiden (siehe Aufgabenstellung).

    Der von generate_walkforward_windows abgedeckte Bereich [start, end)
    ist ausschliesslich fuer Tuning (Optuna-Trials) vorgesehen -- der
    finale Testzeitraum liegt AUSSERHALB von [start, end) und wird von
    dieser Funktion nie beruehrt (siehe assert_no_test_period_leakage).

    Args:
        start, end: Gesamter Trainings-/Validierungszeitraum (nicht der
            finale Testzeitraum!), als alles was pd.Timestamp akzeptiert.
        train_days: Laenge des Trainingsfensters in Tagen.
        val_days: Laenge des Validierungsfensters in Tagen.
        step_days: Schrittweite, um die beide Fenster je Iteration
            vorruecken (step_days < train_days + embargo_days + val_days
            erzeugt ueberlappende Trainingsfenster zwischen Iterationen --
            zulaessig, aber bewusst zu waehlen).
        embargo_days: Puffer zwischen train_end und val_start (Tage), der
            weder trainiert noch validiert wird.

    Returns:
        Liste von dicts mit Timestamp-Keys train_start, train_end,
        val_start, val_end (train_end/val_end EXKLUSIV, wie ueblich bei
        Halboffenen Intervallen in diesem Projekt). Leer, falls selbst ein
        einziges vollstaendiges Fenster nicht in [start, end) passt.
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
