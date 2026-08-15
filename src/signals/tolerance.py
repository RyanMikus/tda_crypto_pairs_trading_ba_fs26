"""EWMA-Toleranzintervall-Signalkette fuer Pairtrading (Lin, Chen & Syu 2021).

Ablauf (Steps 1-4):
    1. Return-Spread pro Paar in einer Clique (build_pairwise_return_spreads).
    2. EWMA-Vol normieren: Xt = Rt / sigma_t (compute_ewma_volatility,
       compute_adjusted_spreads).
    3. Xmax_t = max(|Xt|) je Zeitpunkt (compute_max_adjusted_spread).
    4. Wilks-Toleranzintervall + 1-Schritt-Vergleich (wilks_tolerance_limits,
       generate_tolerance_signals).

Trade-Simulation/Orchestrierung (Steps 5-6) lebt in
notebooks/03_Tradeing/pairtrading_engine.py, das dieses Modul importiert.
"""

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from Src.signals.spread import compute_log_returns, compute_return_spread


# ============================================================================
# 1. EWMA-Volatilitaetsschaetzung
# ============================================================================

def compute_ewma_volatility(returns: pd.Series, *, decay: float) -> pd.Series:
    """RiskMetrics-EWMA nach Lin et al. (2021), Gl. 4A:
        sigma_t^2 = (1 - decay) * a_{t-1}^2 + decay * sigma_{t-1}^2
    a_t = returns - mean(returns), demeaned ueber die gesamte Serie.
    sigma_0^2 = Mittelwert von a_t^2 ueber die ersten min(30, len(returns))
    Beobachtungen (Anlauffenster, Paper nennt keine exakte Groesse).

    decay=0.94 ist RiskMetrics-Standard fuer TAEGLICHE Daten; bei 5-Min-Kerzen
    (288/Tag) ist das eine deutlich kuerzere effektive Halbwertszeit. Deshalb
    kein Default -- Wert empirisch pruefen.

    Args:
        returns: 1D-Return-Serie, darf NaN am Anfang enthalten.
        decay: EWMA-Decay-Faktor in (0, 1).

    Returns:
        Series gleicher Laenge/Index wie returns (sigma_t, nicht sigma_t^2).
        Bedeutet: NaN, falls zu wenig Historie (weniger als 2 gueltige Werte) oder
        falls NaN in returns[t] oder returns[t-1] (keine Schaetzung moeglich).
        Die ergebnisse sagen aus "wie stark die Returns in der Vergangenheit geschwankt haben",
        nicht "wie stark die Returns in der Zukunft schwanken werden" (keine Prognose, nur Historie).
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

def build_pairwise_return_spreads(price_df: pd.DataFrame) -> pd.DataFrame:
    """Rt^(Ai,Aj) = r_Ai,t - r_Aj,t fuer ALLE Paare innerhalb einer Clique.

    Nutzt src.signals.spread.compute_log_returns/compute_return_spread
    (echte Log-Returns, nicht pct_change).

    Args:
        price_df: DataFrame (timestamp x asset_id) mit Preisen, bereits auf
            die Mitglieder EINER Clique gefiltert.

    Returns:
        DataFrame (timestamp x pair), pair als Tuple (asset_id_i, asset_id_j)
        mit i < j (Vorzeichen wird erst in compute_max_adjusted_spread
        interpretiert). Eine Spalte pro Paar.
    """
    returns_df = compute_log_returns(price_df)
    asset_ids = list(returns_df.columns)
    spreads = {
        (a_i, a_j): compute_return_spread(returns_df, a_i, a_j)
        for idx, a_i in enumerate(asset_ids)
        for a_j in asset_ids[idx + 1:]
    }
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

    Bedeutet einfach erklaert: "Welches Paar ist gerade am extremsten im Vergleich zu seiner eigenen
    Historie (EWMA-Vol)?" und "In welche Richtung ist es extrem?".

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

def _wilks_one_sided_index(n: int, *, p_content: float, confidence: float, side: str) -> int:
    """Groesster gueltiger Rang l (unterer Fall) bzw. kleinster gueltiger Rang u
    (oberer Fall) fuer ein einseitiges nichtparametrisches Wilks-Toleranz-
    limit, wie in Lin et al. (2021) Gl. (2) (unten) und Gl. (3) (oben).

    Ordnungsstatistik-Standardresultat: 1-F(X_(l)) ~ Beta(n-l+1, l) und
    F(X_(u)) ~ Beta(u, n-u+1). Damit:
        untere Grenze: P(1-F(X_(l)) >= p) = 1 - Beta.cdf(p; n-l+1, l)
        obere Grenze:  P(F(X_(u)) >= p)   = 1 - Beta.cdf(p; u, n-u+1)
    Beide Wahrscheinlichkeiten steigen monoton mit l bzw. fallen monoton mit
    u -- gesucht wird deshalb das GROESSTE l und das KLEINSTE u, das die
    Konfidenz noch erreicht (engste gueltige Grenze).

    Returns:
        1-indizierter Rang. Faellt auf den Stichproben-Extremwert (l=1 bzw.
        u=n) zurueck, falls keine engere Wahl die Konfidenz erreicht.
    """
    if side == "lower":
        best = 1
        for l in range(1, n + 1):
            if 1.0 - beta_dist.cdf(p_content, n - l + 1, l) >= confidence:
                best = l
        return best
    for u in range(1, n + 1):
        if 1.0 - beta_dist.cdf(p_content, u, n - u + 1) >= confidence:
            return u
    return n


def wilks_tolerance_limits(
    sample: np.ndarray, *, p_content: float, confidence: float,
) -> tuple[float, float]:
    """Nichtparametrische einseitige Wilks-Toleranzgrenzen (X_(l), X_(u)),
    wie in Lin et al. (2021) Gl. (2)/(3) definiert und S. 3 als "lower and
    upper trading signals" verwendet.

    Es handelt sich um ZWEI unabhaengige einseitige Toleranzlimits (nicht ein
    gemeinsames zweiseitiges Intervall nach Gl. 1): l und u werden getrennt
    ueber _wilks_one_sided_index bestimmt, jeweils mit demselben (p_content,
    confidence)-Paar wie im Paper (S. 4: (p, 1-alpha) = (90%, 0.95) bzw.
    (95%, 0.95) fuer beide Seiten gleichermassen).

    Args:
        sample: 1D-Array historischer Beobachtungen, NaN wird entfernt.
        p_content: geforderter abgedeckter Anteil (z.B. 0.95).
        confidence: Konfidenzniveau 1-alpha (z.B. 0.95).

    Returns:
        (lower_limit, upper_limit) = (X_(l), X_(u)), Werte aus `sample`.

    Raises:
        ValueError: falls nach Entfernen von NaN weniger als 3 Beobachtungen
            uebrig bleiben (kein sinnvolles Toleranzintervall moeglich).
    """
    clean = np.sort(np.asarray(sample, dtype=float))
    clean = clean[~np.isnan(clean)]
    n = len(clean)
    if n < 3:
        raise ValueError(f"wilks_tolerance_limits braucht mindestens 3 gueltige Beobachtungen, erhalten: {n}")

    l = _wilks_one_sided_index(n, p_content=p_content, confidence=confidence, side="lower")
    u = _wilks_one_sided_index(n, p_content=p_content, confidence=confidence, side="upper")

    lower_limit = float(clean[l - 1])
    upper_limit = float(clean[u - 1])
    return lower_limit, upper_limit


# ============================================================================
# 4. Rollierende Signal-Generierung (Paper Step 6)
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

    Alle recalib_every Kerzen werden die Toleranzgrenzen aus dem
    zurueckliegenden Fenster xmax[t-lookback_candles:t] geschaetzt (kausal,
    kein Blick auf xmax[t] selbst) und gelten bis zur naechsten
    Rekalibrierung. recalib_every=1 entspricht Paper Step 6 exakt, hoehere
    Werte sind eine Performance-Abkuerzung.

    Args:
        max_spread_df: siehe compute_max_adjusted_spread (xmax, pair, sign).
        lookback_candles: Fenstergroesse in Kerzen (Kerzen- statt Tage-
            Kalibrierung, siehe compute_ewma_volatility).
        p_content, confidence: siehe wilks_tolerance_limits.
        recalib_every: Rekalibrierungs-Intervall in Kerzen.

    Returns:
        DataFrame mit lower_limit/upper_limit (zuletzt geschaetzt), entry
        (bool, xmax > upper_limit), pair/sign (durchgereicht).
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
