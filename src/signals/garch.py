"""GARCH(1,1)-Volatilitaetsschaetzung als Alternative zu EWMA (siehe
src.signals.tolerance.compute_ewma_volatility) fuer die Xt-Normierung der
Renditedifferenz (Lin, Chen & Syu 2021, Gl. 4B).

Eigene Implementierung statt des `arch`-Pakets: dessen interner
Maximum-Likelihood-Optimierer nutzt gradientenbasierte scipy.optimize-Methoden
(BFGS/L-BFGS-B/SLSQP), die auf diesem Windows-Setup mit einem nativen
BLAS/DLL-Konflikt abstuerzen (reproduzierbar bereits bei einem nackten
scipy.optimize.minimize(..., method="BFGS")-Aufruf, unabhaengig vom
arch-Paket). method="Nelder-Mead" (ableitungsfrei) ist nachweislich stabil,
deshalb wird ausschliesslich damit optimiert.

Modell (Bollerslev 1986, wie im Aufgabenstellung-Zusatz-PDF als
Crypto-taugliche Alternative zu EWMA genannt):
    a_t = returns_t - mean(returns)
    sigma_t^2 = omega + alpha * a_{t-1}^2 + beta * sigma_{t-1}^2
Stationaritaetsbedingung: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1.

Parametrisierung folgt PDF-Empfehlung: Student-t-Innovationen (crypto fat
tails), aber die Volatilitaetsschaetzung selbst (sigma_t) ist unabhaengig von
der Innovationsverteilung -- die Verteilungsannahme wirkt sich nur auf die
Log-Likelihood aus, mit der omega/alpha/beta/nu gefittet werden.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


#!#################### NEU: oeffentliches Pendant zu _garch_sigma2, fuer
#!#################### Aufrufer (GarchWilksToleranceStrategy), die den
#!#################### Sigma-Pfad aus BEREITS gefitteten Parametern (z.B. aus
#!#################### compute_garch_volatility()[1] bei Formation) brauchen,
#!#################### ohne bei jeder Bar neu zu fitten.
def garch_sigma_path(returns: pd.Series, *, omega: float, alpha: float, beta: float) -> pd.Series:
    """sigma_t-Pfad (nicht sigma_t^2) fuer gegebene, bereits gefittete
    GARCH(1,1)-Parameter -- siehe _garch_sigma2 fuer die Rekursion.

    Args:
        returns: 1D-Return-/Spread-Serie, darf NaN enthalten (siehe
            _garch_sigma2 -- NaN-Innovationen werden als 0 behandelt).
        omega, alpha, beta: gefittete GARCH(1,1)-Parameter, z.B. aus
            compute_garch_volatility(...)[1].

    Returns:
        Series gleicher Laenge/Index wie returns (sigma_t).
    """
    sigma2 = _garch_sigma2(returns.to_numpy(dtype=float), omega, alpha, beta)
    return pd.Series(np.sqrt(sigma2), index=returns.index)
#!####################


def _garch_sigma2(returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """sigma_t^2-Pfad fuer gegebene GARCH(1,1)-Parameter, kausal (kein
    Blick auf zukuenftige a_t). sigma_0^2 = unbedingte Varianz (long-run
    variance) als Startwert, Standardkonvention."""
    n = len(returns)
    mean_val = np.nanmean(returns)
    a = returns - mean_val

    sigma2 = np.empty(n)
    denom = 1.0 - alpha - beta
    sigma2_0 = omega / denom if denom > 1e-6 else float(np.nanvar(returns))
    sigma2[0] = sigma2_0

    for t in range(1, n):
        prev_a = a[t - 1] if not np.isnan(a[t - 1]) else 0.0
        prev_sigma2 = sigma2[t - 1]
        sigma2[t] = omega + alpha * prev_a ** 2 + beta * prev_sigma2

    return sigma2


def _neg_log_likelihood_student_t(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative Student-t-Log-Likelihood fuer GARCH(1,1) -- Student-t statt
    Normalverteilung, siehe Modul-Docstring (crypto fat tails, PDF-Empfehlung).
    params = [omega, alpha, beta, nu] (nu = Freiheitsgrade, nu > 2 fuer
    endliche Varianz).
    """
    omega, alpha, beta, nu = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0 or nu <= 2.0:
        return 1e10  # harte Straf-Grenze statt constrained optimizer (Nelder-Mead kennt keine Constraints)

    mean_val = np.nanmean(returns)
    a = returns - mean_val
    sigma2 = _garch_sigma2(returns, omega, alpha, beta)
    sigma2 = np.maximum(sigma2, 1e-12)

    # Student-t-Dichte, standardisiert auf Varianz sigma2 (nicht Skalenparameter nu/(nu-2)):
    #   f(a) = Gamma((nu+1)/2) / (Gamma(nu/2) * sqrt(pi*(nu-2)*sigma2))
    #          * (1 + a^2 / ((nu-2)*sigma2))^(-(nu+1)/2)
    log_const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    log_density = (
        log_const - 0.5 * np.log(sigma2)
        - (nu + 1) / 2 * np.log1p(a ** 2 / ((nu - 2) * sigma2))
    )
    valid = ~np.isnan(log_density)
    if valid.sum() == 0:
        return 1e10
    return -float(np.sum(log_density[valid]))


def compute_garch_volatility(
    returns: pd.Series, *, dist: str = "t", max_iter: int = 500,
) -> tuple[pd.Series, dict]:
    """GARCH(1,1)-Volatilitaet fuer eine Rendite-/Spread-Serie, per Maximum-
    Likelihood gefittet (Nelder-Mead, siehe Modul-Docstring).

    Pendant zu tolerance.compute_ewma_volatility, aber mit gefitteten
    (statt fest vorgegebenem decay) Parametern omega/alpha/beta.

    Args:
        returns: 1D-Return-/Spread-Serie, darf NaN enthalten (werden beim
            Fit ignoriert, im sigma_t-Pfad als 0-Innovation behandelt --
            siehe _garch_sigma2).
        dist: nur "t" (Student-t) implementiert -- siehe Modul-Docstring.
        max_iter: max. Nelder-Mead-Iterationen.

    Returns:
        (sigma_series, fit_info): sigma_series wie compute_ewma_volatility
        (sigma_t, nicht sigma_t^2), gleicher Index wie returns. fit_info ist
        ein dict mit den gefitteten Parametern (omega, alpha, beta, nu) und
        "converged" (bool) -- fuer Diagnose/Logging.

    Raises:
        ValueError: falls dist != "t" oder weniger als 30 gueltige
            Beobachtungen vorliegen (GARCH-Fit braucht ausreichend Historie).
    """
    if dist != "t":
        raise ValueError(f"compute_garch_volatility unterstuetzt nur dist='t', erhalten: {dist!r}")

    values = returns.to_numpy(dtype=float)
    n = len(values)
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 30:
        return pd.Series(np.full(n, np.nan), index=returns.index), {
            "omega": np.nan, "alpha": np.nan, "beta": np.nan, "nu": np.nan, "converged": False,
        }

    clean = values[valid_mask]
    sample_var = float(np.var(clean))
    # Startwerte: alpha/beta nahe typischer Finanzmarkt-Kalibrierung
    # (persistente, aber stationaere Vola), omega so gewaehlt, dass die
    # unbedingte Varianz omega/(1-alpha-beta) der Stichprobenvarianz entspricht.
    init_alpha, init_beta = 0.08, 0.90
    init_omega = sample_var * (1.0 - init_alpha - init_beta)
    init_nu = 6.0
    x0 = np.array([init_omega, init_alpha, init_beta, init_nu])

    result = minimize(
        _neg_log_likelihood_student_t, x0, args=(clean,),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-8, "fatol": 1e-8},
    )
    omega, alpha, beta, nu = result.x
    converged = bool(result.success) and omega > 0 and alpha >= 0 and beta >= 0 and (alpha + beta) < 1.0 and nu > 2.0

    if not converged:
        # Fallback: Startwerte selbst verwenden statt eines nicht-stationaeren
        # oder divergierten Fits -- besser eine grobe, aber gueltige
        # Schaetzung als NaN/instabile Werte im gesamten Signalpfad.
        omega, alpha, beta, nu = init_omega, init_alpha, init_beta, init_nu

    sigma2_full = _garch_sigma2(values, omega, alpha, beta)
    # Punkte vor der ersten gueltigen Beobachtung sind nicht sinnvoll bestimmt.
    first_valid = int(np.flatnonzero(valid_mask)[0])
    sigma2_full[:first_valid] = np.nan

    fit_info = {"omega": float(omega), "alpha": float(alpha), "beta": float(beta), "nu": float(nu), "converged": converged}
    return pd.Series(np.sqrt(sigma2_full), index=returns.index), fit_info
