"""Gemeinsame Fixtures für die Backtest-Engine-Tests.

Alles hier ist SYNTHETISCH (kein Zugriff auf data/raw/... oder
data/processed/cliques/...), damit die Testsuite schnell, deterministisch und
unabhängig vom tatsächlichen (grossen) Datenbestand läuft. AAAUSDT/BBBUSDT
sind absichtlich stark kointegriert gebaut, damit PairwiseZScoreStrategy/
ClusterSpreadStrategy in den End-to-End-Tests zuverlässig Trades erzeugen;
CCCUSDT/DDDUSDT sind unabhängige Random Walks als "Rauschen".
"""

import json

import numpy as np
import pandas as pd
import pytest

from Src.backtest.cliques import CliqueTimeline
from Src.backtest.config import BasketParams
from Src.backtest.types import BarContext, Leg, Position, PriceWindow, SpreadDefinition

SYMBOLS = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]
N_CANDLES = 1600  # ~5.5 Tage bei 5-Min-Kerzen


@pytest.fixture
def symbols():
    return list(SYMBOLS)


@pytest.fixture
def id_to_symbol():
    return {i + 1: sym for i, sym in enumerate(SYMBOLS)}


@pytest.fixture
def symbol_to_id(id_to_symbol):
    return {v: k for k, v in id_to_symbol.items()}


@pytest.fixture
def price_index():
    return pd.date_range("2025-01-01", periods=N_CANDLES, freq="5min", tz="UTC")


@pytest.fixture
def synthetic_panel(price_index):
    """AAAUSDT/BBBUSDT teilen einen gemeinsamen Random-Walk-Faktor (stark
    kointegriert, hedge_ratio ~0.5), CCCUSDT/DDDUSDT sind unabhängig."""
    rng = np.random.default_rng(0)
    n = len(price_index)

    common = rng.normal(0, 1.0, n).cumsum()
    aaa = 100 + common + rng.normal(0, 0.15, n)
    bbb = 50 + 0.5 * common + rng.normal(0, 0.15, n)
    ccc = 200 + rng.normal(0, 1.0, n).cumsum()
    ddd = 80 + rng.normal(0, 1.0, n).cumsum()

    close = pd.DataFrame(
        {"AAAUSDT": aaa, "BBBUSDT": bbb, "CCCUSDT": ccc, "DDDUSDT": ddd}, index=price_index,
    )
    close = close.clip(lower=1.0)  # keine negativen/Null-Preise (log-Spreads brauchen das)

    return {
        "open": close.copy(),
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
    }


def write_clique_cache(tmp_path, *, price_index, symbol_to_id, symbols, basket="Basket 1", lookback_days=1):
    """Schreibt einen synthetischen Clique-Cache (ein Snapshot pro Kalendertag,
    immer dieselben Assets) im selben Format wie
    HelperFunctions.discover_cliques_over_dates."""
    rows = []
    for d in pd.date_range(price_index[0].normalize(), price_index[-1].normalize(), freq="D", tz="UTC"):
        rows.append({
            "discovery_date": d,
            "basket": basket,
            "asset_ids": [symbol_to_id[s] for s in symbols],
            "persistence": 0.5,
            "death": 1.0,
            "size": len(symbols),
            "lookback_days": lookback_days,
        })
    df = pd.DataFrame(rows)

    cliques_dir = tmp_path / "cliques"
    cliques_dir.mkdir(exist_ok=True)
    df.to_parquet(cliques_dir / "cliques_raw.parquet", index=False)
    (cliques_dir / "cliques_raw.meta.json").write_text(json.dumps({"variant": "raw"}))
    return cliques_dir


@pytest.fixture
def cliques_cache_dir(tmp_path, price_index, symbol_to_id, symbols):
    return write_clique_cache(tmp_path, price_index=price_index, symbol_to_id=symbol_to_id, symbols=symbols)


@pytest.fixture
def timeline(cliques_cache_dir, id_to_symbol):
    return CliqueTimeline.from_cache(
        variant="raw", id_to_symbol=id_to_symbol, cliques_dir=cliques_cache_dir, baskets=["Basket 1"],
    )


@pytest.fixture
def make_spread_definition():
    """Baut eine SpreadDefinition mit frei wählbaren mu/sigma/weights --
    fürs isolierte Testen von Exit-Regeln/Strategie-Logik ohne echte OLS/
    Johansen-Schätzung."""
    def _make(mu=0.0, sigma=1.0, weights=(1.0, -0.5), symbols=("AAAUSDT", "BBBUSDT"), asset_ids=(1, 2), kind="pairwise"):
        return SpreadDefinition(
            kind=kind, asset_ids=asset_ids, symbols=symbols, weights=weights,
            mu=mu, sigma=sigma,
            fitted_at=pd.Timestamp("2025-01-01", tz="UTC"),
            formation_start=pd.Timestamp("2024-12-25", tz="UTC"),
        )
    return _make


@pytest.fixture
def make_position(make_spread_definition):
    def _make(
        *, params=None, bars_held=0, missing_price_bars=0, membership_key="mk1", basket="Basket 1", spread=None,
        direction=1, entry_death=1.0, entry_lookback_days=1,
    ):
        spread = spread or make_spread_definition()
        legs = [
            Leg(asset_id=a, symbol=s, weight=w)
            for a, s, w in zip(spread.asset_ids, spread.symbols, spread.weights)
        ]
        return Position(
            position_id="pos-1", strategy="pairwise_zscore", basket=basket, spread=spread,
            params=params or BasketParams(),
            membership_key=membership_key, discovery_timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
            entry_death=entry_death, entry_lookback_days=entry_lookback_days,
            direction=direction, legs=legs,
            entry_timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
            entry_signal_value=2.0, entry_spread_value=0.0,
            gross_notional=1000.0, entry_equity=100_000.0,
            bars_held=bars_held, missing_price_bars=missing_price_bars,
        )
    return _make


@pytest.fixture
def make_price_window():
    """Baut ein PriceWindow mit genau EINER Bar aus einem {symbol: price}-dict
    -- reicht fuer alle Stellen, die nur `.last()`/`.log_last()` auf der
    aktuellen Kerze brauchen (Exit-Regeln, generate_entries)."""
    def _make(prices: dict, bar_index=0):
        symbols = list(prices.keys())
        arr = np.array([[prices[s] for s in symbols]], dtype=float)
        idx = pd.date_range("2025-01-01", periods=1, freq="5min", tz="UTC")
        return PriceWindow({"close": arr}, {s: i for i, s in enumerate(symbols)}, idx, bar_index)
    return _make


@pytest.fixture
def make_bar_context():
    def _make(*, params=None, snapshot=None, prices=None, open_positions=(), equity=100_000.0, free_slots=1, basket="Basket 1", timestamp=None, bar_index=0):
        return BarContext(
            timestamp=timestamp or pd.Timestamp("2025-01-02", tz="UTC"), bar_index=bar_index, basket=basket,
            params=params or BasketParams(), snapshot=snapshot, prices=prices,
            open_positions=tuple(open_positions), equity=equity, free_slots=free_slots,
        )
    return _make


@pytest.fixture
def make_multi_bar_price_window():
    """Baut ein PriceWindow mit MEHREREN Kerzen aus {symbol: [preis, preis, ...]}
    -- fuer Regeln, die wie TopologicalBreakRule eine Preis-Historie
    (.history()) statt nur der aktuellen Kerze (.last()) brauchen."""
    def _make(prices: dict, bar_index=None):
        symbols = list(prices.keys())
        n_bars = len(next(iter(prices.values())))
        arr = np.array([[prices[s][i] for s in symbols] for i in range(n_bars)], dtype=float)
        idx = pd.date_range("2025-01-01", periods=n_bars, freq="5min", tz="UTC")
        return PriceWindow({"close": arr}, {s: i for i, s in enumerate(symbols)}, idx, n_bars - 1 if bar_index is None else bar_index)
    return _make


@pytest.fixture
def register_strategy():
    """Registriert eine Test-Strategie in STRATEGY_REGISTRY und entfernt sie
    nach dem Test wieder -- verhindert Verschmutzung des globalen Registry-
    Dicts zwischen Tests."""
    from Src.backtest.strategy import STRATEGY_REGISTRY

    registered = []

    def _register(name, cls):
        STRATEGY_REGISTRY[name] = cls
        registered.append(name)

    yield _register

    for name in registered:
        STRATEGY_REGISTRY.pop(name, None)
