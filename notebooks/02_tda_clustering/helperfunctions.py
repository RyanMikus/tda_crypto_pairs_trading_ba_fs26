
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
from pathlib import Path

import sys
sys.path.insert(0, str(Path("..").resolve()))
from config import (
DATA_DIR, OHLCV_DIR, ASSET_DAILY_PATH, ASSETS_PATH,DATE_COL, TIMESTAMP_COL, CLOSE_COL, ASSET_ID_COL
)


WINDOW_SIZE_DAYS = 30
MIN_ASSET_COVERAGE = 0.95
MIN_PERSISTENCE = 1.0
MAX_EDGE_LENGTH = 2.0
MIN_CLUSTER_SIZE = 2

# Stablecoins und Edelmetall-Token die aus der TDA-Analyse ausgeschlossen werden.
# Diese haben nahezu Null-Returns und würden die Korrelationsstruktur verzerren.
STABLECOIN_SYMBOLS: set[str] = {
    "USDT", "USDC", "BUSD", "DAI", "USDE", "USD1", "TUSD",
    "PAXG", "XAUT", "FDUSD", "USDP", "PYUSD",
}



# Helper Functions
#ignore type warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

class HelperFunctions:

    def __init__(self):
        pass
    
    @staticmethod
    def correlation_distance(returns_window: pd.DataFrame) -> np.ndarray:
        """Berechnet die paarweise Korrelationsdistanz zwischen Assets.

        Verwendet die Formel d(i,j) = sqrt(2 * (1 - rho(i,j))), wobei rho die
        Pearson-Korrelation der Log-Returns ist. Der resultierende Wert liegt
        immer im Bereich [0, 2]: 0 bedeutet perfekte positive Korrelation,
        sqrt(2) bedeutet keine Korrelation, 2 bedeutet perfekte negative
        Korrelation. Diese Metrik wird als Eingabe für VietorisRipsPersistence
        verwendet (metric="precomputed").

        Args:
            returns_window: DataFrame der Form (n_timestamps, n_assets) mit
                periodischen Returns pro Asset als Spalten.

        Returns:
            Symmetrische Distanzmatrix der Form (n_assets, n_assets) als
            numpy Array. Die Diagonale ist 0.
        """
        corr = np.corrcoef(returns_window.to_numpy().T)
        print(f"Korrelationsmatrix: {corr.shape[0]} Assets, {corr.shape[1]} Assets")
        print(f"Korrelationsmatrix: min={np.nanmin(corr):.4f}, max={np.nanmax(corr):.4f}, mean={np.nanmean(corr):.4f}")
        print(f"gesamte corr: {corr}")
        corr = np.nan_to_num(corr, nan=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        dist = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr)))
        np.fill_diagonal(dist, 0.0)
        return dist


    @staticmethod
    def get_universe_asset_ids_for_date(universe_df: pd.DataFrame, end_date: pd.Timestamp) -> list[int]:
        """Gibt alle Asset-IDs zurück, die am angegebenen Datum im Universum sind.

        Filtert den Universe-DataFrame auf den exakten end_date-Wert und gibt
        die eindeutigen, aufsteigend sortierten Asset-IDs zurück. Diese Liste
        dient als Grundlage dafür, welche Assets in ein Sliding-Window-Fenster
        aufgenommen werden.

        Args:
            universe_df: DataFrame mit mindestens den Spalten DATE_COL und
                ASSET_ID_COL. Jede Zeile repräsentiert ein Asset, das an einem
                bestimmten Datum zur qualifizierten Menge gehört.
            end_date: Das Datum, für das das Universum abgefragt wird. Muss
                exakt mit einem Wert in DATE_COL übereinstimmen (Timestamp).

        Returns:
            Sortierte Liste eindeutiger Asset-IDs (int) für end_date.
            Leere Liste, wenn kein Asset für dieses Datum vorhanden ist.
        """
        asset_ids = (
            universe_df.loc[universe_df[DATE_COL] == end_date, ASSET_ID_COL]
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )
        return asset_ids

    @staticmethod
    def build_window_input(
        ohlcv_df: pd.DataFrame,
        universe_df: pd.DataFrame,
        end_date: pd.Timestamp,
        window_size_days: int = WINDOW_SIZE_DAYS,
        min_asset_coverage: float = MIN_ASSET_COVERAGE,
        exclude_asset_ids: set[int] | None = None,
    ) -> dict | None:
        """Baut den vollständigen Eingabe-Dictionary für ein Sliding-Window auf.

        Führt folgende Schritte in Reihenfolge aus:
        1. Bestimmt das Zeitfenster [end_date - window_size_days + 1, end_date].
        2. Holt das qualifizierte Universum für end_date via
           get_universe_asset_ids_for_date.
        3. Filtert ohlcv_df auf dieses Fenster und diese Assets.
        4. Pivotiert die Preise zu einer Matrix (Timestamps × Assets).
        5. Entfernt Assets/Timestamps mit zu vielen NaN-Werten gemäss
           min_asset_coverage.
        6. Berechnet prozentuale Returns via pct_change().
        7. Berechnet die Korrelationsdistanz-Matrix via correlation_distance().

        Args:
            ohlcv_df: OHLCV-DataFrame mit mindestens den Spalten DATE_COL,
                TIMESTAMP_COL, ASSET_ID_COL und CLOSE_COL.
            universe_df: Universe-DataFrame mit DATE_COL und ASSET_ID_COL
                (Ausgabe von Basket-Berechnung).
            end_date: Enddatum des Fensters. Wird auf UTC-Mitternacht normiert.
            window_size_days: Anzahl Kalendertage im Fenster (Standard: 30).
            min_asset_coverage: Mindestanteil nicht-NaN Timestamps, die ein
                Asset haben muss, um im Fenster behalten zu werden (Standard:
                0.95 = 95 %).

        Returns:
            Dictionary mit folgenden Schlüsseln, oder None wenn das Fenster
            zu wenige Daten enthält (< 2 Assets, < 3 Timestamps):
                - start_date (Timestamp): Startdatum des Fensters
                - end_date (Timestamp): Enddatum des Fensters
                - asset_ids (list[int]): Asset-IDs nach Filterung
                - n_assets (int): Anzahl Assets
                - n_timestamps (int): Anzahl Return-Perioden
                - price_matrix (DataFrame): Preis-Matrix (Timestamps × Assets)
                - returns (DataFrame): Return-Matrix (Timestamps × Assets)
                - distance_matrix (ndarray): (n_assets × n_assets) Distanzmatrix
        """
        end_date = pd.to_datetime(end_date, utc=True).normalize()
        start_date = end_date - pd.Timedelta(days=window_size_days - 1)

        eligible_asset_ids = HelperFunctions.get_universe_asset_ids_for_date(universe_df, end_date)
        if exclude_asset_ids:
            eligible_asset_ids = [i for i in eligible_asset_ids if i not in exclude_asset_ids]
        if len(eligible_asset_ids) < 2:
            return None

        window_data = ohlcv_df.loc[
            (ohlcv_df[DATE_COL] >= start_date) &
            (ohlcv_df[DATE_COL] <= end_date) &
            (ohlcv_df[ASSET_ID_COL].isin(eligible_asset_ids)),
            [TIMESTAMP_COL, ASSET_ID_COL, CLOSE_COL]
        ].copy()

        if window_data.empty:
            return None

        price_matrix = (
            window_data
            .pivot_table(
                index=TIMESTAMP_COL,
                columns=ASSET_ID_COL,
                values=CLOSE_COL,
                aggfunc="last",
            )
            .sort_index()
        )

        if price_matrix.empty:
            return None

        min_non_null = int(np.ceil(len(price_matrix) * min_asset_coverage))
        price_matrix = price_matrix.dropna(axis=1, thresh=min_non_null)
        price_matrix = price_matrix.dropna(axis=0, how="any")

        if price_matrix.shape[0] < 3 or price_matrix.shape[1] < 2:
            return None

        returns = price_matrix.pct_change().dropna(how="any")

        if returns.shape[0] < 2 or returns.shape[1] < 2:
            return None

        dist = HelperFunctions.correlation_distance(returns)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "asset_ids": [int(x) for x in returns.columns.tolist()],
            "n_assets": returns.shape[1],
            "n_timestamps": returns.shape[0],
            "price_matrix": price_matrix,
            "returns": returns,
            "distance_matrix": dist,
        }

    @staticmethod
    def extract_persistent_h0_features(diagram: np.ndarray, min_persistence: float = MIN_PERSISTENCE) -> list[dict]:
        """Extrahiert persistente H0-Features aus einem gtda-Persistenzdiagramm.

        H0-Features entsprechen zusammenhängenden Komponenten (Connected
        Components) im Vietoris-Rips-Komplex. Eine hohe Persistenz
        (death - birth) bedeutet, dass sich zwei Gruppen erst bei einem grossen
        Distanz-Schwellenwert zusammenschliessen — also strukturell weit
        voneinander entfernt sind. Schwach persistente Features (Rauschen) werden
        via min_persistence herausgefiltert.

        Das gtda-Format eines Persistenzdiagramms ist ein numpy Array mit drei
        Spalten: [birth, death, dimension]. Diese Methode filtert auf
        dimension == 0 (H0).

        Args:
            diagram: numpy Array der Form (n_features, 3) mit Spalten
                [birth, death, dimension], wie von giotto-tda zurückgegeben.
            min_persistence: Minimale Persistenz (death - birth), die ein
                Feature haben muss, um behalten zu werden (Standard: 1.0).
                Wert 0 behält alle Features.

        Returns:
            Liste von Dictionaries, absteigend nach Persistenz sortiert.
            Jedes Dictionary enthält:
                - birth (float): Filterwert, bei dem die Komponente entsteht
                - death (float): Filterwert, bei dem die Komponente verschwindet
                - persistence (float): death - birth
            Leere Liste wenn diagram leer ist oder kein Feature die
            Schwelle überschreitet.
        """
        if diagram.size == 0:
            return []

        diagram_df = pd.DataFrame(diagram, columns=["birth", "death", "dimension"])

        h0 = diagram_df.loc[diagram_df["dimension"] == 0].copy()

        h0["persistence"] = h0["death"] - h0["birth"]

        h0 = h0.loc[h0["persistence"] > min_persistence].copy()
        h0 = h0.sort_values("persistence", ascending=False)

        return h0[["birth", "death", "persistence"]].to_dict("records")

    @staticmethod
    def build_point_cloud(
        ohlcv_df: pd.DataFrame,
        universe_df: pd.DataFrame,
        end_date: pd.Timestamp,
        window_size_days: int = WINDOW_SIZE_DAYS,
        min_asset_coverage: float = MIN_ASSET_COVERAGE,
        exclude_asset_ids: set[int] | None = None,
    ) -> dict | None:
        """Baut die n-dimensionale Punktewolke für die TDA-Eingabe auf.

        Jeder Zeitschritt (5min-Kerze) wird zu einem Punkt im n-dimensionalen
        Raum, wobei n = Anzahl Assets. Die Koordinate jedes Punktes ist der
        prozentuale Preischange aller Assets an diesem Zeitstempel:

            Punkt_t = (ret_asset1_t, ret_asset2_t, ..., ret_assetN_t)

        Physikalische Bedeutung der Nähe zweier Punkte:
        Zwei Zeitschritte liegen nah beieinander, wenn alle Assets in beiden
        Perioden ähnliche Returns hatten — d.h. dasselbe Marktregime herrschte.
        Dichte Regionen in der Wolke entsprechen wiederkehrenden Joint-Behavior-
        Mustern (z.B. mehrere BTC+ETH +5%-Tage landen auf demselben Punkt).

        Args:
            ohlcv_df: OHLCV-DataFrame mit mindestens den Spalten DATE_COL,
                TIMESTAMP_COL, ASSET_ID_COL und CLOSE_COL.
            universe_df: Universe-DataFrame mit DATE_COL und ASSET_ID_COL.
            end_date: Enddatum des Fensters. Wird auf UTC-Mitternacht normiert.
            window_size_days: Anzahl Kalendertage im Fenster (Standard: 30).
            min_asset_coverage: Mindestanteil nicht-NaN Timestamps pro Asset
                (Standard: 0.95 = 95 %).

        Returns:
            Dictionary mit folgenden Schlüsseln, oder None bei zu wenig Daten:
                - point_cloud (ndarray): Form (n_timestamps, n_assets).
                    Jede Zeile ist ein Punkt, jede Spalte eine Asset-Dimension.
                    Werte sind prozentuale Returns (pct_change).
                - asset_ids (list[int]): Asset-IDs in Spaltenreihenfolge.
                    asset_ids[i] entspricht Spalte i der point_cloud.
                - timestamps (Index): Zeitstempel der Punkte (Zeilenindex).
                - n_points (int): Anzahl Punkte = Anzahl Zeitschritte.
                - n_dimensions (int): Anzahl Dimensionen = Anzahl Assets.
                - start_date (Timestamp): Startdatum des Fensters.
                - end_date (Timestamp): Enddatum des Fensters.
        """
        end_date = pd.to_datetime(end_date, utc=True).normalize()
        start_date = end_date - pd.Timedelta(days=window_size_days - 1)

        eligible_asset_ids = HelperFunctions.get_universe_asset_ids_for_date(universe_df, end_date)
        if exclude_asset_ids:
            eligible_asset_ids = [i for i in eligible_asset_ids if i not in exclude_asset_ids]
        if len(eligible_asset_ids) < 2:
            return None

        window_data = ohlcv_df.loc[
            (ohlcv_df[DATE_COL] >= start_date) &
            (ohlcv_df[DATE_COL] <= end_date) &
            (ohlcv_df[ASSET_ID_COL].isin(eligible_asset_ids)),
            [TIMESTAMP_COL, ASSET_ID_COL, CLOSE_COL]
        ].copy()

        if window_data.empty:
            return None

        # Preismatrix: Zeilen = Timestamps, Spalten = Assets
        price_matrix = (
            window_data
            .pivot_table(
                index=TIMESTAMP_COL,
                columns=ASSET_ID_COL,
                values=CLOSE_COL,
                aggfunc="last",
            )
            .sort_index()
        )

        if price_matrix.empty:
            return None

        # Assets entfernen die zu viele NaN-Timestamps haben
        min_non_null = int(np.ceil(len(price_matrix) * min_asset_coverage))
        price_matrix = price_matrix.dropna(axis=1, thresh=min_non_null)
        # Timestamps entfernen bei denen irgend ein Asset fehlt
        # -> stellt sicher, dass jeder Punkt vollständig n-dimensional ist
        price_matrix = price_matrix.dropna(axis=0, how="any")

        if price_matrix.shape[0] < 3 or price_matrix.shape[1] < 2:
            return None

        # Returns: pct_change() -> erste Zeile wird NaN -> dropna
        # returns hat Form (n_timestamps-1, n_assets)
        # Jede Zeile ist ein Punkt in n-dimensionalem Return-Raum
        returns = price_matrix.pct_change().dropna(how="any")

        if returns.shape[0] < 2 or returns.shape[1] < 2:
            return None

        point_cloud = returns.to_numpy()  # (n_timestamps, n_assets)

        return {
            "point_cloud": point_cloud,
            "asset_ids": [int(x) for x in returns.columns.tolist()],
            "timestamps": returns.index,
            "n_points": point_cloud.shape[0],
            "n_dimensions": point_cloud.shape[1],
            "start_date": start_date,
            "end_date": end_date,
        }

    @staticmethod
    def get_clusters_at_scale(
        dist_matrix: np.ndarray,
        asset_ids: list[int],
        scale: float,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
    ) -> list[list[int]]:
        """Gruppiert Assets per Single-Linkage-Clustering bei einem gegebenen Distanz-Schwellenwert.

        Verwendet scipy's hierarchisches Single-Linkage-Clustering. Bei
        Single-Linkage werden zwei Cluster zusammengeführt, sobald das
        nächste Paar (ein Element aus jedem Cluster) eine Distanz <= scale
        hat. Das entspricht konzeptuell dem Schnitt eines Dendrogramms bei
        Höhe scale.

        Diese Methode wird typischerweise verwendet, um für einen bekannten
        Persistenzwert (z.B. aus extract_persistent_h0_features) die
        konkreten Asset-Gruppen zu materialisieren.

        Args:
            dist_matrix: Quadratische, symmetrische Distanzmatrix der Form
                (n, n) mit Nullen auf der Diagonalen. Muss zur Reihenfolge
                von asset_ids korrespondieren.
            asset_ids: Liste der Asset-IDs in derselben Reihenfolge wie die
                Zeilen/Spalten von dist_matrix.
            scale: Distanz-Schwellenwert, bei dem der Dendrogramm-Schnitt
                erfolgt. Assets innerhalb dieser Distanz werden zum selben
                Cluster zusammengefasst.
            min_cluster_size: Minimale Anzahl Assets, die ein Cluster haben
                muss, um in der Ausgabe zu erscheinen (Standard: 2).
                Singletons werden damit herausgefiltert.

        Returns:
            Liste von Clustern, absteigend nach Clustergrösse sortiert
            (Gleichstand: alphabetisch nach Asset-ID). Jeder Cluster ist
            eine Liste von Asset-IDs (int). Leere Liste wenn weniger als
            2 Assets übergeben wurden oder kein Cluster die Mindestgrösse
            erreicht.
        """
        if len(asset_ids) < 2:
            return []

        condensed = squareform(dist_matrix, checks=False)
        z = linkage(condensed, method="single")
        labels = fcluster(z, t=scale, criterion="distance")

        groups = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(int(asset_ids[idx]))

        clusters = [members for members in groups.values() if len(members) >= min_cluster_size]
        clusters = sorted(clusters, key=lambda x: (-len(x), x))
        return clusters

        