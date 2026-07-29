
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
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

    @staticmethod
    def residualize_returns(
        returns_df: pd.DataFrame,
        *,
        factor_columns: list[str],
    ) -> pd.DataFrame:
        """Entfernt den linearen Einfluss von Faktor-Assets aus jeder anderen Spalte.

        Regressiert jede Spalte von returns_df (ausser factor_columns) einzeln per
        OLS (mit Intercept) auf die factor_columns und gibt die Residuen zurueck.
        Mit einer Faktor-Spalte (z.B. nur "BTC") ist das eine einfache lineare
        Regression; mit mehreren (z.B. "BTC" und "ETH") eine multiple Regression.
        Die Faktor-Spalten selbst werden aus dem Ergebnis entfernt, da ihr eigenes
        Residuum gegen sich selbst degenerate (~0) waere.

        Args:
            returns_df: DataFrame (n_timestamps, n_assets), z.B. Return-Matrix aus
                einer Point Cloud. Muss alle in factor_columns genannten Spalten
                enthalten.
            factor_columns: Spaltennamen, die als Regressoren (Markt-/Faktor-
                Renditen) verwendet werden, z.B. ["BTC"] oder ["BTC", "ETH"].

        Returns:
            DataFrame mit demselben Index wie returns_df und einer Spalte pro
            Nicht-Faktor-Asset aus returns_df; Werte sind die Regressions-Residuen
            (der Teil der Rendite, der nicht durch die Faktoren erklaert wird).

        Raises:
            KeyError: falls eine factor_column nicht in returns_df vorkommt.
        """
        missing_columns = [c for c in factor_columns if c not in returns_df.columns]
        if missing_columns:
            raise KeyError(f"Faktor-Spalten nicht in returns_df gefunden: {missing_columns}")

        factors = returns_df[factor_columns].to_numpy()
        design = np.column_stack([np.ones(len(factors)), factors])

        residuals = {}
        for column in returns_df.columns:
            if column in factor_columns:
                continue
            y = returns_df[column].to_numpy()
            beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            residuals[column] = y - design @ beta

        return pd.DataFrame(residuals, index=returns_df.index)

    @staticmethod
    def residualize_against_pca(
        returns_df: pd.DataFrame,
        *,
        n_components: int = 1,
    ) -> pd.DataFrame:
        """Entfernt den Einfluss der ersten n_components Hauptkomponenten aus jeder Spalte.

        Statt gegen ein benanntes Faktor-Asset (siehe residualize_returns) wird
        hier ein synthetischer Markt-Faktor per PCA aus returns_df selbst
        extrahiert: die ersten n_components Hauptkomponenten-Scores dienen als
        Regressoren fuer eine OLS-Regression (mit Intercept) jeder einzelnen
        Spalte, analog zu residualize_returns. Nuetzlich, wenn kein einzelnes
        Asset (z.B. BTC) als Marktfaktor unterstellt werden soll, sondern die
        dominante gemeinsame Bewegungsrichtung des gesamten Fensters.

        Args:
            returns_df: DataFrame (n_timestamps, n_assets), z.B. Return-Matrix aus
                einer Point Cloud.
            n_components: Anzahl Hauptkomponenten, die als Faktoren verwendet
                werden (Standard: 1 = nur die erste/dominante Komponente).

        Returns:
            DataFrame mit demselben Index und denselben Spalten wie returns_df;
            Werte sind die Regressions-Residuen (der Teil der Rendite, der nicht
            durch die ersten n_components Hauptkomponenten erklaert wird).
        """
        components = PCA(n_components=n_components, random_state=0).fit_transform(
            returns_df.to_numpy()
        )
        design = np.column_stack([np.ones(len(components)), components])

        residuals = {}
        for column in returns_df.columns:
            y = returns_df[column].to_numpy()
            beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            residuals[column] = y - design @ beta

        return pd.DataFrame(residuals, index=returns_df.index)

    @staticmethod
    def compute_rolling_volume_table(
        asset_daily_df: pd.DataFrame,
        *,
        id_column: str,
        date_column: str,
        volume_column: str,
        lookback_days: int,
    ) -> pd.DataFrame:
        """Berechnet den rollierenden Median-Volumen pro Asset ueber lookback_days.

        Gedacht als Glaettung des taeglichen 24h-USD-Volumens, um einzelne
        Volumen-Spitzen (die sonst ein Asset kurzzeitig in einen anderen
        Basket springen liessen) auszugleichen. asset_daily_df enthaelt pro
        Asset nur Zeilen fuer Tage, an denen es die Basis-Volumenschwelle
        (siehe 01_assets.ipynb Filter 2) bereits erfuellt hat -- diese Funktion
        selbst nimmt keine weitere Filterung vor.

        Args:
            asset_daily_df: DataFrame mit mindestens id_column, date_column,
                volume_column.
            id_column: Spaltenname der Asset-ID.
            date_column: Spaltenname des Datums.
            volume_column: Spaltenname des taeglichen USD-Volumens.
            lookback_days: Fenstergroesse (und min_periods) fuer den
                rollierenden Median.

        Returns:
            Kopie der drei genannten Spalten, sortiert nach
            (id_column, date_column), mit zwei neuen Spalten:
                - rolling_median_volume_24h_usd
                - log_rolling_median_volume_24h_usd (log10, NaN wenn <= 0)
        """
        volume_df = asset_daily_df[[id_column, date_column, volume_column]].copy()
        volume_df[date_column] = pd.to_datetime(volume_df[date_column])
        volume_df = volume_df.sort_values([id_column, date_column])

        volume_df["rolling_median_volume_24h_usd"] = (
            volume_df
            .groupby(id_column)[volume_column]
            .transform(
                lambda series: series
                .rolling(lookback_days, min_periods=lookback_days)
                .median()
            )
        )

        volume_df["log_rolling_median_volume_24h_usd"] = np.log10(
            volume_df["rolling_median_volume_24h_usd"].where(
                volume_df["rolling_median_volume_24h_usd"] > 0
            )
        )

        return volume_df

    @staticmethod
    def assign_volume_basket(
        volume: float,
        *,
        basket_upper_bounds: list[float],
    ) -> str | None:
        """Ordnet einen Volumenwert einem von N Baskets zu.

        basket_upper_bounds ist eine aufsteigend sortierte Liste von N-1
        Obergrenzen fuer einen N-Basket-Split -- die Anzahl Baskets und ihre
        Grenzen sind damit frei waehlbar, nicht auf 4 fest verdrahtet.
        "Basket 1" ist immer der oberste (liquideste) Basket ohne Obergrenze,
        "Basket N" der unterste (gerade noch die Eingangsschwelle erfuellend).
        Fuer einen 2-Basket-Split reicht eine einzelne Grenze, z.B.
        basket_upper_bounds=[250_000_000].

        Args:
            volume: Volumenwert (z.B. rolling_median_volume_24h_usd). NaN
                (z.B. zu wenig Historie fuer den rollierenden Median) ergibt
                keinen Basket.
            basket_upper_bounds: Aufsteigend sortierte Liste von N-1
                Volumen-Obergrenzen.

        Returns:
            "Basket {n}" mit n von 1 (hoechstes Volumen) bis N (niedrigstes),
            oder None wenn volume NaN ist.
        """
        if pd.isna(volume):
            return None

        n_baskets = len(basket_upper_bounds) + 1
        for index, upper_bound in enumerate(basket_upper_bounds):
            if volume < upper_bound:
                return f"Basket {n_baskets - index}"

        return "Basket 1"

    @staticmethod
    def build_volume_baskets(
        rolling_volume_df: pd.DataFrame,
        selected_date,
        *,
        id_column: str,
        date_column: str,
        basket_upper_bounds: list[float],
    ) -> pd.DataFrame:
        """Baut die Basket-Zuordnung fuer alle Assets an einem bestimmten Datum.

        Args:
            rolling_volume_df: Ausgabe von compute_rolling_volume_table, muss
                eine Spalte "rolling_median_volume_24h_usd" enthalten.
            selected_date: Datum, fuer das die Baskets gebaut werden.
            id_column: Spaltenname der Asset-ID.
            date_column: Spaltenname des Datums.
            basket_upper_bounds: siehe assign_volume_basket.

        Returns:
            Zeilen von rolling_volume_df fuer selected_date mit neuer Spalte
            "basket", absteigend nach rolling_median_volume_24h_usd sortiert.
            Assets ohne Basket (NaN-Volumen) werden entfernt.
        """
        selected_date = pd.Timestamp(selected_date)

        basket_df = rolling_volume_df.loc[
            rolling_volume_df[date_column] == selected_date
        ].copy()

        basket_df["basket"] = basket_df["rolling_median_volume_24h_usd"].apply(
            lambda volume: HelperFunctions.assign_volume_basket(
                volume,
                basket_upper_bounds=basket_upper_bounds,
            )
        )

        basket_df = (
            basket_df
            .loc[basket_df["basket"].notna()]
            .sort_values("rolling_median_volume_24h_usd", ascending=False)
            .reset_index(drop=True)
        )

        return basket_df

    @staticmethod
    def compare_baskets(
        previous_basket_df: pd.DataFrame,
        current_basket_df: pd.DataFrame,
        *,
        id_column: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Vergleicht Basket-Zuordnungen zweier Daten (siehe build_volume_baskets).

        Args:
            previous_basket_df: Basket-Zuordnung des frueheren Datums.
            current_basket_df: Basket-Zuordnung des aktuellen Datums.
            id_column: Spaltenname der Asset-ID.

        Returns:
            (comparison_df, left_df, transition_df):
                - comparison_df: current_basket_df + previous_basket +
                  basket_change ("entered universe" | "switched basket" |
                  "unchanged").
                - left_df: Assets, die im aktuellen Basket-Set nicht mehr
                  vorkommen ("left universe").
                - transition_df: Kreuztabelle previous_basket x basket.
        """
        previous_basket_map = previous_basket_df.set_index(id_column)["basket"]

        comparison_df = current_basket_df.copy()
        comparison_df["previous_basket"] = comparison_df[id_column].map(
            previous_basket_map
        )

        comparison_df["basket_change"] = np.select(
            [
                comparison_df["previous_basket"].isna(),
                comparison_df["previous_basket"].astype(str)
                != comparison_df["basket"].astype(str),
            ],
            [
                "entered universe",
                "switched basket",
            ],
            default="unchanged",
        )

        previous_assets = set(previous_basket_df[id_column])
        current_assets = set(current_basket_df[id_column])
        left_assets = previous_assets - current_assets

        left_df = (
            previous_basket_df
            .loc[
                previous_basket_df[id_column].isin(left_assets),
                [
                    id_column,
                    "basket",
                    "rolling_median_volume_24h_usd",
                ],
            ]
            .rename(columns={"basket": "previous_basket"})
            .copy()
        )

        left_df["basket_change"] = "left universe"

        transition_df = (
            comparison_df
            .loc[comparison_df["previous_basket"].notna()]
            .groupby(["previous_basket", "basket"], observed=True)
            .size()
            .rename("asset_count")
            .reset_index()
            .pivot(index="previous_basket", columns="basket", values="asset_count")
            .fillna(0)
            .astype(int)
        )

        return comparison_df, left_df, transition_df

    @staticmethod
    def build_basket_comparison(
        rolling_volume_df: pd.DataFrame,
        *,
        previous_date,
        selected_date,
        id_column: str,
        date_column: str,
        basket_upper_bounds: list[float],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Baut Basket-Zuordnungen fuer zwei Daten und vergleicht sie.

        Args:
            rolling_volume_df: siehe build_volume_baskets.
            previous_date: frueheres Vergleichsdatum.
            selected_date: aktuelles Datum.
            id_column: Spaltenname der Asset-ID.
            date_column: Spaltenname des Datums.
            basket_upper_bounds: siehe assign_volume_basket.

        Returns:
            (previous_basket_df, current_basket_df, comparison_df, left_df,
            transition_df) -- siehe build_volume_baskets/compare_baskets.
        """
        previous_basket_df = HelperFunctions.build_volume_baskets(
            rolling_volume_df,
            previous_date,
            id_column=id_column,
            date_column=date_column,
            basket_upper_bounds=basket_upper_bounds,
        )

        current_basket_df = HelperFunctions.build_volume_baskets(
            rolling_volume_df,
            selected_date,
            id_column=id_column,
            date_column=date_column,
            basket_upper_bounds=basket_upper_bounds,
        )

        comparison_df, left_df, transition_df = HelperFunctions.compare_baskets(
            previous_basket_df,
            current_basket_df,
            id_column=id_column,
        )

        return (
            previous_basket_df,
            current_basket_df,
            comparison_df,
            left_df,
            transition_df,
        )

    @staticmethod
    def describe_volume_baskets(
        basket_upper_bounds: list[float],
        min_volume_24h_usd: float,
    ) -> list[str]:
        """Baut lesbare Basket-Definitionen fuer eine gegebene Schwellenliste.

        Args:
            basket_upper_bounds: siehe assign_volume_basket.
            min_volume_24h_usd: absolute Eingangsschwelle (unteres Ende von
                Basket N).

        Returns:
            Liste von Strings, einer pro Basket, von "Basket 1" (oben) bis
            "Basket N" (unten), z.B.
            "Basket 2: $100,000,000 <= volume < $250,000,000".
        """
        n_baskets = len(basket_upper_bounds) + 1
        bounds = [min_volume_24h_usd] + list(basket_upper_bounds) + [None]

        lines = []
        for index in range(n_baskets):
            label = f"Basket {n_baskets - index}"
            lower = bounds[index]
            upper = bounds[index + 1]
            if upper is None:
                lines.append(f"{label}: volume >= ${lower:,.0f}")
            else:
                lines.append(f"{label}: ${lower:,.0f} <= volume < ${upper:,.0f}")

        return lines

        