import datetime

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from IPython.display import display, clear_output

from Src.clustering.helpers import HelperFunctions

# Manche Umgebungen (z.B. VS Code) erkennen den Default automatisch als
# zusammengesetzten, "+"-getrennten Wert (z.B. "vscode+notebook"). In dem Fall
# rendert jeder fig.show()-Aufruf einmal PRO Renderer in der Liste -> dieselbe
# Figur erscheint mehrfach, auch ohne jede Interaktion. Nur den ersten Eintrag
# behalten (die von Plotly selbst erkannte, zur Umgebung passende Wahl), statt
# ihn durch einen fest einprogrammierten Renderer zu ersetzen, der in dieser
# Umgebung evtl. gar nicht unterstützt wird (-> leerer Output).
if "+" in pio.renderers.default:
    pio.renderers.default = pio.renderers.default.split("+")[0]
# "browser" öffnet einen System-Browser-Prozess und blockiert/crasht den Kernel
# bei headless nbconvert-Ausführung -> stattdessen inline im Notebook rendern.
elif pio.renderers.default == "browser":
    pio.renderers.default = "notebook"

# Referenzpalette (siehe dataviz-Skill): diverging blue<->red, neutraler Grau-Mittelpunkt.
# Passt zu Returns, da diese natürlich um 0 zentriert und vorzeichenbehaftet sind.
_DIVERGING_SCALE = [
    [0.0, "#0d366b"],
    [0.25, "#256abf"],
    [0.5, "#f0efec"],
    [0.75, "#c1443f"],
    [1.0, "#7a1f1f"],
]
_MARKER_COLOR = "#2a78d6"
# Matplotlib-Äquivalent von _DIVERGING_SCALE (gleiche Stops, gleichmässig verteilt
# bei 0/0.25/0.5/0.75/1.0) für Heatmaps ausserhalb von Plotly.
_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "diverging_blue_red", [stop[1] for stop in _DIVERGING_SCALE]
)

# Kategoriale Referenzpalette (dataviz-Skill), feste Reihenfolge, nie zyklisch neu
# zugewiesen. Slot 0 ist reserviert für "noch nicht verschmolzen / Singleton" (grau).
_CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
_SINGLETON_GREY = "#c3c2b7"
_INK_MUTED = "#898781"
_INK_SECONDARY = "#52514e"


def _compute_vr_layout(distance_matrix):
    """MDS-2D-Layout + Single-Linkage-Baum, wiederverwendbar für Skelett-Plots.

    Ausgelagert aus plot_vietoris_rips_epsilon_slider, damit sowohl das
    eigenständige Slider-Widget als auch build_persistent_cluster_widget
    (das sein eigenes epsilon über einen gemeinsamen Output-Bereich statt
    über ein zusätzliches, verschachteltes Output-Widget steuert) dieselbe
    MDS-Berechnung nutzen können, ohne sie zu duplizieren.
    """
    coords = MDS(
        n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress=False
    ).fit_transform(distance_matrix)
    z = linkage(squareform(distance_matrix, checks=False), method="single")
    return coords, z


def _build_vr_skeleton_figure(
    distance_matrix, asset_symbols, epsilon, coords, z, *,
    min_cluster_size=2, show_radius=True, width=None, height=650,
):
    """Baut die reine Vietoris-Rips-1-Skelett-Figur (keine Anzeige/Widgets).

    Siehe plot_vietoris_rips_epsilon_slider für die Bedeutung der Argumente.
    coords/z kommen von _compute_vr_layout (einmal pro Fenster berechnet,
    nicht bei jeder epsilon-Änderung neu).

    Args:
        width, height: Feste Pixel-Grösse der Figure. width=None (Default)
            lässt Plotly die Breite responsiv bestimmen; explizit gesetzt,
            z.B. um dieselbe Grösse wie ein danebenstehendes Matplotlib-
            Dendrogramm zu erzwingen (siehe build_persistent_cluster_widget).
    """
    n = len(asset_symbols)
    labels = fcluster(z, t=epsilon, criterion="distance")
    sizes = {lbl: int(np.sum(labels == lbl)) for lbl in set(labels)}

    color_map = {}
    color_i = 0
    for lbl in sorted(sizes, key=lambda l: -sizes[l]):
        if sizes[lbl] >= min_cluster_size:
            color_map[lbl] = _CATEGORICAL[color_i % len(_CATEGORICAL)]
            color_i += 1
        else:
            color_map[lbl] = _SINGLETON_GREY
    node_colors = [color_map[lbl] for lbl in labels]

    edge_x, edge_y = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] <= epsilon:
                edge_x += [coords[i, 0], coords[j, 0], None]
                edge_y += [coords[i, 1], coords[j, 1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line={"width": 1, "color": _INK_MUTED}, opacity=0.5,
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers+text",
        marker={"size": 16, "color": node_colors, "line": {"width": 1, "color": "white"}},
        text=asset_symbols, textposition="top center",
        textfont={"size": 10, "color": _INK_SECONDARY},
        hovertext=[f"{s}<br>Cluster {lbl} ({sizes[lbl]} Assets)" for s, lbl in zip(asset_symbols, labels)],
        hoverinfo="text", showlegend=False,
    ))

    shapes = []
    if show_radius and epsilon > 0:
        radius = epsilon / 2.0
        for i in range(n):
            cx, cy = coords[i, 0], coords[i, 1]
            shapes.append({
                "type": "circle", "xref": "x", "yref": "y",
                "x0": cx - radius, "y0": cy - radius, "x1": cx + radius, "y1": cy + radius,
                "line": {"color": node_colors[i], "width": 1},
                "fillcolor": node_colors[i], "opacity": 0.12,
                "layer": "below",
            })

    n_clusters = len(sizes)
    n_big = sum(1 for lbl in sizes if sizes[lbl] >= min_cluster_size)
    layout_kwargs = {
        "title": (
            f"Vietoris-Rips 1-Skelett bei epsilon={epsilon:.4f}  |  "
            f"{n_clusters} Cluster ({n_big} mit >= {min_cluster_size} Assets)"
        ),
        "xaxis": {"visible": False}, "yaxis": {"visible": False, "scaleanchor": "x", "scaleratio": 1},
        "shapes": shapes,
        "height": height, "margin": {"l": 10, "r": 10, "t": 50, "b": 10},
        "plot_bgcolor": "#fcfcfb",
    }
    if width is not None:
        layout_kwargs["width"] = width
    fig.update_layout(**layout_kwargs)
    return fig


def plot_vietoris_rips_epsilon_slider(
    distance_matrix, asset_symbols, max_edge_length=None, min_cluster_size=2,
    show_radius=True,
):
    """Interaktiver Vietoris-Rips-Graph: Assets als Knoten, Kanten bei Distanz <= epsilon.

    Zeigt direkt das Geburts-/Verschmelzungsverhalten der H0-Features:
    bei epsilon=0 ist jedes Asset isoliert (n Cluster), mit wachsendem epsilon verschmelzen
    Cliquen, bis am Ende alles ein einziger Cluster ist. Die Knotenposition kommt aus MDS
    auf der Korrelationsdistanzmatrix (Assets mit ähnlichem Return-Profil liegen nah beieinander),
    die Kanten sind der eigentliche Vietoris-Rips-1-Skelett bei der gewählten Schwelle.

    Eigenständiges Widget mit eigenem Slider + Output. Wird aktuell NICHT von
    build_persistent_cluster_widget verwendet (das Skelett wurde dort entfernt),
    bleibt aber für eigenständige Exploration in einer einzelnen Zelle nutzbar.

    Args:
        distance_matrix: (n_assets, n_assets) symmetrische Korrelationsdistanzmatrix,
            z.B. helper.correlation_distance(returns_df). Diagonale = 0.
        asset_symbols: Liste der Asset-Symbole, Reihenfolge passend zu distance_matrix.
        max_edge_length: Oberes Ende des epsilon-Sliders. Default: max(distance_matrix).
        min_cluster_size: Cluster unterhalb dieser Grösse werden als Singletons (grau)
            statt mit einer Cluster-Farbe dargestellt.
        show_radius: Falls True, wird um jeden Knoten ein Kreis mit Radius=epsilon/2
            gezeichnet -- berühren sich zwei Kreise, liegt eine Kante vor. Rein
            illustrativ: MDS bildet die Distanzmatrix nur approximativ in 2D ab
            (Stress > 0 bei > 2 "wirklichen" Dimensionen), die Kreise zeigen also
            die ungefähre, nicht die exakte Nachbarschaftsschwelle.
    """
    n = len(asset_symbols)
    if distance_matrix.shape != (n, n):
        raise ValueError(f"distance_matrix hat Shape {distance_matrix.shape}, erwartet ({n}, {n}).")

    dist_max = float(np.max(distance_matrix))
    eps_max = max_edge_length if max_edge_length is not None else dist_max

    coords, z = _compute_vr_layout(distance_matrix)

    slider = widgets.FloatSlider(
        min=0.0, max=eps_max, step=eps_max / 200,
        value=0.0, description="epsilon:", continuous_update=False,
        readout_format=".4f", style={"description_width": "initial"},
        layout=widgets.Layout(width="600px"),
    )
    out = widgets.Output()

    def render(*_change):
        fig = _build_vr_skeleton_figure(
            distance_matrix, asset_symbols, slider.value, coords, z,
            min_cluster_size=min_cluster_size, show_radius=show_radius,
        )
        # Eigener Output + explizites clear_output(wait=True) VOR jedem neuen Plot:
        # go.Figure().show() allein ersetzt die vorherige Ausgabe nicht zuverlässig
        # innerhalb von widgets.interact -> ohne das hängt sich bei jedem Schritt
        # des Sliders eine weitere Figur an die Zelle an, statt sie zu ersetzen.
        with out:
            clear_output(wait=True)
            fig.show()

    slider.observe(render, names="value")
    display(widgets.VBox([slider, out]))
    render()


def plot_correlation_heatmap(
    matrix, asset_symbols, *,
    value_range=(-1.0, 1.0),
    cmap=None,
    colorbar_label="Korrelation",
    title=None,
    annotate=True,
    annotate_fmt="{:.2f}",
    annotate_fontsize=None,
    max_annotated_assets=30,
    figsize=None,
    ax=None,
    show=True,
):
    """Heatmap einer (n_assets, n_assets) Matrix mit optionalen Zahlen pro Zelle.

    Generische Heatmap-Funktion für sowohl die Korrelationsmatrix (Werte in
    [-1, 1], diverging Farbskala zentriert auf 0, Default) als auch die
    Korrelationsdistanzmatrix (Werte in [0, sqrt(2)] -- dafür value_range und
    cmap explizit übergeben) -- ersetzt die zuvor in 02_pointclouds.ipynb /
    03_residual_pointclouds.ipynb duplizierten ad-hoc imshow-Blöcke ohne
    Zahlen-Annotation.

    Args:
        matrix: (n_assets, n_assets) symmetrische Matrix.
        asset_symbols: Liste der Asset-Symbole, Reihenfolge passend zu matrix.
        value_range: (vmin, vmax) für die Farbskala. Default (-1, 1) für
            Korrelationen; für Distanzen z.B. (0, sqrt(2)) übergeben.
        cmap: Matplotlib-Colormap oder Name. Default: diverging blue<->red
            (siehe _DIVERGING_SCALE), passend für um 0 zentrierte Werte.
        colorbar_label: Beschriftung der Farbskala.
        title: Optionaler Titel. Default generisch mit Asset-Anzahl.
        annotate: Falls True, wird jede Zelle mit ihrem Wert beschriftet.
            Wird automatisch deaktiviert, wenn mehr als max_annotated_assets
            Assets vorhanden sind (sonst unleserlich).
        annotate_fmt: Format-String für die Zellen-Beschriftung.
        annotate_fontsize: Schriftgrösse der Zellen-Beschriftung. Default (None)
            skaliert automatisch mit der Asset-Anzahl (kleinere Schrift bei
            mehr Assets), damit die Zahlen bei vielen Assets nicht überlappen,
            bei wenigen Assets aber gut lesbar gross bleiben.
        max_annotated_assets: Schwelle, ab der annotate automatisch False wird.
        figsize: Optionale (Breite, Höhe) in Zoll. Ignoriert, falls ax übergeben wird.
        ax: Optionale matplotlib Axes zum Zeichnen (z.B. eine in Tkinter
            eingebettete FigureCanvasTkAgg-Achse). Falls None, wird eine neue
            Figure erstellt.
        show: Falls True (Default) und ax None ist, wird plt.show() aufgerufen.
            Beim Einbetten (ax übergeben) sollte dies auf False gesetzt werden.

    Returns:
        (fig, ax): die verwendete Figure und Axes.
    """
    n = len(asset_symbols)
    if matrix.shape != (n, n):
        raise ValueError(f"matrix hat Shape {matrix.shape}, erwartet ({n}, {n}).")

    if cmap is None:
        cmap = _DIVERGING_CMAP

    owns_figure = ax is None
    if owns_figure:
        if figsize is None:
            figsize = (max(8, n * 0.35), max(6, n * 0.3))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    vmin, vmax = value_range
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, label=colorbar_label)

    ax.set_xticks(range(n)); ax.set_xticklabels(asset_symbols, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(asset_symbols, fontsize=7)
    ax.set_title(title or f"Heatmap ({n} Assets)")

    if annotate and n <= max_annotated_assets:
        # Skaliert von ~11pt (wenige Assets) bis ~6pt (nahe max_annotated_assets),
        # damit die Zahlen bei kleinen Matrizen gut lesbar sind und bei grossen
        # Matrizen trotzdem nicht überlappen.
        fontsize = annotate_fontsize if annotate_fontsize is not None else max(6.0, min(11.0, 220.0 / n))
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                ax.text(j, i, annotate_fmt.format(value), ha="center", va="center",
                        fontsize=fontsize, color="black")

    if owns_figure:
        fig.tight_layout()
        if show:
            plt.show()
            # Ohne explizites close() bleibt die Figure im globalen pyplot-Zustand
            # registriert -> wird sie innerhalb eines ipywidgets.Output erzeugt
            # (with out: ...), führt das beim Verlassen des Blocks zu einem
            # ZWEITEN, doppelten Rendering derselben Figur.
            plt.close(fig)

    return fig, ax


def plot_dendrogram_with_clusters(
    distance_matrix, asset_symbols, clusters, *, z=None,
    persistence_threshold=None, epsilon_cap=None,
    figsize=None, ax=None, show=True, y_max=None, dpi=None,
):
    """Single-Linkage-Dendrogramm mit eingefärbten, hierarchie-bereinigten Clustern.

    Im Unterschied zu plot_dendrogram (ein einziger globaler Höhen-Schnitt) werden
    hier die von HelperFunctions.get_persistent_clusters gefundenen, jeweils an
    ihrer eigenen Skala geschnittenen Cliquen eingefärbt: jeder Ast INNERHALB
    eines gefundenen Clusters (inkl. seiner eigenen Merge-Linie) bekommt dessen
    Farbe, alles ausserhalb (Vorfahren, die selbst nicht mehr persistent genug
    waren, und Äste ohne jeden legitimen Cluster) bleibt grau. Assets, die zu
    keinem Cluster gehören, erscheinen so weiterhin vollständig im Baum, nur
    ungefärbt -- alle Assets des Baskets sind immer im Bild.

    Args:
        distance_matrix: (n_assets, n_assets) symmetrische Korrelationsdistanzmatrix.
        asset_symbols: Liste der Asset-Symbole, Reihenfolge passend zu distance_matrix.
        clusters: Liste von Cluster-Dicts wie von get_persistent_clusters
            zurückgegeben (braucht die Schlüssel "node_id" und "size").
        z: Optionale, bereits berechnete Linkage-Matrix (von get_persistent_clusters
            zurückgegeben) -- vermeidet doppelte Berechnung. Falls None, wird sie
            neu aus distance_matrix berechnet.
        persistence_threshold: Nur für die Titelzeile/Legende.
        epsilon_cap: Falls gesetzt, gestrichelte Referenzlinie bei dieser Höhe
            (obere Grenze der Cluster-Suche).
        figsize, ax, show: siehe plot_dendrogram.
        dpi: Falls gesetzt, feste Auflösung für figsize -> Pixel-Grösse
            (figsize_zoll * dpi), unabhängig von rcParams -- z.B. um exakt
            dieselbe Pixel-Grösse wie eine danebenstehende Plotly-Figur zu
            erzwingen (siehe build_persistent_cluster_widget).
        y_max: Falls gesetzt, feste y-Achse [0, y_max] statt Auto-Skalierung auf
            den tatsächlichen Merge-Bereich dieses Fensters -- damit bleibt die
            Achse zwischen Fenstern/Baskets vergleichbar und epsilon_cap landet
            nicht zufällig am oberen Bildrand.

    Returns:
        Die verwendete scipy-Linkage-Matrix z.
    """
    n = len(asset_symbols)
    if distance_matrix.shape != (n, n):
        raise ValueError(f"distance_matrix hat Shape {distance_matrix.shape}, erwartet ({n}, {n}).")

    if z is None:
        z = linkage(squareform(distance_matrix, checks=False), method="single")

    # children-Map aus z rekonstruieren (dieselbe Struktur wie in
    # HelperFunctions.get_persistent_clusters), um pro Cluster den kompletten
    # Teilbaum unterhalb seines node_id einzufärben.
    children = {}
    for row_idx, (c1, c2, _dist, _count) in enumerate(z):
        children[n + row_idx] = (int(c1), int(c2))

    node_color: dict[int, str] = {}
    legend_entries = []
    clusters_sorted = sorted(clusters, key=lambda c: -c["size"])
    for i, cluster in enumerate(clusters_sorted):
        color = _CATEGORICAL[i % len(_CATEGORICAL)]
        legend_entries.append((color, cluster))
        stack = [cluster["node_id"]]
        while stack:
            node_id = stack.pop()
            node_color[node_id] = color
            if node_id in children:
                left, right = children[node_id]
                stack.append(left)
                stack.append(right)

    def link_color_func(node_id):
        return node_color.get(node_id, _INK_MUTED)

    owns_figure = ax is None
    if owns_figure:
        if figsize is None:
            figsize = (max(10, n * 0.35), 6)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    dendrogram(z, labels=asset_symbols, ax=ax, link_color_func=link_color_func)

    has_legend_entries = False
    if epsilon_cap is not None:
        ax.axhline(y=epsilon_cap, color=_INK_SECONDARY, linestyle="--", linewidth=1.2,
                   label=f"epsilon_cap = {epsilon_cap:.4f}")
        has_legend_entries = True

    for color, cluster in legend_entries:
        ax.plot([], [], color=color, linewidth=8,
                 label=f"{cluster['size']} Assets (Persistenz {cluster['persistence']:.4f})")
        has_legend_entries = True

    if has_legend_entries:
        ax.legend(fontsize=8, loc="upper right")

    ax.set_ylabel("Korrelationsdistanz (epsilon)")
    title = f"Single-Linkage-Dendrogramm ({n} Assets, {len(clusters)} persistente Cluster)"
    if persistence_threshold is not None:
        title += f"\npersistence_threshold={persistence_threshold:.4f}"
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)

    if y_max is not None:
        ax.set_ylim(0, y_max)

    if owns_figure:
        fig.tight_layout()
        if show:
            plt.show()
            plt.close(fig)  # siehe Kommentar in plot_correlation_heatmap

    return z


def plot_h0_barcode_on_ax(distance_matrix, asset_symbols, ax, *, x_max=None):
    """Zeichnet den H0-Persistenz-Barcode als Matplotlib-Plot.

    Ein Balken pro Cluster von birth bis death=epsilon (siehe
    build_persistent_cluster_widget), auf eine übergebene Matplotlib-Axes
    gezeichnet -- damit kann der Barcode z.B. als zweites Panel in derselben
    Figure wie das Dendrogramm liegen (ein gemeinsamer plt.show()-Aufruf).
    Das garantiert dasselbe Resize-Verhalten wie die Korrelations-/Distanz-
    Heatmap (die bereits nachweislich sauber sowohl horizontal als auch
    vertikal mit dem Notebook-Fenster mitskaliert).

    Args:
        distance_matrix: (n_assets, n_assets) symmetrische Korrelationsdistanzmatrix.
        asset_symbols: Liste der Asset-Symbole, Reihenfolge passend zu distance_matrix.
        ax: Matplotlib-Axes zum Zeichnen.
        x_max: Falls gesetzt, feste x-Achse [0, x_max] statt Auto-Skalierung auf
            die grösste beobachtete Death-Distanz dieses Fensters.

    Returns:
        Anzahl gezeichneter Balken (int).
    """
    n = len(asset_symbols)
    z = linkage(squareform(distance_matrix, checks=False), method="single")

    members = {i: [asset_symbols[i]] for i in range(n)}
    birth = {i: 0.0 for i in range(n)}
    rows = []
    for row_idx, (c1, c2, dist, _count) in enumerate(z):
        c1, c2 = int(c1), int(c2)
        merged = members[c1] + members[c2]
        cluster_id = n + row_idx
        members[cluster_id] = merged
        birth[cluster_id] = float(dist)
        for c in (c1, c2):
            rows.append({"birth": birth[c], "death": float(dist), "size": len(members[c])})
    rows.sort(key=lambda r: r["death"])

    bound = x_max if x_max is not None else (max((r["death"] for r in rows), default=1.0) or 1.0)
    for y, r in enumerate(rows):
        color = _CATEGORICAL[min(r["size"] - 2, len(_CATEGORICAL) - 1)] if r["size"] >= 2 else _SINGLETON_GREY
        ax.plot([r["birth"], r["death"]], [y, y], color=color, linewidth=6, solid_capstyle="butt")

    ax.set_xlim(0, bound * 1.05)
    ax.set_ylim(-1, max(len(rows), 1))
    ax.set_yticks([])
    ax.set_xlabel("Korrelationsdistanz (epsilon)")
    ax.set_ylabel("Cluster-Intervall (Todeszeitpunkt)")
    ax.set_title(f"Cluster-Lebensdauer-Barcode ({len(rows)} Intervalle)")

    return len(rows)


def build_persistent_cluster_widget(
    *,
    ohlcv_df, universe_df, rolling_volume_df,
    id_to_symbol, id_column, date_column,
    basket_labels, basket_upper_bounds, excluded_ids,
    data_min, data_max,
    default_lookback_days=30, max_lookback_days=100,
    default_persistence_threshold=0.3,
    default_epsilon_cap=None,
    default_date=None,
    point_cloud_fn=None,
):
    """Interaktiver Tab-Explorer: pro Basket Lookback/Datum/Schwellen einstellbar,
    darunter Korrelations- und Distanz-Heatmap, Vietoris-Rips-Skelett, hierarchie-
    bereinigtes Dendrogramm und Persistenz-Barcode für das aktuell gewählte Fenster.

    Nutzt HelperFunctions.get_persistent_clusters: der EINE global
    persistenteste Knoten pro Fenster, sofern er die Schwelle schafft
    (siehe dortige Docstring für die genaue Logik).

    Args:
        ohlcv_df, universe_df: siehe HelperFunctions.build_point_cloud.
        rolling_volume_df: Ausgabe von HelperFunctions.compute_rolling_volume_table.
        id_to_symbol: dict asset_id -> Symbol (für Anzeige/Achsenbeschriftung).
        id_column, date_column: Spaltennamen in universe_df/rolling_volume_df.
        basket_labels: Liste der Basket-Labels (z.B. ["Basket 1", "Basket 2"]),
            ein Tab pro Label.
        basket_upper_bounds: siehe HelperFunctions.assign_volume_basket.
        excluded_ids: Set von Asset-IDs, die generell ausgeschlossen werden
            (z.B. Stablecoins), zusätzlich zur Basket-Filterung.
        data_min, data_max: datetime.date-Grenzen der verfügbaren OHLCV-Daten.
        default_lookback_days: Anfangswert des Lookback-Sliders (Tage). Entweder
            eine einzelne Zahl (für alle Baskets gleich) oder ein dict
            {basket_label: Tage} für kalibrierte, basketspezifische Werte
            (z.B. BEST_LOOKBACK aus Abschnitt 5) -- fehlt ein Label im dict,
            wird auf 30 zurückgefallen.
        max_lookback_days: Oberes Ende des Lookback-Sliders.
        default_persistence_threshold: Anfangswert der Persistenz-Schwelle.
            Wie default_lookback_days entweder eine Zahl oder ein dict
            {basket_label: Schwelle} (z.B. BEST_THRESHOLD aus Abschnitt 5).
        default_epsilon_cap: Anfangswert des Epsilon-Caps. Default (None) -> 1.0
            (einer der beiden PDF-Beispielwerte für max_edge_length, siehe
            HelperFunctions.get_persistent_clusters).
        default_date: Anfangsdatum (datetime.date). Default (None) -> data_max.
        point_cloud_fn: Callable mit Keyword-Argumenten (ohlcv_df, universe_df,
            end_date, window_size_days, exclude_asset_ids), das ein Dict im
            Format von HelperFunctions.build_point_cloud zurückgibt (oder
            None). Default (None) -> HelperFunctions.build_point_cloud selbst.
            Austauschbar, um z.B. residualisierte Punktwolken einzuhängen
            (siehe HelperFunctions.build_residual_point_cloud in
            03_residual_pointclouds.ipynb) -- window_size_days entspricht dann
            dem topologischen Lookback, während das (längere, feste)
            Regressionsfenster in der übergebenen Funktion selbst
            eingeschlossen ist.

    Returns:
        Das äussere widgets.Tab-Objekt (bereits via display() angezeigt).
    """
    if default_epsilon_cap is None:
        default_epsilon_cap = 1.0
    if point_cloud_fn is None:
        point_cloud_fn = HelperFunctions.build_point_cloud

    def _per_basket(value, label, fallback):
        return value.get(label, fallback) if isinstance(value, dict) else value

    all_asset_ids = set(universe_df[id_column].unique())

    # Datum wird EINMAL erstellt und über alle Basket-Tabs hinweg geteilt (statt
    # pro Tab neu), damit eine Änderung in einem Tab automatisch in allen
    # anderen übernommen wird. ipywidgets erlaubt, dieselbe Widget-Instanz an
    # mehreren Stellen im Widget-Baum einzuhängen -- jede Stelle ist eine eigene
    # Ansicht auf denselben Zustand und bleibt automatisch synchron; jedes Tab
    # registriert unten trotzdem seinen EIGENEN render() als Observer, damit
    # alle vier Baskets bei einer Datumsänderung neu zeichnen. Lookback bleibt
    # bewusst PRO Basket unabhängig (nicht Teil dieses Wunsches).
    date_picker = widgets.DatePicker(
        description="Datum:", value=(default_date or data_max), min=data_min, max=data_max,
    )
    date_minus = widgets.Button(description="-1 Tag", layout=widgets.Layout(width="80px"))
    date_plus = widgets.Button(description="+1 Tag", layout=widgets.Layout(width="80px"))

    def on_date_minus(_b):
        if date_picker.value is not None and date_picker.value > data_min:
            date_picker.value = date_picker.value - datetime.timedelta(days=1)

    def on_date_plus(_b):
        if date_picker.value is not None and date_picker.value < data_max:
            date_picker.value = date_picker.value + datetime.timedelta(days=1)

    date_minus.on_click(on_date_minus)
    date_plus.on_click(on_date_plus)

    def _build_tab(label):
        lb_default = _per_basket(default_lookback_days, label, 30)
        pt_default = _per_basket(default_persistence_threshold, label, 0.3)

        lookback_slider = widgets.IntSlider(
            value=lb_default, min=1, max=max_lookback_days, step=1,
            description="Lookback (Tage):", continuous_update=False,
            style={"description_width": "initial"},
        )
        lookback_minus = widgets.Button(description="-1 Tag", layout=widgets.Layout(width="80px"))
        lookback_plus = widgets.Button(description="+1 Tag", layout=widgets.Layout(width="80px"))

        persistence_slider = widgets.FloatSlider(
            value=pt_default, min=0.0, max=2.0, step=0.005,
            description="Persistenz-Schwelle:", continuous_update=False,
            readout_format=".3f", style={"description_width": "initial"},
            layout=widgets.Layout(width="420px"),
        )
        epsilon_cap_slider = widgets.FloatSlider(
            value=default_epsilon_cap, min=0.1, max=2.0, step=0.005,
            description="Epsilon-Cap:", continuous_update=False,
            readout_format=".3f", style={"description_width": "initial"},
            layout=widgets.Layout(width="420px"),
        )

        out = widgets.Output()
        dendro_out = widgets.Output()

        def on_lookback_minus(_b):
            if lookback_slider.value > lookback_slider.min:
                lookback_slider.value -= 1

        def on_lookback_plus(_b):
            if lookback_slider.value < lookback_slider.max:
                lookback_slider.value += 1

        lookback_minus.on_click(on_lookback_minus)
        lookback_plus.on_click(on_lookback_plus)

        # Schutz gegen mehrfaches, redundantes Rendern: manche Notebook-
        # Frontends (u.a. beobachtet in VS Code, insbesondere bei einem
        # Widget wie date_picker, das gleichzeitig in mehreren Tabs eingehängt
        # ist -- jeder Tab ist eine eigene Ansicht auf denselben Zustand) lösen
        # pro Nutzeraktion mehrere "value changed"-Events aus, statt nur eines.
        # Ohne Schutz würde jedes dieser redundanten Events einen kompletten
        # weiteren Render-Durchlauf auslösen -> derselbe Plot erscheint
        # mehrfach hintereinander. Da _last_render_key den zuletzt tatsächlich
        # gerenderten Zustand festhält, wird ein Durchlauf übersprungen,
        # sobald sich seit dem letzten Render NICHTS an den relevanten Werten
        # geändert hat -- unabhängig davon, wie oft render() aufgerufen wird.
        _last_render_key = [None]

        def render(*_change):
            lookback_days = lookback_slider.value
            date = date_picker.value
            persistence_threshold = persistence_slider.value
            epsilon_cap = epsilon_cap_slider.value

            render_key = (lookback_days, date, persistence_threshold, epsilon_cap)
            if render_key == _last_render_key[0]:
                return
            _last_render_key[0] = render_key

            # Bereich IMMER leeren, auch bei einem frühen return unten -- sonst
            # bleibt das Dendrogramm vom letzten gültigen Zustand sichtbar,
            # während "out" schon eine Fehlermeldung für den neuen
            # (ungültigen) Zustand zeigt.
            with dendro_out:
                clear_output(wait=True)

            with out:
                clear_output(wait=True)

                if date is None:
                    print("Bitte ein Datum wählen.")
                    return
                if date < data_min or date > data_max:
                    print(f"Bitte ein Datum zwischen {data_min} und {data_max} wählen.")
                    return

                date_ts = pd.Timestamp(date, tz="UTC")
                start_date = date - datetime.timedelta(days=lookback_days - 1)
                if start_date < data_min:
                    print(
                        f"Lookback von {lookback_days} Tagen reicht vor {data_min} zurück "
                        "— kürzeren Lookback oder späteres Datum wählen."
                    )
                    return

                basket_df_at_date = HelperFunctions.build_volume_baskets(
                    rolling_volume_df, date_ts,
                    id_column=id_column, date_column=date_column,
                    basket_upper_bounds=basket_upper_bounds,
                )
                basket_asset_ids = set(
                    basket_df_at_date.loc[basket_df_at_date["basket"] == label, id_column]
                )
                if len(basket_asset_ids) < 2:
                    print(f"{label}: weniger als 2 Assets an diesem Datum — keine Cluster-Analyse möglich.")
                    return

                basket_exclude_ids = excluded_ids | (all_asset_ids - basket_asset_ids)
                result = point_cloud_fn(
                    ohlcv_df=ohlcv_df, universe_df=universe_df, end_date=date_ts,
                    window_size_days=lookback_days, exclude_asset_ids=basket_exclude_ids,
                )
                if result is None:
                    print(f"{label}: zu wenig gemeinsame Preis-Historie für dieses Fenster.")
                    return

                aids = result["asset_ids"]
                syms = [id_to_symbol.get(i, f"ID:{i}") for i in aids]
                returns_df = pd.DataFrame(result["point_cloud"], columns=syms, index=result["timestamps"])
                dist_matrix = HelperFunctions.correlation_distance(returns_df)
                corr = np.corrcoef(returns_df.to_numpy().T)

                clusters, z = HelperFunctions.get_persistent_clusters(
                    dist_matrix, aids,
                    persistence_threshold=persistence_threshold,
                    epsilon_cap=epsilon_cap,
                )

                # Alle vier Plots als EIN Superplot (2x2-Grid, ein einzelnes
                # plt.show()/plt.close()) statt zwei getrennter Figuren:
                # oben Dendrogramm + Barcode, darunter Korrelations- und
                # Distanz-Heatmap. Ein gemeinsames plt.show() garantiert
                # identisches, bereits bewährtes Resize-Verhalten (horizontal
                # UND vertikal) für alle vier Panels gleichzeitig -- eine
                # separate Plotly-Figur reagierte in manchen Notebook-
                # Frontends nicht zuverlässig auf Grössenänderungen des
                # Containers, daher Matplotlib für den Barcode (Preis: keine
                # interaktiven Hover-Tooltips mehr). Distanz-Skala fix
                # auf [0, 2] (theoretisches Maximum, siehe correlation_distance:
                # d=sqrt(2*(1-rho)), rho in [-1,1] -> d in [0,2]) statt sqrt(2)
                # ("keine Korrelation") -- sonst werden alle (leicht) anti-
                # korrelierten Paare oberhalb sqrt(2) auf dieselbe Randfarbe
                # geklemmt und sind nicht mehr unterscheidbar.
                super_fig, ((dendro_ax, barcode_ax), (corr_ax, dist_ax)) = plt.subplots(
                    2, 2, figsize=(max(14, len(syms) * 0.6), max(12, len(syms) * 0.55))
                )
                # y_max=2.0: siehe Distanz-Heatmap unten -- feste Achse, alle
                # Assets/Merges bleiben sichtbar statt an sqrt(2) "abgeschnitten".
                plot_dendrogram_with_clusters(
                    dist_matrix, syms, clusters, z=z,
                    persistence_threshold=persistence_threshold, epsilon_cap=epsilon_cap,
                    ax=dendro_ax, show=False, y_max=2.0,
                )
                plot_h0_barcode_on_ax(dist_matrix, syms, barcode_ax, x_max=2.0)
                plot_correlation_heatmap(
                    corr, syms, ax=corr_ax, show=False,
                    title=f"{label}: Korrelationsmatrix ({lookback_days}-Tage-Fenster · {result['n_points']} Punkte)",
                )
                plot_correlation_heatmap(
                    dist_matrix, syms, ax=dist_ax, show=False,
                    value_range=(0, 2.0), cmap="BuPu_r",
                    colorbar_label="Korrelationsdistanz (epsilon)",
                    title=f"{label}: Distanz-Heatmap",
                )
                super_fig.tight_layout()
                with dendro_out:
                    plt.show()
                    plt.close(super_fig)

        for w in (lookback_slider, date_picker, persistence_slider, epsilon_cap_slider):
            w.observe(render, names="value")

        controls = widgets.VBox([
            widgets.HBox([lookback_slider, lookback_minus, lookback_plus]),
            widgets.HBox([date_picker, date_minus, date_plus]),
            widgets.HBox([persistence_slider, epsilon_cap_slider]),
        ])

        render()
        return widgets.VBox([controls, out, dendro_out])

    tab = widgets.Tab()
    tab.children = [_build_tab(label) for label in basket_labels]
    for i, label in enumerate(basket_labels):
        tab.set_title(i, label)

    display(tab)
    return tab