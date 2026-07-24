import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
import math

# Default-Renderer "browser" öffnet einen System-Browser-Prozess und blockiert/crasht
# den Kernel bei headless nbconvert-Ausführung -> stattdessen inline im Notebook rendern.
if pio.renderers.default == "browser":
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

# Kategoriale Referenzpalette (dataviz-Skill), feste Reihenfolge, nie zyklisch neu
# zugewiesen. Slot 0 ist reserviert fuer "noch nicht verschmolzen / Singleton" (grau).
_CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
_SINGLETON_GREY = "#c3c2b7"
_INK_MUTED = "#898781"
_INK_SECONDARY = "#52514e"


def plot_anchor_vs_others_plotly(returns_window, anchor):
    others = [c for c in returns_window.columns if c != anchor]
    n_cols = 5
    n_rows = math.ceil(len(others) / n_cols)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=others)

    for i, other in enumerate(others):
        row, col = divmod(i, n_cols)
        corr = returns_window[anchor].corr(returns_window[other])
        fig.add_trace(
            go.Scatter(
                x=returns_window[anchor], y=returns_window[other],
                mode="markers", marker=dict(size=4),
                name=f"{other} (r={corr:.2f})",
            ),
            row=row + 1, col=col + 1,
        )

    fig.update_layout(height=250 * n_rows, showlegend=False, title=f"{anchor} vs. alle Assets im Window")
    fig.show()


def plot_point_cloud_3d_interactive(point_cloud, asset_symbols, timestamps=None, color_by_x=True):
    """Interaktiver 3D-Scatter der Point Cloud mit frei wählbaren Achsen-Assets.

    Drei Dropdowns wählen, welches Asset auf X/Y/Z abgebildet wird, sodass man
    durch die n-dimensionale Wolke "wandern" kann, indem man die Achsen-
    Zuordnung wechselt. Plotly liefert Rotieren/Zoomen/Pan nativ im Notebook.

    Args:
        point_cloud: ndarray der Form (n_points, n_assets), z.B. result["point_cloud"].
        asset_symbols: Liste der Asset-Symbole in Spaltenreihenfolge (Länge n_assets).
        timestamps: Optionaler Zeitindex (Länge n_points) für Hover-Text.
        color_by_x: Falls True, werden Punkte nach ihrem Wert auf der X-Achse
            eingefärbt (diverging blue<->red, zentriert auf 0). Sonst einheitliche Farbe.
    """
    n_points, n_dims = point_cloud.shape
    if len(asset_symbols) != n_dims:
        raise ValueError(f"asset_symbols hat {len(asset_symbols)} Einträge, point_cloud aber {n_dims} Dimensionen.")

    hover_ts = None
    if timestamps is not None:
        hover_ts = [str(t) for t in timestamps]

    def render(x_asset, y_asset, z_asset):
        xi, yi, zi = asset_symbols.index(x_asset), asset_symbols.index(y_asset), asset_symbols.index(z_asset)
        x, y, z = point_cloud[:, xi], point_cloud[:, yi], point_cloud[:, zi]

        if color_by_x:
            bound = float(np.max(np.abs(x))) or 1.0
            marker = dict(
                size=3, color=x, colorscale=_DIVERGING_SCALE,
                cmin=-bound, cmax=bound,
                colorbar=dict(title=f"{x_asset}<br>Return"),
                opacity=0.6,
            )
        else:
            marker = dict(size=3, color=_MARKER_COLOR, opacity=0.6)

        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=marker,
            text=hover_ts,
            hovertemplate=(
                f"{x_asset}: %{{x:.4%}}<br>{y_asset}: %{{y:.4%}}<br>{z_asset}: %{{z:.4%}}"
                + ("<br>%{text}" if hover_ts is not None else "")
                + "<extra></extra>"
            ),
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title=f"{x_asset} Return",
                yaxis_title=f"{y_asset} Return",
                zaxis_title=f"{z_asset} Return",
            ),
            title=f"3D Point Cloud: {x_asset} / {y_asset} / {z_asset}  ({n_points} Punkte)",
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.show()

    default_x = asset_symbols[0]
    default_y = asset_symbols[1] if n_dims > 1 else asset_symbols[0]
    default_z = asset_symbols[2] if n_dims > 2 else asset_symbols[0]

    widgets.interact(
        render,
        x_asset=widgets.Dropdown(options=asset_symbols, value=default_x, description="X-Achse:"),
        y_asset=widgets.Dropdown(options=asset_symbols, value=default_y, description="Y-Achse:"),
        z_asset=widgets.Dropdown(options=asset_symbols, value=default_z, description="Z-Achse:"),
    )


def plot_vietoris_rips_epsilon_slider(distance_matrix, asset_symbols, max_edge_length=None, min_cluster_size=2):
    """Interaktiver Vietoris-Rips-Graph: Assets als Knoten, Kanten bei Distanz <= epsilon.

    Zeigt direkt das Geburts-/Verschmelzungsverhalten der H0-Features:
    bei epsilon=0 ist jedes Asset isoliert (n Cluster), mit wachsendem epsilon verschmelzen
    Cliquen, bis am Ende alles ein einziger Cluster ist. Die Knotenposition kommt aus MDS
    auf der Korrelationsdistanzmatrix (Assets mit aehnlichem Return-Profil liegen nah beieinander),
    die Kanten sind der eigentliche Vietoris-Rips-1-Skelett bei der gewaehlten Schwelle.

    Args:
        distance_matrix: (n_assets, n_assets) symmetrische Korrelationsdistanzmatrix,
            z.B. helper.correlation_distance(returns_df). Diagonale = 0.
        asset_symbols: Liste der Asset-Symbole, Reihenfolge passend zu distance_matrix.
        max_edge_length: Oberes Ende des epsilon-Sliders. Default: max(distance_matrix).
        min_cluster_size: Cluster unterhalb dieser Groesse werden als Singletons (grau)
            statt mit einer Cluster-Farbe dargestellt.
    """
    n = len(asset_symbols)
    if distance_matrix.shape != (n, n):
        raise ValueError(f"distance_matrix hat Shape {distance_matrix.shape}, erwartet ({n}, {n}).")

    dist_max = float(np.max(distance_matrix))
    eps_max = max_edge_length if max_edge_length is not None else dist_max

    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress=False).fit_transform(distance_matrix)
    z = linkage(squareform(distance_matrix, checks=False), method="single")

    def render(epsilon):
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
            line=dict(width=1, color=_INK_MUTED), opacity=0.5,
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1], mode="markers+text",
            marker=dict(size=16, color=node_colors, line=dict(width=1, color="white")),
            text=asset_symbols, textposition="top center",
            textfont=dict(size=10, color=_INK_SECONDARY),
            hovertext=[f"{s}<br>Cluster {lbl} ({sizes[lbl]} Assets)" for s, lbl in zip(asset_symbols, labels)],
            hoverinfo="text", showlegend=False,
        ))

        n_clusters = len(sizes)
        n_big = sum(1 for lbl in sizes if sizes[lbl] >= min_cluster_size)
        fig.update_layout(
            title=(
                f"Vietoris-Rips 1-Skelett bei epsilon={epsilon:.4f}  |  "
                f"{n_clusters} Cluster ({n_big} mit >= {min_cluster_size} Assets)"
            ),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=650, margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="#fcfcfb",
        )
        fig.show()

    widgets.interact(
        render,
        epsilon=widgets.FloatSlider(
            min=0.0, max=eps_max, step=eps_max / 200,
            value=0.0, description="epsilon:", continuous_update=False,
            readout_format=".4f", style={"description_width": "initial"},
            layout=widgets.Layout(width="600px"),
        ),
    )


def plot_dendrogram(distance_matrix, asset_symbols, color_threshold=None, figsize=None):
    """Single-Linkage-Dendrogramm der Assets, optional mit Schwellenlinie.

    Baut den Cluster-Baum aus der Korrelationsdistanzmatrix auf: jedes Asset
    startet als eigenes Blatt, Verschmelzungen erfolgen entlang der y-Achse
    (Korrelationsdistanz). Ein niedriger Merge-Punkt bedeutet, dass zwei
    Assets/Cliquen bereits bei kleiner Distanz zusammengehoeren, ein hoher
    Merge-Punkt zeigt eine spaete, schwaechere Verbindung.

    Args:
        distance_matrix: (n_assets, n_assets) symmetrische Korrelationsdistanzmatrix.
        asset_symbols: Liste der Asset-Symbole, Reihenfolge passend zu distance_matrix.
        color_threshold: Falls gesetzt, wird bei dieser Distanz eine horizontale
            Schwellenlinie eingezeichnet und Aeste unterhalb der Schwelle farbig
            (statt grau) dargestellt, analog zu scipy's eigenem color_threshold.
        figsize: Optionale (Breite, Hoehe) in Zoll. Default skaliert mit der Anzahl Assets.
    """
    n = len(asset_symbols)
    if distance_matrix.shape != (n, n):
        raise ValueError(f"distance_matrix hat Shape {distance_matrix.shape}, erwartet ({n}, {n}).")

    z = linkage(squareform(distance_matrix, checks=False), method="single")

    if figsize is None:
        figsize = (max(10, n * 0.35), 6)

    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(
        z, labels=asset_symbols, ax=ax,
        color_threshold=color_threshold,
        above_threshold_color=_INK_MUTED,
    )
    if color_threshold is not None:
        ax.axhline(y=color_threshold, color=_CATEGORICAL[7], linestyle="--", linewidth=1.2,
                   label=f"Merge Scale epsilon* = {color_threshold:.4f}")
        ax.legend()

    ax.set_ylabel("Korrelationsdistanz (epsilon)")
    ax.set_title(f"Single-Linkage-Dendrogramm ({n} Assets)")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.show()
    return z


def plot_h0_persistence_barcode(distance_matrix, asset_symbols):
    """Cluster-Lebensdauer-Barcode: ein Balken pro Cluster von birth bis death=epsilon.

    Jede Zeile der linkage-Matrix ist ein Merge-Event: zwei Cluster verschmelzen bei
    epsilon=dist zu einem neuen, groesseren Cluster. Dieser neue Cluster wird an genau
    dieser Distanz "geboren" (birth=dist des Merges, der ihn erzeugt hat) und "stirbt"
    wieder, wenn er seinerseits mit einem anderen Cluster verschmilzt (death=dist des
    naechsten Merges, an dem er beteiligt ist). Die urspruenglichen Einzel-Assets (die
    Blaetter des Baums) existieren dagegen von Anfang an, ihr birth ist daher 0.
    Balkenlaenge = Persistenz = wie lange dieser konkrete Cluster als eigenstaendige
    Einheit bestand, bevor er in einen groesseren Cluster aufging. Lange Balken mit
    vielen beteiligten Assets sind die strukturell stabilen, interessanten Cliquen.

    Args:
        distance_matrix: (n_assets, n_assets) symmetrische Korrelationsdistanzmatrix.
        asset_symbols: Liste der Asset-Symbole, Reihenfolge passend zu distance_matrix.
    """
    n = len(asset_symbols)
    if distance_matrix.shape != (n, n):
        raise ValueError(f"distance_matrix hat Shape {distance_matrix.shape}, erwartet ({n}, {n}).")

    z = linkage(squareform(distance_matrix, checks=False), method="single")

    # Fuer jeden Cluster (0..n-1 = urspruengliche Assets, n.. = durch Merges
    # entstandene Cluster) merken wir uns Asset-Mitglieder UND birth-Distanz.
    # Die Original-Assets existieren von Anfang an -> birth = 0.0.
    members = {i: [asset_symbols[i]] for i in range(n)}
    birth = {i: 0.0 for i in range(n)}

    rows = []
    for row_idx, (c1, c2, dist, _count) in enumerate(z):
        c1, c2 = int(c1), int(c2)
        merged = members[c1] + members[c2]
        cluster_id = n + row_idx
        members[cluster_id] = merged
        birth[cluster_id] = float(dist)

        # c1 und c2 sterben genau jetzt (sie verschmelzen zu cluster_id) -> ihr
        # eigener Lebensdauer-Balken laeuft von ihrem eigenen birth bis hier.
        for c in (c1, c2):
            rows.append({
                "birth": birth[c], "death": float(dist),
                "size": len(members[c]), "assets": members[c], "row": row_idx,
            })

    rows.sort(key=lambda r: r["death"])

    fig = go.Figure()
    bound = max((r["death"] for r in rows), default=1.0) or 1.0
    for y, r in enumerate(rows):
        color = _CATEGORICAL[min(r["size"] - 2, len(_CATEGORICAL) - 1)] if r["size"] >= 2 else _SINGLETON_GREY
        fig.add_trace(go.Scatter(
            x=[r["birth"], r["death"]], y=[y, y], mode="lines",
            line=dict(width=8, color=color),
            hovertemplate=(
                f"Birth: {r['birth']:.4f}<br>Death: {r['death']:.4f}<br>"
                f"Persistenz: {r['death'] - r['birth']:.4f}<br>Groesse: {r['size']} Assets<br>"
                f"Assets: {', '.join(r['assets'][:12])}{'...' if len(r['assets']) > 12 else ''}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title=f"Cluster-Lebensdauer-Barcode ({len(rows)} Cluster-Intervalle)",
        xaxis=dict(title="Korrelationsdistanz (epsilon)", range=[0, bound * 1.05]),
        yaxis=dict(title="Cluster-Intervall (sortiert nach Todeszeitpunkt)", showticklabels=False),
        height=max(400, 18 * len(rows)),
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="#fcfcfb",
    )
    fig.show()
    print(f"Anzahl Cluster-Intervalle: {len(rows)}")    
    return z