import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from plotly.colors import get_colorscale
from scipy.interpolate import splev, splprep
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import triangulate
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from shapash.utils.utils import adjust_title_height

logger = logging.getLogger(__name__)


def compute_tsne_projection(
    values_to_project: pd.DataFrame,
    random_state: int = 79,
    perplexity: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 2D TSNE projection of high-dimensional data.

    Parameters
    ----------
    values_to_project : pd.DataFrame
        DataFrame containing the high-dimensional data to be projected.
    random_state : int, default=79
        Random seed for reproducibility of the TSNE projection.
    perplexity : int or None, default=None
        Perplexity parameter for TSNE. If None, it is automatically set to
        ``min(30, max(2, n_samples // 3))`` to match the original logic.

    Returns
    -------
    x : np.ndarray
        1D array containing the x coordinates of the TSNE projection.
    y : np.ndarray
        1D array containing the y coordinates of the TSNE projection.
    """
    n_samples = values_to_project.shape[0]
    if perplexity is None:
        perplexity = min(30, max(2, n_samples // 3))

    projections = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="random",
        random_state=random_state,
    ).fit_transform(values_to_project)

    return projections


def build_tsne_title(
    title: Optional[str],
    subtitle: Optional[str],
    addnote: Optional[str],
    *,
    style_dict: dict,
    height: int,
) -> dict:
    """
    Build the Plotly title dictionary for a TSNE projection plot.

    Parameters
    ----------
    title : str or None
        Main title of the plot. If None or empty, a default title is used.
    subtitle : str or None
        Subtitle displayed below the main title.
    addnote : str or None
        Additional note displayed with the subtitle.
    style_dict : dict
        Style dictionary containing the ``dict_title`` base configuration.
    height : int
        Figure height, used to adjust the vertical position of the title.

    Returns
    -------
    dict
        Plotly title dictionary (text + positioning).
    """
    if not title:
        title = "TSNE Projection Plot"

    if subtitle and addnote:
        title += "<br><sup>" + subtitle + " - " + addnote + "</sup>"
    elif subtitle:
        title += "<br><sup>" + subtitle + "</sup>"
    elif addnote:
        title += "<br><sup>" + addnote + "</sup>"

    dict_title = style_dict["dict_title"] | {
        "text": title,
        "y": adjust_title_height(height),
    }

    return dict_title


def compute_kmeans_labels(
    X: np.ndarray,
    n_clusters: int = 9,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute KMeans clustering on 2D points.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        Input points.
    n_clusters : int, default=9
        Number of clusters.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    labels : np.ndarray
        Cluster label for each point.
    centers : np.ndarray
        Cluster centers.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_


def move_points_towards_centroid(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    factor: float = 1.0,
) -> np.ndarray:
    """
    Move points towards their cluster centroid.

    Parameters
    ----------
    X : np.ndarray
        Original points.
    labels : np.ndarray
        Cluster labels.
    centers : np.ndarray
        Cluster centroids.
    factor : float, default=1.0
        Contraction factor (1.0 = no contraction,
        <1 pulls points towards centroid).

    Returns
    -------
    np.ndarray
        Moved points.
    """
    X_moved = np.zeros_like(X)

    for i in range(len(X)):
        c = centers[labels[i]]
        p = X[i]
        X_moved[i] = c + factor * (p - c)

    return X_moved


def compute_concave_hull(
    points: np.ndarray,
    alpha: float = 5.0,
) -> Polygon:
    """
    Compute a concave hull (alpha shape) from points, with robust fallbacks.

    This function NEVER returns None.
    If concave hull computation fails, it falls back to:
    1) convex hull
    2) minimal enclosing circle approximation

    Parameters
    ----------
    points : np.ndarray of shape (n_points, 2)
        Input points.
    alpha : float, default=5.0
        Alpha parameter controlling concavity.

    Returns
    -------
    Polygon
        A valid polygon representing the cluster shape.
    """
    pts = MultiPoint(points)

    # --- Attempt concave hull ---
    try:
        triangles = triangulate(pts)
        filtered = [tri for tri in triangles if tri.area < alpha]

        if filtered:
            merged = filtered[0]
            for tri in filtered[1:]:
                merged = merged.union(tri)

            if merged.geom_type == "Polygon" and merged.area > 0:
                return Polygon(merged.exterior.coords)

    except Exception as exc:
        logger.debug(
            "Concave hull computation failed, falling back to convex hull.",
            exc_info=exc,
        )

    # --- Fallback 1: convex hull ---
    try:
        hull = pts.convex_hull
        if hull.geom_type == "Polygon" and hull.area > 0:
            return hull

    except Exception as exc:
        logger.debug(
            "Convex hull computation failed, falling back to circle approximation.",
            exc_info=exc,
        )

    # --- Fallback 2: minimal circle-like polygon ---
    center = points.mean(axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))

    angles = np.linspace(0, 2 * np.pi, 32)
    circle = np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        ]
    )

    return Polygon(circle)


def smooth_polygon_contour(
    poly: Polygon,
    smoothing: float = 0.02,
    nb_points: int = 400,
) -> Polygon:
    """
    Smooth polygon contour using a periodic spline.

    This function is designed to be fail-safe: if spline fitting fails due to
    degenerate inputs (duplicate points, insufficient distinct points, etc.),
    it returns the original polygon.

    Parameters
    ----------
    poly : Polygon
        Input polygon.
    smoothing : float, default=0.02
        Spline smoothing factor.
    nb_points : int, default=400
        Number of points in smoothed contour.

    Returns
    -------
    Polygon
        Smoothed polygon. If smoothing fails, returns the original polygon.
    """
    if poly is None:
        raise ValueError("Cannot smooth a None polygon.")
    if not isinstance(poly, Polygon):
        raise TypeError(f"Expected Polygon, got {type(poly)}")

    x, y = poly.exterior.xy
    pts = np.column_stack([x, y])

    # Ensure closed contour
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    # Remove consecutive duplicates (common after unions/scaling)
    diffs = np.diff(pts, axis=0)
    keep = np.any(np.abs(diffs) > 1e-12, axis=1)
    pts = np.vstack([pts[0], pts[1:][keep]])

    # Ensure closure again
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    # Need enough distinct points for k=3 spline; be conservative
    unique_pts = np.unique(np.round(pts, decimals=12), axis=0)
    if unique_pts.shape[0] < 8:
        return poly

    # splprep requires m > k (k=3 by default), so m >= 4 (we already enforce >= 8)
    try:
        tck, _ = splprep(
            [pts[:, 0], pts[:, 1]],
            s=smoothing,
            per=True,
        )
        u = np.linspace(0, 1, nb_points)
        xs, ys = splev(u, tck)
        smoothed = Polygon(np.column_stack([xs, ys]))

        # Sometimes spline can produce invalid self-intersections; keep it safe
        if not smoothed.is_valid or smoothed.area <= 0:
            return poly

        return smoothed

    except Exception:
        # Fail-safe fallback
        return poly


def _scale_polygon(
    poly: Polygon,
    center: np.ndarray,
    scale: float,
) -> Polygon:
    """
    Scale a polygon around a given center.

    Parameters
    ----------
    poly : Polygon
        Polygon to scale.
    center : np.ndarray
        Scaling center.
    scale : float
        Scale factor.

    Returns
    -------
    Polygon
        Scaled polygon.
    """
    pts = np.array(poly.exterior.coords)
    scaled = center + scale * (pts - center)
    return Polygon(scaled)


def expand_polygons_independently(
    polys: list[Polygon],
    centers: list[np.ndarray],
    eps: float = 0.02,
    max_iter: int = 100,
) -> tuple[list[Polygon], np.ndarray]:
    """
    Expand each polygon independently until collision.

    Parameters
    ----------
    polys : list of Polygon
        Initial polygons.
    centers : list of np.ndarray
        Polygon centers.
    eps : float, default=0.02
        Incremental scale step.
    max_iter : int, default=100
        Maximum number of iterations.

    Returns
    -------
    final_polys : list of Polygon
        Expanded polygons.
    scales : np.ndarray
        Final scale for each polygon.
    """
    n = len(polys)
    scales = np.ones(n)
    curr_polys = polys.copy()
    active = [True] * n

    for _ in range(max_iter):
        any_growth = False

        for i in range(n):
            if not active[i]:
                continue

            candidate = _scale_polygon(polys[i], centers[i], scales[i] + eps)

            collision = any(candidate.intersects(curr_polys[j]) for j in range(n) if j != i)

            if collision:
                active[i] = False
            else:
                scales[i] += eps
                curr_polys[i] = candidate
                any_growth = True

        if not any_growth:
            break

    return curr_polys, scales


def scale_points_within_cluster(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    label_to_index: dict[int, int],
) -> np.ndarray:
    """
    Scale points so they fill the expanded cluster polygons.

    Parameters
    ----------
    X : np.ndarray
        Original points.
    labels : np.ndarray
        Cluster labels.
    centers : np.ndarray
        Cluster centers.
    scales : np.ndarray
        Final scale per cluster.
    label_to_index : dict
        Mapping from original cluster label -> index in centers/scales.

    Returns
    -------
    np.ndarray
        Scaled points.
    """
    X_scaled = X.copy()

    for i in range(len(X)):
        lab = labels[i]

        if lab not in label_to_index:
            continue  # cluster without valid patate

        idx = label_to_index[lab]
        c = centers[idx]
        X_scaled[i] = c + scales[idx] * (X[i] - c)

    return X_scaled


def _resolve_colorscale(colorscale):
    if isinstance(colorscale, str):
        cs = get_colorscale(colorscale)
    else:
        cs = colorscale

    # cs = [[pos, "rgb(...)"], ...]
    colors = []
    for _, c in cs:
        nums = list(map(int, re.findall(r"\d+", c)))
        colors.append(np.array(nums[:3], dtype=float))
    return colors


def _interpolate_color(colors, t):
    n = len(colors)
    if n == 1:
        return colors[0]

    pos = t * (n - 1)
    i0 = int(np.floor(pos))
    i1 = min(i0 + 1, n - 1)

    w = pos - i0
    return (1 - w) * colors[i0] + w * colors[i1]


def value_to_rgba(value, colorscale, cmin, cmax, alpha=0.3):
    """
    Map a scalar value to an RGBA color string using a continuous colorscale.

    The input value is normalized between ``cmin`` and ``cmax``, clipped to
    the [0, 1] interval, and used to interpolate a color from the provided
    colorscale. The resulting color is returned as an RGBA string, with a
    configurable alpha channel.

    Parameters
    ----------
    value : float
        Scalar value to map to a color.
    colorscale : str or list
        Colorscale definition. Can be a named colorscale or an explicit list
        of colors used for interpolation.
    cmin : float
        Minimum value of the normalization range.
    cmax : float
        Maximum value of the normalization range.
    alpha : float, default=0.3
        Opacity of the resulting color, between 0 (fully transparent) and
        1 (fully opaque).

    Returns
    -------
    str
        RGBA color string formatted as ``"rgba(r,g,b,a)"``, where ``r``,
        ``g``, and ``b`` are integers in [0, 255].

    Notes
    -----
    If ``cmin`` equals ``cmax``, the value is mapped to the midpoint of the
    colorscale.
    """
    if cmax == cmin:
        t = 0.5
    else:
        t = (value - cmin) / (cmax - cmin)

    t = float(np.clip(t, 0, 1))

    colors = _resolve_colorscale(colorscale)
    rgb = _interpolate_color(colors, t)

    r, g, b = rgb.astype(int)
    return f"rgba({r},{g},{b},{alpha})"


def encode_color_value(color_value):
    """
    Encode color_value for plotting.
    - Numeric: returned as-is
    - Categorical: label-encoded with mapping

    Returns
    -------
    encoded_values : pd.Series
    is_categorical : bool
    label_mapping : dict or None
    """

    is_categorical = color_value.dtype == "object" or color_value.dtype.name == "category"

    if not is_categorical:
        return color_value.astype(float), False, None

    le = LabelEncoder()
    encoded = le.fit_transform(color_value.astype(str))

    label_mapping = dict(enumerate(le.classes_))

    return pd.Series(encoded, index=color_value.index), True, label_mapping
