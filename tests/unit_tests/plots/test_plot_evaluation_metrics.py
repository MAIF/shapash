import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import shapash.plots.plot_evaluation_metrics as evaluation_metrics
from shapash.style.style_utils import define_style, get_palette


def test_plot_clustering_by_explainability_draws_clusters_when_enabled(monkeypatch):
    projections = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    values_to_project = pd.DataFrame(index=range(len(projections)))
    color_value = [pd.DataFrame({"color": [0.0, 0.25, 0.75, 1.0]})]

    monkeypatch.setattr(evaluation_metrics, "move_points_towards_centroid", lambda points, *_args, **_kwargs: points)
    monkeypatch.setattr(evaluation_metrics, "compute_concave_hull", lambda *_args, **_kwargs: polygon)
    monkeypatch.setattr(evaluation_metrics, "smooth_polygon_contour", lambda shape: shape)
    monkeypatch.setattr(
        evaluation_metrics,
        "expand_polygons_independently",
        lambda polygons, *_args, **_kwargs: (polygons, np.ones(len(polygons))),
    )
    monkeypatch.setattr(evaluation_metrics, "scale_points_within_cluster", lambda points, *_args: points)

    fig = evaluation_metrics.plot_clustering_by_explainability(
        values_to_project=values_to_project,
        hv_text={"clusters": ["cluster detail"], "points": [["point"] * len(projections)]},
        color_value=color_value,
        projections=projections,
        labels=np.zeros(len(projections), dtype=int),
        centers=np.array([[0.5, 0.5]]),
        style_dict=define_style(get_palette("default")),
        show_clusters=True,
    )

    assert len(fig.data) == 2
    assert fig.data[0].fill == "toself"
    assert fig.data[0].name == "Cluster 0"
    assert fig.data[1].mode == "markers"


def test_prediction_regression_plot_trims_outliers_above_500_rows():
    rng = np.random.default_rng(0)
    index = pd.RangeIndex(501)
    y_target = pd.DataFrame({"target": 10 + rng.normal(size=len(index))}, index=index)
    y_pred = pd.DataFrame({"target": y_target["target"] + rng.normal(0, 0.1, size=len(index))}, index=index)
    prediction_error = ((y_target - y_pred).abs() / y_target).rename(columns={"target": "error"})

    fig, _ = evaluation_metrics._prediction_regression_plot(
        y_target=y_target,
        y_pred=y_pred,
        prediction_error=prediction_error,
        list_ind=index.tolist(),
        style_dict=define_style(get_palette("default")),
        round_digit=2,
    )

    assert len(fig.data[1].x) < len(index)
