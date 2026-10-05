import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

import shapash.utils.clustering as clustering


@pytest.mark.parametrize("perplexity, expected", [(None, 4), (2, 2)])
def test_compute_tsne_projection_sets_perplexity(monkeypatch, perplexity, expected):
    captured = {}

    class FakeTSNE:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit_transform(self, values):
            return np.zeros((len(values), 2))

    monkeypatch.setattr(clustering, "TSNE", FakeTSNE)
    values = pd.DataFrame(np.zeros((12, 3)))

    projection = clustering.compute_tsne_projection(values, perplexity=perplexity)

    assert projection.shape == (12, 2)
    assert captured["perplexity"] == expected
    assert captured["random_state"] == 79


@pytest.mark.parametrize(
    "title, subtitle, addnote, expected",
    [
        (None, None, None, "TSNE Projection Plot"),
        ("Projection", "Dataset", "PCA", "Projection<br><sup>Dataset - PCA</sup>"),
        ("Projection", "Dataset", None, "Projection<br><sup>Dataset</sup>"),
        ("Projection", None, "PCA", "Projection<br><sup>PCA</sup>"),
    ],
)
def test_build_tsne_title_combinations(title, subtitle, addnote, expected):
    result = clustering.build_tsne_title(
        title,
        subtitle,
        addnote,
        style_dict={"dict_title": {"font": {"size": 12}}},
        height=500,
    )

    assert result["text"] == expected
    assert result["font"] == {"size": 12}
    assert "y" in result


def test_compute_kmeans_labels_returns_labels_and_centers():
    points = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])

    labels, centers = clustering.compute_kmeans_labels(points, n_clusters=2)

    assert labels.shape == (4,)
    assert centers.shape == (2, 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_move_points_towards_centroid_respects_factor():
    points = np.array([[0.0, 0.0], [4.0, 2.0]])
    labels = np.array([0, 0])
    centers = np.array([[2.0, 1.0]])

    assert np.array_equal(clustering.move_points_towards_centroid(points, labels, centers, factor=1), points)
    assert np.array_equal(clustering.move_points_towards_centroid(points, labels, centers, factor=0), centers[[0, 0]])


def test_compute_concave_hull_uses_concave_and_convex_hulls():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])

    concave = clustering.compute_concave_hull(points, alpha=10)
    convex = clustering.compute_concave_hull(points, alpha=0)

    assert isinstance(concave, Polygon)
    assert isinstance(convex, Polygon)
    assert concave.area > 0
    assert convex.area == pytest.approx(4)


def test_compute_concave_hull_falls_back_when_filtered_triangles_are_disconnected(monkeypatch):
    triangles = [
        Polygon([(0, 0), (0.5, 0), (0, 0.5)]),
        Polygon([(0.5, 0.5), (1, 0.5), (1, 1)]),
    ]
    monkeypatch.setattr(clustering, "triangulate", lambda _points: triangles)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    result = clustering.compute_concave_hull(points, alpha=10)

    assert result.geom_type == "Polygon"
    assert result.area == pytest.approx(1)


def test_compute_concave_hull_uses_circle_for_degenerate_convex_hull():
    result = clustering.compute_concave_hull(np.array([[0.0, 0.0], [2.0, 0.0]]))

    assert isinstance(result, Polygon)
    assert result.area > 0


def test_compute_concave_hull_falls_back_to_circle_when_hull_fails(monkeypatch):
    class BrokenMultiPoint:
        @property
        def convex_hull(self):
            raise RuntimeError("hull unavailable")

    def fail_triangulation(_points):
        raise RuntimeError("triangulation unavailable")

    monkeypatch.setattr(clustering, "MultiPoint", lambda _points: BrokenMultiPoint())
    monkeypatch.setattr(clustering, "triangulate", fail_triangulation)
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])

    result = clustering.compute_concave_hull(points)

    assert isinstance(result, Polygon)
    assert result.area > 0


def test_smooth_polygon_contour_validates_and_smooths():
    with pytest.raises(ValueError, match="Cannot smooth a None polygon"):
        clustering.smooth_polygon_contour(None)
    with pytest.raises(TypeError, match="Expected Polygon"):
        clustering.smooth_polygon_contour([(0, 0), (1, 0), (0, 1)])

    small_polygon = Polygon([(0, 0), (1, 0), (0, 1)])
    assert clustering.smooth_polygon_contour(small_polygon) is small_polygon

    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    polygon = Polygon(np.column_stack([np.cos(angles), np.sin(angles)]))
    result = clustering.smooth_polygon_contour(polygon, nb_points=50)
    assert isinstance(result, Polygon)
    assert result.is_valid


def test_smooth_polygon_contour_returns_input_when_spline_fails(monkeypatch):
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    polygon = Polygon(np.column_stack([np.cos(angles), np.sin(angles)]))

    def fail_spline(*_args, **_kwargs):
        raise ValueError("spline failed")

    monkeypatch.setattr(clustering, "splprep", fail_spline)

    assert clustering.smooth_polygon_contour(polygon) is polygon


def test_smooth_polygon_contour_returns_input_for_invalid_spline_result(monkeypatch):
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    polygon = Polygon(np.column_stack([np.cos(angles), np.sin(angles)]))
    monkeypatch.setattr(clustering, "splprep", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(clustering, "splev", lambda *_args, **_kwargs: ([0, 1, 0, 1, 0], [0, 1, 1, 0, 0]))

    assert clustering.smooth_polygon_contour(polygon) is polygon


def test_scale_and_expand_polygons_stop_on_collision():
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    center = np.array([0.5, 0.5])
    scaled = clustering._scale_polygon(polygon, center, 2)
    assert scaled.area == pytest.approx(4)

    polygons = [polygon, Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])]
    centers = [center, np.array([2.5, 0.5])]
    expanded, scales = clustering.expand_polygons_independently(polygons, centers, eps=1, max_iter=5)

    assert scales.tolist() == [2, 1]
    assert expanded[0].area == pytest.approx(4)
    assert expanded[1].equals(polygons[1])


def test_expand_polygons_independently_grows_isolated_polygon_and_handles_empty_input():
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    expanded, scales = clustering.expand_polygons_independently(
        [polygon], [np.array([0.5, 0.5])], eps=0.1, max_iter=2
    )
    assert scales[0] == pytest.approx(1.2)
    assert expanded[0].area > polygon.area

    empty_polygons, empty_scales = clustering.expand_polygons_independently([], [])
    assert empty_polygons == []
    assert empty_scales.size == 0


def test_scale_points_within_cluster_skips_unmapped_labels():
    points = np.array([[1.0, 1.0], [3.0, 1.0]])
    result = clustering.scale_points_within_cluster(
        points,
        labels=np.array([5, 9]),
        centers=np.array([[0.0, 0.0]]),
        scales=np.array([2.0]),
        label_to_index={5: 0},
    )

    assert np.array_equal(result, [[2.0, 2.0], [3.0, 1.0]])


def test_value_to_rgba_interpolates_and_clips_colors():
    colorscale = [[0, "rgb(0, 10, 20)"], [1, "rgb(100, 110, 120)"]]

    assert clustering.value_to_rgba(0.5, colorscale, 0, 1) == "rgba(50,60,70,0.3)"
    assert clustering.value_to_rgba(-1, colorscale, 0, 1) == "rgba(0,10,20,0.3)"
    assert clustering.value_to_rgba(2, colorscale, 0, 1, alpha=1) == "rgba(100,110,120,1)"
    assert clustering.value_to_rgba(5, colorscale, 5, 5) == "rgba(50,60,70,0.3)"


def test_interpolate_color_returns_single_color_without_interpolation():
    color = np.array([10.0, 20.0, 30.0])

    assert np.array_equal(clustering._interpolate_color([color], 0.75), color)


def test_value_to_rgba_accepts_named_colorscale():
    result = clustering.value_to_rgba(0.5, "Viridis", 0, 1)

    assert result.startswith("rgba(")
    assert result.endswith(",0.3)")


def test_encode_color_value_handles_numeric_and_categorical_series():
    numeric = pd.Series([1, 2], dtype="int64")
    encoded_numeric, numeric_is_categorical, numeric_mapping = clustering.encode_color_value(numeric)
    assert encoded_numeric.tolist() == [1.0, 2.0]
    assert not numeric_is_categorical
    assert numeric_mapping is None

    categorical = pd.Series(["b", "a", "b"])
    encoded_categorical, categorical_is_categorical, categorical_mapping = clustering.encode_color_value(categorical)
    assert encoded_categorical.tolist() == [1, 0, 1]
    assert categorical_is_categorical
    assert categorical_mapping == {0: "a", 1: "b"}
