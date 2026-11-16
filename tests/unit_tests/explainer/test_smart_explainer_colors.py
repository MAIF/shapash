"""
End-to-end tests of color customization: overrides given to SmartExplainer reach the plots.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from shapash import SmartExplainer


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    y = pd.Series((x["a"] + x["b"] > 0).astype(int), name="y")
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(x, y)
    contributions = pd.DataFrame(rng.normal(size=x.shape), columns=x.columns, index=x.index)
    return model, x, y, contributions


def _explainer(data, **kwargs):
    model, x, y, contributions = data
    xpl = SmartExplainer(model=model, **kwargs)
    xpl.compile(x=x, contributions=[-contributions, contributions], y_target=y)
    return xpl


def test_named_contribution_scale_reaches_plots(data):
    """A diverging scale name drives the contribution and correlation plots."""
    xpl = _explainer(data, colors_dict={"contrib_colorscale": "RdBu"})
    scale = xpl.colors_dict["contrib_colorscale"]

    contribution = xpl.plot.contribution_plot("a")
    correlations = xpl.plot.correlations_plot()

    assert scale[0] == "rgb(103, 0, 31)"
    assert contribution.layout.coloraxis.colorscale[0][1] == scale[0]
    assert correlations.layout.coloraxis.colorscale[-1][1] in scale


def test_named_confusion_scale_reaches_plot(data):
    """A scale name drives the confusion matrix."""
    xpl = _explainer(data, colors_dict={"confusion_matrix_colorscale": "viridis"})

    fig = xpl.plot.confusion_matrix_plot()

    assert fig.data[0].colorscale[0][1] == "rgb(68, 1, 84)"


@pytest.mark.parametrize("colors", ["viridis", ["#000000", "#222222", "#444444", "#666666", "#888888", "#aaaaaa"]])
def test_cumulative_importance_accepts_names_and_hex(data, colors):
    """The matplotlib-based cumulative importance plot accepts a scale name and hex colors."""
    xpl = _explainer(data, colors_dict={"feature_contributions_cumulative": colors})

    fig = xpl.plot.features_importance(mode="cumulative")

    assert fig is not None


def test_define_style_with_named_scale(data):
    """define_style accepts the same values as the constructor."""
    xpl = _explainer(data)

    xpl.define_style(colors_dict={"interaction_scale": "viridis", "featureimp_bar": {"1": "#000000"}})

    assert xpl.plot._style_dict["interactions_col_scale"] == ["rgb(68, 1, 84)", "rgb(34, 144, 140)", "rgb(253, 231, 37)"]
    assert xpl.plot.features_importance().data[-1].marker.color[0] == "rgb(0, 0, 0)"


def test_invalid_color_raises_before_any_change(data):
    """An invalid color raises and leaves the current colors in place."""
    xpl = _explainer(data)
    before = xpl.colors_dict

    with pytest.raises(ValueError, match="Invalid value for 'title_color'"):
        xpl.define_style(colors_dict={"title_color": "notacolor"})

    assert xpl.colors_dict is before
