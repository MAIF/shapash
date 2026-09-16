import numpy as np
import pandas as pd
import pytest

from shapash.utils.sampling import subset_sampling


class TestSubsetSampling:
    def test_selection_within_max_points_returned_as_is(self):
        df = pd.DataFrame({"x": range(10)})
        idx, note = subset_sampling(df, selection=[1, 2, 3], max_points=50)
        assert idx == [1, 2, 3]
        assert "user-defined" in note

    def test_no_selection_df_within_max_points_returns_all_index(self):
        df = pd.DataFrame({"x": range(10)}, index=range(10))
        idx, note = subset_sampling(df, selection=None, max_points=50)
        assert idx == list(range(10))
        assert note is None

    def test_list_selection_random_subset_when_no_col(self):
        df = pd.DataFrame({"x": range(200)})
        selection = list(range(150))
        idx, note = subset_sampling(df, selection=selection, max_points=50, col=None)
        assert len(idx) == 50
        assert "random Subset" in note

    def test_list_selection_smart_subset_with_col(self):
        df = pd.DataFrame({"cat": np.arange(200) % 5})
        idx, note = subset_sampling(df, selection=list(range(150)), max_points=50, col="cat", col_value_count=5)
        assert len(idx) == 50
        assert "smart Subset" in note

    def test_no_selection_random_subset_when_no_col(self):
        df = pd.DataFrame({"x": range(3000)})
        idx, note = subset_sampling(df, selection=None, max_points=50, col=None)
        assert len(idx) == 50
        assert "random Subset" in note

    def test_no_selection_smart_subset_uses_kmeans_for_high_cardinality_numeric_col(self):
        df = pd.DataFrame({"val": np.random.default_rng(0).random(200) * 100})
        idx, note = subset_sampling(df, selection=None, max_points=50, col="val", col_value_count=200)
        assert len(idx) == 50
        assert "smart Subset" in note

    def test_invalid_selection_type_raises_value_error(self):
        df = pd.DataFrame({"x": range(10)})
        with pytest.raises(ValueError, match="must be a list"):
            subset_sampling(df, selection="not-a-list", max_points=5)
