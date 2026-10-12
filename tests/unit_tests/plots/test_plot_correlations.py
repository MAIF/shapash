from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from shapash.plots.plot_correlations import compute_corr, plot_correlations


def test_plot_correlations_samples_rows_when_sample_size_is_set():
    row_count = 250
    df = pd.DataFrame(
        {
            "feature_a": np.arange(row_count),
            "feature_b": np.arange(row_count) ** 2,
            "category": [f"category_{index}" for index in range(row_count)],
        }
    )
    original_df = df.copy(deep=True)

    with patch("shapash.plots.plot_correlations.compute_corr", wraps=compute_corr) as compute_corr_spy:
        plot_correlations(df, sample_size=20, how="pearson")

    sampled_df = compute_corr_spy.call_args.args[0]
    assert len(sampled_df) == 20
    assert_frame_equal(df, original_df)
