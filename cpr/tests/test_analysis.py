import pandas as pd
import pytest

from ..prediction.analysis import compare_r2_distributions


def test_compare_r2_1_equal():
    df_in = pd.DataFrame({
        "m1_r2": [1, 2, 3, 4, 5],
        "m2_r2": [1, 2, 3, 4, 5],
    })

    expected = pd.Series({"m1_better_m2": 0., "m2_better_m1": 0., "n_splits": 5., "median_diff": 0.})

    comp = compare_r2_distributions(df_in, "m1_r2", "m2_r2")
    pd.testing.assert_series_equal(expected, comp)


def test_compare_r2_2_diff():
    df_in = pd.DataFrame({
        "m1_r2": [2, 3, 4, 5, 5],
        "m2_r2": [1, 2, 3, 4, 5],
    })

    expected = pd.Series({"m1_better_m2": 4 / 5., "m2_better_m1": 0., "n_splits": 5., "median_diff": -1.})

    comp = compare_r2_distributions(df_in, "m1_r2", "m2_r2")
    pd.testing.assert_series_equal(expected, comp)


def test_compare_r2_3_group():
    df_in = pd.DataFrame({
        "g": ["A", "A", "A", "B", "B", "B"],
        "m1_r2": [1, 1, 1, 2, 2, 1],
        "m2_r2": [.5, .5, .5, .5, .5, 2],
    })

    expected = pd.DataFrame({"m1_better_m2": [1., 2 / 3.], "m2_better_m1": [0., 1 / 3.], "n_splits": [3., 3.],
                             "median_diff": [-.5, -1.5]
                             },
                            index=pd.Index(["A", "B"], name="g"))

    comp = df_in.groupby("g").apply(compare_r2_distributions, "m1_r2", "m2_r2")
    pd.testing.assert_frame_equal(expected, comp)
