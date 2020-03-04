from ..prediction.utils import _concat_features, fillna_mia
from ..prediction.learning_pipeline import check_missings, format_gs_results, check_train_test_split
import pandas as pd
import pytest


def test_concat_features():
    df_expected = pd.DataFrame({'a': [1, 2, pd.np.nan],
                                'b': [pd.np.nan, 22, 33]},
                               index=["s1", "s2", "s3"])
    df1 = pd.DataFrame({'a': [2, 1]}, index=["s2", "s1"])
    df2 = pd.DataFrame({'b': [22, 33]}, index=["s2", "s3"])

    df_out = _concat_features([df1, df2])
    pd.testing.assert_frame_equal(df_out, df_expected)


def test_concat_features_prefix():
    df1 = pd.DataFrame({'a': [1, 2]}, index=["s1", "s2"])
    df2 = pd.DataFrame({'b': [11, 22]}, index=["s1", "s2"])
    df_expected = pd.DataFrame({'f1__a': [1, 2], 'f2__b': [11, 22]},
                               index=["s1", "s2"])
    df_out = _concat_features([df1, df2], ['f1', 'f2'])
    pd.testing.assert_frame_equal(df_out, df_expected)


def test_check_missings_ok():
    df = pd.DataFrame({'a': [1, 2, 3],
                       'b': [pd.np.nan, 22, 33]},
                      index=["s1", "s2", "s3"])
    allowed_missing_cols = ['b']
    check_missings(df, allowed_missing_cols)


def test_check_missings_missing():
    df = pd.DataFrame({'a': [pd.np.nan, 2, 3],
                       'b': [pd.np.nan, 22, 33],
                       'c': [1, 22, 33]},
                      index=["s1", "s2", "s3"])
    allowed_missing_cols = ['b']

    with pytest.raises(RuntimeError) as e:
        check_missings(df, allowed_missing_cols)
    assert str(e.value) == "Columns ['a'] contain missings, but are not included in the allowed_missing_cols"


def test_format_gs_results():
    from sklearn import datasets, svm, model_selection, ensemble
    X, y = datasets.make_regression(50, 3, 3)
    svr = svm.SVR()
    gs = model_selection.RandomizedSearchCV(svr, cv=3,
                                            param_distributions={'kernel': ['linear', 'poly'],
                                                                 'gamma': [.1, 1, 'auto']},
                                            iid=False,
                                            return_train_score=True)
    gs.fit(X, y)
    gs_results = format_gs_results(gs, 0)
    assert 'kernel' in gs_results.columns
    assert 'gamma' in gs_results.columns
    assert gs_results.shape == (6, 5)

    gs_results = format_gs_results(gs, 1)
    assert gs_results.split.unique() == [1]


def test_check_train_test_split():
    check_train_test_split(['s1', 's2'], ['s3', 's4'])
    with pytest.raises(AssertionError) as e:
        check_train_test_split(['s1', 's2'], ['s1', 's4'])


def test_fillna_mia():
    X = pd.DataFrame([[1, 2], [3, 4], [5, pd.np.nan]], columns=["a", "b"])
    X_expected = pd.DataFrame([[1, 2., 2], [3, 4., 4.], [5, -999., 999.]], columns=["a", "b", "b_missing"])
    X_filled = fillna_mia(X)
    assert list(X_filled.columns) == ["a", "b", "b_missing"]
    pd.testing.assert_frame_equal(X_filled, X_expected)
