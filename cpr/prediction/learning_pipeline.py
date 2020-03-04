import pandas as pd
import sklearn
import numpy as np
from joblib import Memory, Parallel, delayed

import logging
import time
import pickle
from pathlib import Path
import copy

logger = logging.getLogger(__name__)

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection._validation import _shuffle
from sklearn.model_selection import learning_curve
from sklearn.utils import check_random_state
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

import multiprocessing


def recode_sex(df, sex_col):
    df_ = df.copy()
    df_[sex_col] = df_[sex_col].replace({'F': 0, 'M': 1})
    return df_


def recode_diagnosis(df, diag_col):
    df_ = df.copy()
    df_[diag_col] = df_[diag_col].replace({'hc': 0, 'mci': 1, 'dem': 2})
    return df_


def get_iterative_imputer(X, random_state, feature_names_after_pca=None):
    """
    if pca run prior to imputer feature_names_after_pca needs to be specified
    """
    imputer = IterativeImputer(add_indicator=True, random_state=random_state, n_nearest_features=50)
    if feature_names_after_pca:
        missing_cols = [c + "_missing" for c in X.columns[X.isna().any()]]
        feature_names = feature_names_after_pca + missing_cols
    else:
        missing_cols = [c + "_missing" for c in X.columns[X.isna().any()]]
        feature_names = X.columns.to_list() + missing_cols
    return imputer, feature_names


def test_get_iterative_imputer():
    X = pd.DataFrame({'v1': [1, 2, 3, 4, np.nan, 5],
                      'v2': [11, 12, 13, 14, 15, 16]
                      }
                     )
    expected_feature_names = ['v1', 'v2', 'v1_missing']

    imputer, feature_names = get_iterative_imputer(X, random_state=1)
    assert isinstance(imputer, IterativeImputer)
    assert expected_feature_names == feature_names


def test_get_iterative_imputer_after_pca():
    X = pd.DataFrame({
        'v1': [1, 2, 3, 4, np.nan, 5],
        'v2': [11, 12, 13, 14, 15, 16],
        'fmri__1': [1, 2, 3, 4, 4.4, 5],
        'fmri__2': [1, 2, 3, 4, 4.4, 5],
    }
    )
    expected_feature_names = ['fmri__pca_0', 'v1', 'v2', 'v1_missing']

    pca_transformer, feature_names_after_pca = \
        get_fmri_pca_transformer(X.columns, random_state=1, n_components=1)
    imputer, feature_names = get_iterative_imputer(X, random_state=1, feature_names_after_pca=feature_names_after_pca)
    assert isinstance(imputer, IterativeImputer)
    assert expected_feature_names == feature_names


def check_missings(X, allowed_missing_cols):
    X_ = X.drop(columns=set(X.columns) & set(allowed_missing_cols))
    missings = X_.isna().any()
    missings = missings[missings]
    missing_cols = missings.index.to_list()
    if missing_cols:
        raise RuntimeError(
            f"Columns {missing_cols} contain missings, but are not included in the allowed_missing_cols")


def format_gs_results(gs, split):
    """
    Takes a grid search object and a split number and returns a data frame of form
            split  mean_train_score_gs  mean_test_score_gs  gs_param_1 gs_param_2
    0      0             0.386432            0.278424  linear   0.1
    1      0            -0.011290           -0.161823    poly   0.1
    """
    gs_results_split = (pd.DataFrame(gs.cv_results_)[['params', 'mean_train_score', 'mean_test_score']].
                        rename(columns={'mean_train_score': 'mean_train_score_gs',
                                        'mean_test_score': 'mean_test_score_gs'}
                               )
                        )
    gs_results_split = pd.concat(
        [gs_results_split.drop(columns=['params']),
         pd.DataFrame((d for idx, d in gs_results_split["params"].iteritems()))],
        axis=1)
    gs_results_split.insert(0, "split", split)
    return gs_results_split


def get_fmri_pca_transformer(feature_names, random_state, n_components):
    from sklearn.decomposition import PCA
    from sklearn.compose import ColumnTransformer
    pca_features = [s for s in feature_names if s.startswith("fmri__")]
    pca = make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=random_state))
    pca_transformer = ColumnTransformer(remainder='passthrough',
                                        transformers=[('pca', pca, pca_features)]
                                        )
    # preprocessor.fit_transform will return the pca columns first, followed by remaining columns
    pca_out_features = [f"fmri__pca_{i}" for i in range(n_components)]
    remaining_cols = pd.Index(feature_names).drop(pca_features).to_list()
    if pca_features:
        feature_names_after_pca = pca_out_features + remaining_cols
    else:
        feature_names_after_pca = remaining_cols
    return pca_transformer, feature_names_after_pca


def get_pipeline(X_train, random_state, n_jobs=1, model_type="basic"):
    if model_type == "basic":
        rf = RandomForestRegressor(n_jobs=n_jobs, random_state=random_state)
        imputer, feature_names = get_iterative_imputer(X_train, random_state)
        param_pref = "randomforestregressor__"
        pipeline = make_pipeline(imputer, rf)

    elif model_type.startswith("basic+fmripca"):
        n_components = int(model_type.split("+fmripca")[-1])
        pca_transformer, feature_names_after_pca = \
            get_fmri_pca_transformer(X_train.columns, random_state=random_state, n_components=n_components)
        imputer, feature_names = get_iterative_imputer(X_train, random_state,
                                                       feature_names_after_pca=feature_names_after_pca)
        rf = RandomForestRegressor(n_jobs=n_jobs, random_state=random_state)
        param_pref = "randomforestregressor__"
        pipeline = make_pipeline(pca_transformer, imputer, rf)

    elif model_type == "rfc":
        imputer, feature_names = get_iterative_imputer(X_train, random_state)
        param_pref = "randomforestclassifier__"
        rfc = RandomForestClassifier()
        pipeline = make_pipeline(imputer, rfc)

    elif model_type.startswith("rfc+fmripca"):
        n_components = int(model_type.split("+fmripca")[-1])
        pca_transformer, feature_names_after_pca = \
            get_fmri_pca_transformer(X_train.columns, random_state=random_state, n_components=n_components)
        imputer, feature_names = get_iterative_imputer(X_train, random_state,
                                                       feature_names_after_pca=feature_names_after_pca)

        param_pref = "randomforestclassifier__"
        rfc = RandomForestClassifier()
        pipeline = make_pipeline(pca_transformer, imputer, rfc)

    elif model_type == "ridgecv":
        imputer, feature_names = get_iterative_imputer(X_train, random_state)
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        param_pref = ""
        alphas = np.logspace(-3, 5, 100)
        pipeline = make_pipeline(imputer, StandardScaler(), RidgeCV(alphas))

    elif model_type.startswith("ridgecv+fmripca"):
        n_components = int(model_type.split("+fmripca")[-1])
        pca_transformer, feature_names_after_pca = \
            get_fmri_pca_transformer(X_train.columns, random_state=random_state, n_components=n_components)
        imputer, feature_names = get_iterative_imputer(X_train, random_state,
                                                       feature_names_after_pca=feature_names_after_pca)
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        param_pref = ""
        alphas = np.logspace(-3, 5, 100)
        pipeline = make_pipeline(pca_transformer, imputer, StandardScaler(), RidgeCV(alphas))

    else:
        raise NotImplementedError(model_type)

    return pipeline, feature_names, param_pref


def train(X_train, y_train, sample_weights_train, split, run_gs, param_dist, random_state, model_type, n_jobs,
          dump_dir):
    pipeline, feature_names, param_pref = get_pipeline(X_train, random_state, n_jobs, model_type)

    param_dist = {param_pref + k: param_dist[k] for k in param_dist.keys()}

    try:
        if run_gs:
            # to silence future warning and not change in future sklearn versions
            r2 = make_scorer(r2_score, multioutput='uniform_average')
            gs = RandomizedSearchCV(pipeline, cv=5, param_distributions=param_dist, n_jobs=n_jobs,
                                    return_train_score=True,
                                    scoring=r2, n_iter=18,
                                    # to make each iteration inner loop use different parameters, shift random_state
                                    random_state=random_state + split,
                                    iid=False, )
            if isinstance(sample_weights_train, np.ndarray):
                do_sample_weights = True
            elif isinstance(sample_weights_train, bool):
                do_sample_weights = sample_weights_train
            else:
                do_sample_weights = False

            if do_sample_weights:
                weights = {f"{param_pref}sample_weight": sample_weights_train}
            else:
                weights = {}
            gs.fit(X_train, y_train, **weights)

            best_estimator = gs.best_estimator_
            gs_results = format_gs_results(gs, split)
        else:
            pipeline.fit(X_train, y_train)
            best_estimator = pipeline
            gs_results = pd.DataFrame()
    except Exception as e:
        dump_dir.mkdir(parents=True, exist_ok=True)
        out_file = dump_dir / f"dump_{split}.pkl"
        with open(out_file, 'wb') as fi:
            pickle.dump({"X_train": X_train, "y_train": y_train, "gs": gs, "e": e}, fi)
        best_estimator = None
        gs_results = pd.DataFrame([])
        feature_names = []
    return best_estimator, gs_results, feature_names


def get_test_predictions(model, X_test, y_test, split):
    y_pred = model.predict(X_test)
    predictions = pd.DataFrame(y_pred,
                               index=y_test.index,
                               columns=[c + "_pred" for c in y_test.columns]
                               )
    predictions.insert(0, 'split', split)
    return predictions


def check_train_test_split(train_subjects, test_subjects):
    overlap = set(train_subjects) & set(test_subjects)
    assert len(overlap) == 0, f"Subjects in train and test split {overlap}"


def permuted_predictions(model, X, y, train_index, test_index, split, random_state, n_jobs):
    """
    Based on sklearn.model_selection.permutation_test_score, but only permutes once and returns predictions
    """
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    random_state = check_random_state(random_state)
    groups = None

    model_permuted = clone(model)
    if list(model_permuted.named_steps.keys())[-1] == 'randomforestregressor':
        model_permuted.named_steps['randomforestregressor'].set_params(n_jobs=n_jobs)

    y_train_permuted = pd.DataFrame(_shuffle(y_train.values, groups=groups, random_state=random_state),
                                    index=y_train.index, columns=y_train.columns)
    X_train, y_train_permuted = X_train.align(y_train_permuted, axis=0)

    y_test_permuted = pd.DataFrame(_shuffle(y_test.values, groups=groups, random_state=random_state),
                                   index=y_test.index, columns=y_test.columns)
    X_test, y_test_permuted = X_test.align(y_test_permuted, axis=0)

    model_permuted.fit(X_train, y_train_permuted)

    predictions_permuted = get_test_predictions(model_permuted, X_test, y_test_permuted, split)
    predictions_permuted = pd.merge(y_test_permuted, predictions_permuted, on='subject', how='outer')

    return predictions_permuted


def get_sample_weights(y, weight_samples):
    """
    if weight_samples is string or Path, assume that it is file with sample weights,
    else, fit kde and derive sample weights
    """
    from sklearn.neighbors.kde import KernelDensity

    if isinstance(weight_samples, bool):
        kde = KernelDensity().fit(y)
        sample_weights = kde.score_samples(y)
        sample_weights *= -1
    elif isinstance(weight_samples, str) or isinstance(weight_samples, Path):
        df = pd.read_pickle(weight_samples)
        _, df = y.align(df, join="left", axis=0)
        sample_weights = df.values.squeeze()
    else:
        raise RuntimeError(weight_samples)

    return sample_weights


def inner_loop(out_base_dir, X, y, split, train_index, test_index, run_gs, param_dist, random_state, model_type,
               sample_weights, n_jobs, save_full_learning_info):
    t1 = time.time()
    logger.info(f"Split {split} started")

    if isinstance(sample_weights, np.ndarray):
        sample_weights_train = sample_weights[train_index]
    elif sample_weights == None:
        sample_weights_train = sample_weights
    else:
        raise RuntimeError(sample_weights)

    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    train_subjects = X.iloc[train_index].index.values
    test_subjects = X.iloc[test_index].index.values
    check_train_test_split(train_subjects, test_subjects)

    # dump for debugging

    # training
    model, gs_results, feature_names_model_in = train(X_train, y_train, sample_weights_train, split, run_gs,
                                                      param_dist, random_state, model_type, n_jobs,
                                                      dump_dir=out_base_dir)

    if model:
        predictions = get_test_predictions(model, X_test, y_test, split)

        predictions_training = get_test_predictions(model, X_train, y_train, split)
        predictions_permuted = permuted_predictions(model, X, y, train_index, test_index, split, random_state, n_jobs)
        feature_names_pipe_in = model.named_steps['columntransformer']._feature_names_in

        model_out_dir = out_base_dir / "models"
        model_out_dir.mkdir(exist_ok=True, parents=True)
        out_file = model_out_dir / f"model_split_{split}.pkl"
        with open(out_file, 'wb') as fi:
            pickle.dump(model, fi)
    else:
        predictions = pd.DataFrame([])
        predictions_training = pd.DataFrame([])
        predictions_permuted = pd.DataFrame([])
        feature_names_pipe_in = []

    t2 = time.time()
    execution_time = t2 - t1
    logger.info(f"Split {split} inner loop done. {execution_time:.1f} seconds")

    info_dict = {'train_subjects': train_subjects, 'test_subjects': test_subjects,
                 'train_index': train_index, 'test_index': test_index,
                 'execution_time': execution_time,
                 'feature_names_model_in': feature_names_model_in,
                 'feature_names_pipe_in': feature_names_pipe_in,

                 }
    if save_full_learning_info:
        info_dict.update({'X_train': X_train,
                          'X_test': X_test,
                          'y_train': y_train,
                          'y_test': y_test,
                          'sample_weights_train': sample_weights_train,
                          })
    split_info = {split: info_dict}

    return predictions, predictions_training, predictions_permuted, gs_results, split_info


def prepare_X_y(X, y, modality, allowed_missing_cols):
    # check X, y
    assert (X.index == y.index).all()

    if 'clinical' in modality:
        X = recode_sex(X, 'clin__risk__demo_sex')
        X = recode_diagnosis(X, 'clin__assess__diag')

        if allowed_missing_cols == 'oasis3_basic':
            from .info import oasis3_basic_allowed_missing_cols
            allowed_missing_cols = oasis3_basic_allowed_missing_cols
    else:
        allowed_missing_cols = []
    check_missings(X, allowed_missing_cols)
    return X


def get_stratification(y):
    """
    Takes single or multicolumn y df, aligns polarity of each columns with first column, z-scores and averages values
    and returns 20 groups (percentile)
    """
    from scipy.stats import zscore

    corr = y.corr().iloc[0]
    neg_indicators = corr < 0
    y_pos = y.copy()
    y_pos.loc[:, neg_indicators] = -1 * y_pos.loc[:, neg_indicators]

    z = y_pos.apply(zscore).mean(1)
    stratification = pd.qcut(z, q=20, duplicates="drop")
    return stratification


def learn(X: pd.DataFrame, y: pd.DataFrame, learning_dir, model_name, modality,
          model_type="basic", random_state=1234, allowed_missing_cols='oasis3_basic', run_gs=True, n_splits=1000,
          weight_samples=False, n_jobs_outer=None, save_full_learning_info=False):
    out_base_dir = learning_dir / f"model-{model_name}_modality-{modality}"
    out_base_dir.mkdir(parents=True, exist_ok=True)

    param_dist = {"criterion": ["mse", "mae"],
                  "max_depth": [3, 5, 7, 10, 15, 20, 40, 50, None],
                  "n_estimators": [256],
                  "max_features": ["sqrt"]
                  }
    if model_type.startswith("rfc"):
        param_dist["criterion"] = ["gini", "entropy"]

    if not n_jobs_outer:
        n_jobs_outer = 1 if n_splits < (2 * multiprocessing.cpu_count()) else -1
    n_jobs_inner = multiprocessing.cpu_count() // n_jobs_outer

    X = prepare_X_y(X, y, modality, allowed_missing_cols)

    stratification = get_stratification(y)
    if weight_samples:
        sample_weights = get_sample_weights(y, weight_samples)
    else:
        sample_weights = None

    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=.2, random_state=random_state)
    t1 = time.time()
    predictions, predictions_training, predictions_permuted, gs_results, split_info = \
        zip(*Parallel(n_jobs=n_jobs_outer)(
            delayed(inner_loop)
            (out_base_dir, X, y, split, train_index, test_index, run_gs, param_dist, random_state, model_type,
             sample_weights, n_jobs_inner, save_full_learning_info)
            # input params
            for split, (train_index, test_index) in enumerate(cv.split(X, stratification))
        ))

    t2 = time.time()
    logger.info(f"Outer loop done. {t2 - t1:.1f} seconds")

    df_predictions = pd.concat(predictions)
    df_gs = pd.concat(gs_results)
    df_predictions_training = pd.concat(predictions_training)
    df_predictions_permuted = pd.concat(predictions_permuted)

    # save data

    df_predictions.to_pickle(out_base_dir / "df_predictions.pkl")
    df_predictions_training.to_pickle(out_base_dir / "df_predictions_training.pkl")
    df_predictions_permuted.to_pickle(out_base_dir / "df_predictions_permuted.pkl")
    df_gs.to_pickle(out_base_dir / "df_gs.pkl")

    # unpack list of dicts into one dict
    split_info = {k: v for d in split_info for k, v in d.items()}
    with open(out_base_dir / "split_info.pkl", 'wb') as fi:
        pickle.dump(split_info, fi)

    info = {'X': X, 'y': y, 'sample_weights': sample_weights, 'stratification': stratification,
            'random_state': random_state}
    with open(out_base_dir / "in_data.pkl", 'wb') as fi:
        pickle.dump(info, fi)


def run_learning_curve(X, y, learning_dir, model_name, modality, model_type, best_parameters, random_state=1234,
                       allowed_missing_cols='oasis3_basic'):
    params = best_parameters[modality]

    X = prepare_X_y(X, y, modality, allowed_missing_cols)
    pipeline, feature_names, param_pref = get_pipeline(X, random_state, n_jobs=-1, model_type=model_type)
    pipeline.set_params(**params)

    # to silence future warning and not change in future sklearn versions
    r2 = make_scorer(r2_score, multioutput='uniform_average')
    train_sizes, train_scores, valid_scores = learning_curve(pipeline, X, y, cv=5, scoring=r2,
                                                             random_state=random_state, n_jobs=-1,
                                                             # with .1 PCA cannot do 100 components
                                                             train_sizes=np.linspace(0.2, 1.0, 5)
                                                             )

    # format dfs
    df_train = pd.DataFrame(train_scores, index=pd.Index(train_sizes, name='train_size'),
                            columns=["cv_" + str(i) for i in range(train_scores.shape[1])]
                            )
    df_train.insert(0, "kind", "train")

    df_valid = pd.DataFrame(valid_scores, index=pd.Index(train_sizes, name='train_size'),
                            columns=["cv_" + str(i) for i in range(valid_scores.shape[1])]
                            )
    df_valid.insert(0, "kind", "validation")
    df = pd.concat((df_train, df_valid), axis=0).reset_index()

    # save data
    p = learning_dir / "learning_curves" / f"model-{model_name}_modality-{modality}"
    p.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p / "learning_scores.pkl")


def get_pi_split(data_root_dir, X, y, split, modality, info_all_splits, random_state, n_jobs=1):
    model_file = data_root_dir / f"models/model_split_{split}.pkl"
    model = pd.read_pickle(model_file)

    info_split = info_all_splits[split]
    X_test = X.loc[info_split["test_subjects"]]
    y_test = y.loc[info_split["test_subjects"]]

    r2 = make_scorer(r2_score, multioutput='uniform_average')
    pi = permutation_importance(model, X_test, y_test, n_repeats=5, scoring=r2, random_state=random_state,
                                n_jobs=n_jobs)

    df = pd.DataFrame({"feature": model.named_steps['columntransformer']._feature_names_in,
                       "permutation_importance": pi.importances_mean,
                       })
    df.insert(0, "split", split)
    df.insert(0, "modality", modality)

    return df


def run_permutation_importance(X, y, learning_dir, model_name, modality,
                               n_splits, n_jobs_outer=None, random_state=1234):
    data_root_dir = learning_dir / f"model-{model_name}_modality-{modality}"

    if not n_jobs_outer:
        n_jobs_outer = 1 if n_splits < (2 * multiprocessing.cpu_count()) else -1
    n_jobs_inner = multiprocessing.cpu_count() // n_jobs_outer

    allowed_missing_cols = 'oasis3_basic'
    X = prepare_X_y(X, y, modality, allowed_missing_cols)

    info_file = data_root_dir / "split_info.pkl"
    info_all_splits = pd.read_pickle(info_file)

    dfs = Parallel(n_jobs=n_jobs_outer)(delayed(get_pi_split)
                                        (data_root_dir, X, y, split, modality, info_all_splits, random_state,
                                         n_jobs=n_jobs_inner)
                                        # input params
                                        for split in range(n_splits)
                                        )
    df = pd.concat(dfs)
    df.to_pickle(data_root_dir / "permutation_importance.pkl")
