from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, f1_score
import numpy as np
from scipy import stats


def load_and_format_learning_curves(learning_dir, model_name, modalities):
    dfs = []
    for modality in modalities:
        f = learning_dir / f"model-{model_name}_modality-{modality}" / "learning_scores.pkl"
        df_ = pd.read_pickle(f)
        df_.insert(0, 'modality', modality)
        dfs.append(df_)
    df = pd.concat(dfs)
    df = df.melt(id_vars=['modality', 'train_size', 'kind'], var_name="cv_split", value_name="score")
    df.cv_split = df.cv_split.str.replace("cv_", "")
    return df


def load_predictions(learning_dir, model_name, modalities, pickle_file):
    dfs = []
    for modality in modalities:
        f = learning_dir / f"model-{model_name}_modality-{modality}" / pickle_file
        df_ = pd.read_pickle(f)
        df_.insert(0, 'modality', modality)
        dfs.append(df_)
    df = pd.concat(dfs)
    assert not df.isna().any().any(), "NaNs detected"
    return df


def load_and_format_predictions(learning_dir, model_name, modalities, y_file, pickle_file="df_predictions.pkl",
                                y_cols_renamed=None, sample_weights_file=None):
    """
    loads results and returns df like
    subject	modality	split	mmse_pred	sob_pred	mmse	sob
    xxxx	mini	    0	    -0.16	    0.11	    -0.19	-0.12

    """
    learning_dir = Path(learning_dir)
    df_predictions = load_predictions(learning_dir, model_name, modalities, pickle_file)
    df_true = pd.read_pickle(y_file)
    df = pd.merge(df_predictions, df_true, on='subject', how='left')

    if sample_weights_file:
        sample_weights = pd.read_pickle(sample_weights_file)
        df = pd.merge(df, sample_weights, on='subject', how='left')

    if y_cols_renamed:
        for orig, new in y_cols_renamed.items():
            df.columns = [c.replace(orig, new) for c in df.columns]
    return df


def load_permuted_predictions(learning_dir, model_name, modalities):
    dfs = []
    for modality in modalities:
        f = learning_dir / f"model-{model_name}_modality-{modality}" / "df_predictions_permuted.pkl"
        df_ = pd.read_pickle(f)
        df_.insert(0, 'modality', modality)
        dfs.append(df_)
    df = pd.concat(dfs)
    assert not df.isna().any().any(), "NaNs detected"
    return df


def load_and_format_premuted_predictions(learning_dir, model_name, modalities, y_cols_renamed=None,
                                         sample_weights_file=None):
    """
    loads results and returns df like
    subject	modality	split	mmse_slope_pred	sob_slope_pred	mmse_slope	sob_slope
    xxxx	mini	    0	    -0.16	        0.11	        -0.19	    -0.12
    """
    learning_dir = Path(learning_dir)
    df = load_permuted_predictions(learning_dir, model_name, modalities)

    if sample_weights_file:
        sample_weights = pd.read_pickle(sample_weights_file)
        df = pd.merge(df, sample_weights, on='subject', how='left')

    if y_cols_renamed:
        for orig, new in y_cols_renamed.items():
            df.columns = [c.replace(orig, new) for c in df.columns]
    return df


def calc_metrics(df, y_cols=['sob_slope', 'mmse_slope'], metric_names=['r2', 'mae'], group_cols=['modality', 'split'],
                 kind="", weight_col=None):
    """
    requires df like
    subject	modality	split	mmse_slope_pred	sob_slope_pred	mmse_slope	sob_slope
    xxxx	mini	    0	    -0.16	        0.11	        -0.19	    -0.12

    Returns df like
    modality	split	r2_sob	pearsonr2_sob	r2_average
    mini	    0	    0.31	0.21	        0.26
    """
    y_pred_cols = [f"{c}_pred" for c in y_cols]

    def get_weights(d, weight_col):
        if weight_col:
            weights = d[weight_col]
        else:
            weights = None
        return weights

    def r2(d, true_col, pred_col, weight_col=None):
        weights = get_weights(d, weight_col)
        return r2_score(d[true_col], d[pred_col], sample_weight=weights)

    def mae(d, true_col, pred_col, weight_col=None):
        weights = get_weights(d, weight_col)
        return mean_absolute_error(d[true_col], d[pred_col], sample_weight=weights)

    def f1(d, true_col, pred_col, weight_col=None):
        weights = get_weights(d, weight_col)
        return f1_score(d[true_col], d[pred_col], average="macro", sample_weight=weights)

    def pearsonr2(d, true_col, pred_col, weight_col=None):
        if weight_col:
            import warnings
            warnings.warn("weighting not implemented for correlation, computing unweighted scores")
        return d[[true_col, pred_col]].corr().iloc[0, 1] ** 2

    metrics = pd.DataFrame()

    for y_col, y_pred_col in zip(y_cols, y_pred_cols):
        if 'r2' in metric_names:
            metrics[f'r2_{y_col}'] = df.groupby(group_cols).apply(r2, y_col, y_pred_col, weight_col)
        if 'pearsonr2' in metric_names:
            metrics[f'pearsonr2_{y_col}'] = df.groupby(group_cols).apply(pearsonr2, y_col, y_pred_col, weight_col)
        if 'mae' in metric_names:
            metrics[f'mae_{y_col}'] = df.groupby(group_cols).apply(mae, y_col, y_pred_col, weight_col)
        if 'f1' in metric_names:
            metrics[f'f1_{y_col}'] = df.groupby(group_cols).apply(f1, y_col, y_pred_col, weight_col)

    if len(y_cols) > 1:
        if 'r2' in metric_names:
            metrics['r2_average'] = df.groupby(group_cols).apply(r2, y_cols, y_pred_cols)
        if 'f1' in metric_names:
            metrics['f1_average'] = df.groupby(group_cols).apply(f1, y_cols, y_pred_cols)
        if 'pearsonr2' in metric_names:
            pearson_cols = [f'pearsonr2_{c}' for c in y_cols]
            metrics['pearsonr2_average'] = metrics[pearson_cols].mean(axis=1)

    metrics["kind"] = kind

    return metrics.reset_index()


def load_gs(learning_dir, model_name, modalities):
    dfs = []
    for modality in modalities:
        f = learning_dir / f"model-{model_name}_modality-{modality}" / "df_gs.pkl"
        df_ = pd.read_pickle(f)
        df_.insert(0, 'modality', modality)
        dfs.append(df_)
    df = pd.concat(dfs)
    return df


def load_and_format_gs(learning_dir, model_name, modalities):
    learning_dir = Path(learning_dir)
    df = load_gs(learning_dir, model_name, modalities)
    df.fillna(0, inplace=True)
    return df


def load_and_format_fi(learning_dir, model_name, modalities, n_splits=None):
    learning_dir = Path(learning_dir)
    dfs = []
    for modality in modalities:
        print(modality)
        f = learning_dir / f"model-{model_name}_modality-{modality}" / "split_info.pkl"
        split_info = pd.read_pickle(f)

        model_files = list(learning_dir.glob(f"model-{model_name}_modality-{modality}/models/model*.pkl"))
        model_files.sort()
        if n_splits:
            model_files = model_files[:n_splits]

        for model_file in model_files:
            model = pd.read_pickle(model_file)
            split = int(model_file.name.split("model_split_")[-1].split(".pkl")[0])

            feature_names = split_info[split]['feature_names_model_in']

            fi = pd.DataFrame({"fi": model.named_steps['randomforestregressor'].feature_importances_,
                               "feature_name": feature_names
                               })
            fi.insert(0, 'modality', modality)
            fi.insert(0, 'split', split)
            dfs.append(fi)
    df = pd.concat(dfs, sort=False).reset_index(drop=True)
    return df


def metrics_to_long(df, metric_type, y_cols=['sob_slope', 'mmse_slope']):
    """
    requires df like
    modality	split	r2_sob	pearsonr2_sob	r2_average
    mini	    0	    0.31	0.21	        0.26

    returns df like
	modality	split	kind	target	r2
    0	mini	0	    test	average	    0.26
    """
    allowed_metrics = ['pearsonr2', 'r2', 'f1', 'mae']
    if metric_type not in allowed_metrics:
        raise Exception(metric_type)

    if (metric_type in ['r2', 'f1', 'pearsonr2']) and (len(y_cols) > 1):
        value_vars = [f'{metric_type}_average']
    else:
        value_vars = []
    value_vars += [f"{metric_type}_{c}" for c in y_cols]

    long = df.melt(id_vars=['modality', 'split', 'kind'],
                   value_vars=value_vars,
                   var_name='target',
                   value_name=metric_type)
    for m in allowed_metrics:
        long["target"] = long["target"].str.replace(f"{m}_", "")

    return long


def label_median(ax, max_median_line=True, font_size=10):
    # https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
    lines = ax.get_lines()
    categories = ax.get_yticks()
    max_x = max([lines[1 + cat * 6].get_xdata()[0] for cat in categories])
    max_median = max([lines[4 + cat * 6].get_xdata()[0] for cat in categories])

    for cat in categories:
        # every 4th line at the interval of 6 is median line
        # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        # x = round(lines[4+cat*6].get_xdata()[0],1)
        x = round(lines[4 + cat * 6].get_xdata()[0], 2)
        x_pos = lines[1 + cat * 6].get_xdata()[0]

        ax.text(
            x_pos + max_x * .05,
            cat - categories.max() * .01,
            f'{x}',
            ha='left',
            va='bottom',
            fontweight='bold',
            size=font_size,
            color='black',
        )
    if max_median_line:
        ax.axvline(max_median, color="k", ls="--")


def compare_r2_distributions(df, m1_col, m2_col):
    """
    Compoares two distributions of R2 values

    returns Series with
    - "m1_better_m2": ratio of R2 values m1>m2
    - "m2_better_m1": ratio of R2 values m2>m1
    - "n_splits"
    - "median_diff": median difference of splits (m2-m1)

    compare_r2_distributions(df_in, "m1_r2", "m2_r2")
    df_in.groupby("g").apply(compare_r2_distributions, "m1_r2", "m2_r2")
    """
    comp = pd.Series({
        "m1_better_m2": (df[m1_col] > df[m2_col]).mean(),
        "m2_better_m1": (df[m2_col] > df[m1_col]).mean(),
        "n_splits": len(df),
        "median_diff": (df[m2_col] - df[m1_col]).median()

    })
    return comp
