import pandas as pd
import logging
from joblib import Memory

logger = logging.getLogger(__name__)


def _concat_features(feature_dfs, prefixes=None):
    """
    Horizontally concatenate feature dataframes

    :param feature_dfs: List of dataframes
    :param prefixes: List of strings prepended to the columns of each
    dataframe separately ('{prefix}__{orig_col_name}')
    :return: Dataframe with all features
    """
    if prefixes and (len(prefixes) != len(feature_dfs)):
        raise Exception("Length of dfs and prefixes does not aggree")

    if prefixes:
        feature_dfs = [df.add_prefix(pref + '__') for df, pref in
                       zip(feature_dfs, prefixes)]

    df = pd.concat(feature_dfs, axis=1, sort=True)
    return df


def concat_modality_features(features, strategy):
    strategies = {
        "clinical": ['clinical'],
        "structural": ['fs'],
        "structGlobScort": ['structGlobScort'],
        "functional": ['fmri'],
        "fullcon": ["fullcon"],

        #
        "clinical+structural": ['clinical', 'fs'],
        "clinical+structGlobScort": ['clinical', 'structGlobScort'],
        "clinical+functional": ['clinical', 'fmri'],
        'clinical+fullcon': ["clinical", "fullcon"],

        #
        'clinical+structGlobScort+functional': ["clinical", "structGlobScort", "fmri"],
        'clinical+structGlobScort+fullcon': ["clinical", "structGlobScort", "fullcon"],
    }
    modalities = strategies[strategy]
    feature_list = [features[m] for m in modalities]
    df = _concat_features(feature_list)
    return df


def _load_features(files, verify_integrity=True):
    """
    :param files: list of pickled dataframes that will be loaded and stacked
    :param verify_integrity:
    :return: df
    """
    dfs = [pd.read_pickle(f) for f in files]
    df = pd.concat(dfs, axis=0, sort=True, verify_integrity=verify_integrity)
    return df


def _flatten_list(l):
    return [item for sublist in l for item in sublist]


def stack_subjects_features(participant_labels, feature_base_dir, clinical_feature_file, modality):
    features = {}
    # brain
    file_pattern = {
        "fs": "sub-{subject}/ses-*/sub-{subject}_ses-*_desc-fsFeatures.pkl",
        "fmri": "sub-{subject}/ses-*/sub-{subject}_ses-*_desc-fmriFeatures.pkl",
        "fullcon": "sub-{subject}/ses-*/sub-{subject}_ses-*_desc-fmriFullFeatures.pkl"
    }
    feature_files = {}

    load_kind_brain = []
    if ('structural' in modality) or ('structGlobScort' in modality):
        load_kind_brain.append("fs")
    if 'functional' in modality:
        load_kind_brain.append("fmri")
    if 'fullcon' in modality:
        load_kind_brain.append("fullcon")

    for kind in load_kind_brain:
        feature_files[kind] = []
        for subject in participant_labels:
            search_pattern = file_pattern[kind].format(subject=subject)
            files_found = list(feature_base_dir.glob(search_pattern))
            if len(files_found) == 0:
                raise FileNotFoundError(f"{search_pattern} not found for {subject}")
            if len(files_found) > 1:
                raise FileNotFoundError(f"{search_pattern} more than one file found for {subject}")
            feature_files[kind].append(files_found[0])

    if load_kind_brain:
        features = {mod: _load_features(files) for mod, files in feature_files.items()}

        # check dimensions are reasonable

        for mod, df in features.items():
            assert len(feature_files[mod]) == len(df)

    if 'structGlobScort' in modality:
        features['structGlobScort'] = features['fs'].filter(regex='subcor|glob')
        del features['fs']

    # clinical
    if 'clinical' in modality:
        clin = load_and_filter_df(clinical_feature_file, participant_labels)
        features["clinical"] = clin
        feature_files["clinical"] = clinical_feature_file

    return features, feature_files


def check_in_index(df, lookfor):
    missing = set(lookfor) - set(df.index)
    if missing:
        raise RuntimeError(f"Requested subjects missing from df: {missing}")


def load_and_filter_df(pickle_file, participant_labels):
    df = pd.read_pickle(pickle_file)
    df = pd.DataFrame(df)  # ensure one-col tables are dataframes not series
    check_in_index(df, participant_labels)
    df = df.filter(items=participant_labels, axis=0)
    assert len(df) == len(participant_labels)
    return df


def load_features_cached(cache_dir, participant_label, feature_base_dir, clinical_feature_file, modality='clinical'):
    memory = Memory(cache_dir / "load", verbose=0)

    allowed_modalities = ["clinical", "structural", "structGlobScort", "functional", "fullcon"]
    modality_parts = modality.split("+")
    for m in modality_parts:
        if m not in allowed_modalities:
            raise ValueError(m, modality)

    features, feature_files = memory.cache(stack_subjects_features)(participant_label, feature_base_dir,
                                                                    clinical_feature_file, modality)

    stacked_features = memory.cache(concat_modality_features)(features, modality)
    return stacked_features


def fillna_mia(X, fill_value=999):
    X_ = X.copy()
    nans = X_.isna().any()
    nan_cols = nans[nans].index
    X_pos = X_[nan_cols].fillna(value=fill_value)
    X_pos.columns = [c + "_missing" for c in X_pos.columns]
    X_.fillna(value=-fill_value, inplace=True)
    X_ = pd.concat((X_, X_pos), axis=1)
    return X_
