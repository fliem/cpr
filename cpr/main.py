from pathlib import Path
import pandas as pd

from .utils import check_subject, SubjectData, get_baseline_session_mapping, get_subjects_in_dir, bids_to_tmp
from .features import fs, fmri
from .prediction.utils import load_features_cached, load_and_filter_df
from .prediction.learning_pipeline import learn, run_learning_curve, run_permutation_importance
from collections import Iterable
from bids import BIDSLayout

import logging

VALID_STAGES = {'participant': ['preprocessing', 'feature_extraction', 'prepare', 'learn'],
                'group': ['learn', 'learning_curve', 'permutation_importance']}


def _check_args(analysis_level, stages, participant_label, session_label, baseline_sessions_file, check_pickles=None):
    # checks if input args are reasonable and removes sub- and ses- prefixes

    # set default stages
    if not stages:
        if analysis_level == "participant":
            stages = ["prepare"]
        elif analysis_level == "group":
            raise NotImplementedError()

    for stage in stages:
        if stage not in VALID_STAGES[analysis_level]:
            raise RuntimeError(f"analysis level {analysis_level} not compatible with stage {stage}. "
                               f"Allowed are {VALID_STAGES[analysis_level]}")

    if not isinstance(participant_label, Iterable) or isinstance(participant_label, str):
        participant_label = [participant_label]
    subjects = [p.lstrip("sub-") for p in participant_label]

    if len(subjects) > 1 and session_label:
        raise RuntimeError(f"session_label only works with a single subject, but you specified {participant_label}.")

    if session_label and baseline_sessions_file:
        raise RuntimeError(f"Either specify session_label or baseline_session_file, but not both!")

    session = session_label.lstrip("ses-") if session_label else None

    if baseline_sessions_file:
        if not Path(baseline_sessions_file).is_file():
            raise RuntimeError(f"Baseline session file does not exist {baseline_sessions_file}")

    if check_pickles:
        for f in check_pickles:
            if f:
                if not f.is_file():
                    raise RuntimeError(f"{f} not an existing file")
                if '.pkl' not in f.suffixes:
                    raise RuntimeError(f"{f} not a pickle file")

    return subjects, session, stages


def main(bids_dir, output_dir, analysis_level, stages, participant_label=None, session_label=None,
         mr_baseline_sessions_file=None, clinical_feature_file=None, target_file=None,
         fs_license_file=None, learning_out_subdir=None, test_run=False, n_cpus=1, verbose=False,
         modalities=[], model_name="basic", model_type="basic", n_splits=1000,
         best_cv_parameters_lcurve=None, weight_samples=False, n_jobs_outer=None,
         save_full_learning_info=False):
    bids_dir = Path(bids_dir)
    output_dir = Path(output_dir)
    learning_dir = output_dir / "learning"

    # get logger
    logger = logging.getLogger(__file__)
    formatter = {"format": '%(asctime)s %(levelname)-8s %(message)s', "datefmt": '%Y-%m-%d %H:%M:%S'}
    if verbose:
        logging.basicConfig(level=logging.DEBUG, **formatter)
        logger.info("Setting logging level to DEBUG")
    else:
        logging.basicConfig(level=logging.INFO, **formatter)
        logger.info("Setting logging level to INFO")

    if not participant_label:
        participant_label = get_subjects_in_dir(bids_dir)

    subjects, session, stages = _check_args(analysis_level, stages, participant_label, session_label,
                                            mr_baseline_sessions_file,
                                            check_pickles=[clinical_feature_file, target_file])

    # assemble subject info
    session_mapping = get_baseline_session_mapping(subjects, session, mr_baseline_sessions_file)

    if analysis_level == "participant":
        tmp_dir_obj, tmp_bids_dir = bids_to_tmp(bids_dir, subjects, session_mapping)
        layout = BIDSLayout(tmp_bids_dir)

        subject_data_list = []
        for subject in subjects:
            session = session_mapping[subject]
            logger.info(f"Checking subject {subject}")
            check_subject(layout, subject, session)
            subject_data_list.append(SubjectData(layout, subject, session, output_dir))

        # PREPARE
        for subject_data in subject_data_list:
            # PREPROCESSING
            logger.info(f"Running {subject_data.subject} {subject_data.session}")
            logger.debug(f"**** SUBJECT INFO ****\n{subject_data}\n**** SUBJECT INFO ****")
            if ("prepare" in stages) or ("preprocessing" in stages):
                found, missing = subject_data.check_preprocessing(raise_on_missing=False)
                if missing:
                    logger.info("Preprocessing not found. Running preprocessing")
                    logger.debug(f"Found: {found}.\nMissing {missing}")

                    subject_data.run_fmriprep(fs_license_file, test_run, n_cpus)
                else:
                    logger.info("Preprocessing found.")
                    logger.debug(f"Found: {found}.\nMissing {missing}")

            if ("prepare" in stages) or ("feature_extraction" in stages):
                found, missing = subject_data.check_features(raise_on_missing=False)
                if missing:
                    logger.info(f"Features not found. Running feature extraction "
                                f"{subject_data.subject} {subject_data.session}")
                    logger.info(f"Feature extraction freesurfer")
                    fs.extract_fs_features(subject_data)

                    logger.info(f"Feature extraction fmri")
                    fmri.extract_connectivity_features(subject_data)
                    logger.info(f"Feature extraction done {subject_data.subject} {subject_data.session}")
                    _, _ = subject_data.check_features(raise_on_missing=True)
                else:
                    logger.info("Features found.")
                    logger.debug(f"Found: {found}.\nMissing {missing}")

            logger.info(f"Done with preparation of {subject_data.subject} {subject_data.session}")

    if ("learn" in stages) or ("learning_curve" in stages) or ("permutation_importance" in stages):
        if learning_out_subdir:
            learning_dir = learning_dir / learning_out_subdir
        learning_dir.mkdir(parents=True, exist_ok=True)
        feature_base_dir = output_dir / "features"
        cache_dir = learning_dir / "cache"

        logger.info("Loading features")

        for modality in modalities:
            logger.info(modality)
            X = load_features_cached(cache_dir, participant_label, feature_base_dir,
                                     clinical_feature_file, modality)
            y = load_and_filter_df(target_file, participant_label)
            logger.info("Loading features. Done")

            run_gs = False if "ridgecv" in model_type else True
            if "learn" in stages:
                learn(X, y, learning_dir, model_name, modality, model_type=model_type, run_gs=run_gs,
                      n_splits=n_splits, weight_samples=weight_samples, n_jobs_outer=n_jobs_outer,
                      save_full_learning_info=save_full_learning_info)

            if "learning_curve" in stages:
                run_learning_curve(X, y, learning_dir, model_name, modality, model_type,
                                   best_parameters=best_cv_parameters_lcurve)

            if "permutation_importance" in stages:
                run_permutation_importance(X, y, learning_dir, model_name, modality, n_splits=n_splits,
                                           n_jobs_outer=n_jobs_outer)
        logger.info("Learning done.")
