import numpy as np
import pandas as pd
from nilearn import input_data, datasets, connectome, image
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def get_confounds(confounds_file):
    """
    :param confounds_file: fmriprep confounds file (.tsv)
    :return: dataframe with 36 confounds parameters

    takes a fmriprep confounds file and creates data frame with
    Satterthwaite's 36P confound regressors.

    Satterthwaite, T. D., Elliott, M. A., Gerraty, R. T., Ruparel, K.,
    Loughead, J., Calkins, M. E., et al. (2013).
    An improved framework for confound regression and filtering for control of
    motion artifact in the preprocessing of resting-state functional
    connectivity data. NeuroImage, 64, 240–256.
    http://doi.org/10.1016/j.neuroimage.2012.08.052
    """

    df = pd.read_csv(confounds_file, sep="\t")

    p9 = df[['csf', 'white_matter', 'global_signal',
             'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]

    p9_der = p9.diff().fillna(0)
    p9_der.columns = [c + "_der" for c in p9_der.columns]
    p18 = pd.concat((p9, p9_der), axis=1)
    p18_2 = p18 ** 2
    p18_2.columns = [c + "_2" for c in p18_2.columns]
    confounds = pd.concat((p18, p18_2), axis=1)

    return confounds


def nan_empty_rois_in_conmat(conmat, emtpy_ind):
    for i in emtpy_ind:
        conmat[:, i] = np.nan
        conmat[i, :] = np.nan
    return conmat


def extract_full_connectivity_matrix(fmri_file, mask_file, confounds_file, tr):
    confounds = get_confounds(confounds_file)

    masker_pars = {"mask_img": str(mask_file),
                   "detrend": True,
                   "standardize": True,
                   "low_pass": 0.1,
                   "high_pass": 0.01,
                   "t_r": tr,
                   "smoothing_fwhm": 6,
                   "radius": 5,
                   "allow_overlap": True
                   }

    atlas = datasets.fetch_coords_seitzman_2018()
    network_rois_ind = atlas.networks != "unassigned"

    coords = np.vstack((atlas.rois['x'],
                        atlas.rois['y'],
                        atlas.rois['z']
                        )).T
    coords = coords[network_rois_ind]
    time_series, excluded_rois = extract_time_series(masker_pars, fmri_file, confounds.values, coords)

    correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
    connectivity_matrix = correlation_measure.fit_transform([time_series])[0]
    connectivity_matrix = nan_empty_rois_in_conmat(connectivity_matrix, excluded_rois.keys())

    return connectivity_matrix, atlas.networks[network_rois_ind], excluded_rois


def extract_time_series(masker_pars: dict, fmri_file: Path, confounds: np.array, coords: np.array):
    """
    If standard extraction fails with Sphere around seed i is empty, try for each roi separately
    """
    excluded_rois = {}
    masker = input_data.NiftiSpheresMasker(seeds=coords, **masker_pars)

    try:
        time_series = masker.fit_transform(str(fmri_file), confounds=confounds)
    except ValueError as err:
        if err.args[0].startswith("Sphere around seed "):
            # try each coord separaetly
            time_series, excluded_rois = extract_time_series_empty(masker_pars, fmri_file, confounds, coords)
        else:
            raise err
    return time_series, excluded_rois


def extract_time_series_empty(masker_pars: dict, fmri_file: Path, confounds: np.array, coords: np.array):
    # if rois are empty replace coords with mid-brain-coords (where we know is signal) and replace in timeseries with 0
    max_empty_rois = 10
    excluded_roi_ind = []
    time_series = []
    orig_coords = coords.copy()

    while len(excluded_roi_ind) <= max_empty_rois:
        masker = input_data.NiftiSpheresMasker(seeds=coords, **masker_pars)
        try:
            time_series = masker.fit_transform(str(fmri_file), confounds=confounds)
            break
        except ValueError as err:
            if err.args[0].startswith("Sphere around seed "):
                empty_roi_ind = int(err.args[0].lstrip("Sphere around seed #").rstrip(" is empty"))
                logger.debug(f"ROI {empty_roi_ind} {coords[empty_roi_ind]} empty")
                excluded_roi_ind.append(empty_roi_ind)
                coords[empty_roi_ind] = np.array([0, 0, 0])
            else:
                raise err
    if len(excluded_roi_ind) > max_empty_rois:
        raise RuntimeError(f"More than {max_empty_rois} are empty. Stopping.")

    n_vols = time_series.shape[0]
    for i in excluded_roi_ind:
        time_series[:, i] = np.zeros((n_vols,))
    excluded_rois = {i: list(orig_coords[i]) for i in excluded_roi_ind}
    return time_series, excluded_rois


def connectivity_matrix_to_df(raw_mat, roi_names):
    """
    takes a connectivity matrix (e.g., numpy array) and a list of
    roi_names (strings) returns data frame with roi_names as index and
    column names
    e.g.
         r1   r2   r3   r4
    r1  0.0  0.3  0.7  0.2
    r2  0.3  0.0  0.6  0.5
    r3  0.7  0.6  0.0  0.9
    r4  0.2  0.5  0.9  0.0
    """
    con_df = pd.DataFrame(raw_mat, index=roi_names, columns=roi_names)
    return con_df


def _sort_rois(a, b):
    l = [a, b]
    l.sort()
    return pd.Series(l, index=['roi1', 'roi2'])


def get_df_utri_long(df, col_name='fc'):
    """
    takes symmetrical data frame
    only retains upper diagonal and transforms to long format
    returns data frame that looks like:
            roi1   roi2  fc
    0       r1     r2   0.3
    1       r1     r3   0.7
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # sanity check if matrix is symmetrical
    df_test = df.copy()
    df_test.fillna(0, inplace=True)
    assert np.allclose(df_test, df_test.T), "matrix not symmetrical"
    del df_test

    # index for upper triangle
    ind_df = pd.DataFrame(np.triu(np.ones(df.shape), 1).astype(np.bool), index=df.index, columns=df.columns)
    ind_df_long = ind_df.stack().reset_index()
    ind_df_long.columns = ['roi1', 'roi2', 'keep']

    df_long = df.stack(dropna=False).reset_index()
    df_long = df_long.loc[ind_df_long.keep].reset_index(drop=True)
    df_long.columns = ['roi1', 'roi2', col_name]

    # since we have undirected connections, we sort the rois so that we ensure the pairing treats r1->r2 the same as
    # r2->r1
    sorted_rois = df_long.apply(lambda row: _sort_rois(row['roi1'],
                                                       row['roi2']), axis=1)
    df_long = pd.concat((df_long.drop(columns=['roi1', 'roi2']), sorted_rois), axis=1)

    return df_long[['roi1', 'roi2', col_name]]


def downsample_to_networks(df):
    """
    :param df: long dataframe with columns ['roi1', 'roi2'] and a third column with fc values
    :return: dataframe with mean values per roi-pair

    e.g.,
    df:
            r1   r2   r3   r1
        r1  1.0  0.3  0.7  0.2
        r2  0.3  1.0  0.6  0.5
        r3  0.7  0.6  1.0  0.9
        r1  0.2  0.5  0.9  1.0

    returns:
          roi1 roi2   fc
        0   r1   r1  0.2
        1   r1   r2  0.4
        2   r1   r3  0.8
        3   r2   r3  0.6

    """
    mean_networks = df.groupby(['roi1', 'roi2']).mean().reset_index()
    mean_networks['roi'] = mean_networks[['roi1', 'roi2']].apply(lambda x: '_'.join(x), axis=1)
    mean_networks['tmp'] = 'tmp'
    mean_networks_wide = mean_networks.pivot(index='tmp', columns='roi', values='fc').reset_index(drop=True)
    mean_networks_wide.columns.name = None
    return mean_networks_wide


def extract_connectivity_features(subject_data):
    subject_data.feature_dir.mkdir(parents=True, exist_ok=True)

    func_info = subject_data.preprocessed_files["fmri"]
    for run in func_info.keys():
        func_info[run]["tr"] = subject_data.in_data["func_runs"][run]["tr"]

    con_mats = []
    excluded_rois = {}
    for run, info in func_info.items():
        logger.info(f"Extracting conmat for {subject_data.subject}, {subject_data.session}, {run}")
        con_mat, networks, excluded_rois[run] = extract_full_connectivity_matrix(info["fmri_file"], info["mask_file"],
                                                                                 info["confounds_file"], info["tr"])
        # fishers z
        np.fill_diagonal(con_mat, 0)
        con_mat_z = np.arctanh(con_mat)
        con_mats.append(con_mat_z)

    # average multiple runs
    con_mats = np.array(con_mats)
    mean_con_mat = con_mats.mean(0)
    matrix_df = connectivity_matrix_to_df(mean_con_mat, networks)

    # extract upper triangle and downsample to network level
    matrix_long = get_df_utri_long(matrix_df)
    mean_networks = downsample_to_networks(matrix_long)

    # format
    mean_networks.columns = [f"fmri__{c}" for c in mean_networks.columns]
    mean_networks["subject"] = subject_data.subject
    mean_networks.set_index("subject", drop=True, inplace=True)

    # prepare for fullcon
    roi_ind = [f"{n}_{i}" for i, n in enumerate(networks)]
    matrix_df_full = connectivity_matrix_to_df(mean_con_mat, roi_ind)
    matrix_long_full = get_df_utri_long(matrix_df_full)
    matrix_long_full["roi"] = "fmri__" + matrix_long_full["roi1"] + "__" + matrix_long_full["roi2"]
    matrix_long_full.drop(columns=["roi1", "roi2"], inplace=True)
    matrix_long_full["subject"] = subject_data.subject
    fullcon_df = matrix_long_full.pivot(index='subject', columns='roi', values='fc')
    fullcon_df.columns.name = None
    fullcon_df = fullcon_df.fillna(0)

    # save
    out_file = (subject_data.feature_dir /
                f"sub-{subject_data.subject}_ses-{subject_data.session}_desc-fmriFeatures.pkl")
    mean_networks.to_pickle(out_file)

    out_file = (subject_data.feature_dir /
                f"sub-{subject_data.subject}_ses-{subject_data.session}_desc-fmriFullFeatures.pkl")
    fullcon_df.to_pickle(out_file)

    out_file = (subject_data.feature_dir /
                f"sub-{subject_data.subject}_ses-{subject_data.session}_desc-conMats.npy")
    np.save(out_file, con_mats)

    out_file = (subject_data.feature_dir /
                f"sub-{subject_data.subject}_ses-{subject_data.session}_desc-excludedRois.json")
    with open(out_file, 'w') as fi:
        json.dump(excluded_rois, fi, indent=4)
