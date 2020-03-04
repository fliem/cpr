import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from bids import BIDSLayout
from ..features.fs import get_aseg_df, collect_fs_tables
from ..features.fmri import get_confounds, connectivity_matrix_to_df, get_df_utri_long, downsample_to_networks, \
    extract_time_series, extract_time_series_empty
from nilearn import datasets
from nilearn.masking import compute_epi_mask
from .feature_columns import expected_columns

output_dir = str(Path(__file__).parent / "test_data/preprocessed/freesurfer")
fs_dir = Path(output_dir, f"preprocessing/fmriprep/freesurfer")
fmriprep_dir = Path(__file__).parent / "test_data/fmriprep"


# tests/test_data/preprocessed/freesurfer/preprocessed/fmriprep/freesurfer/sub-01/stats

def test_get_aseg_df():
    df = get_aseg_df("01", fs_dir)
    assert df.shape == (1, 66)


def test_collect_fs_tables():
    dfs = collect_fs_tables("01", fs_dir)
    for k in dfs:
        assert dfs[k].columns.tolist() == expected_columns[k]
        assert dfs[k].shape == (1, len(expected_columns[k]))


def test_get_confounds():
    f = fmriprep_dir / 'sub-01_ses-s1_task-rest_desc-confounds_regressors.tsv'
    df = get_confounds(f)
    assert df.shape == (5, 36)


def test_extract_time_series():
    data = datasets.fetch_development_fmri(1)
    fmri_file = data.func[0]
    mask_img = compute_epi_mask(fmri_file)
    masker_pars = {"mask_img": mask_img,
                   "detrend": True,
                   "standardize": True,
                   "smoothing_fwhm": 6,
                   "radius": 5,
                   "allow_overlap": True
                   }
    atlas = datasets.fetch_coords_seitzman_2018()
    coords = np.vstack((atlas.rois['x'],
                        atlas.rois['y'],
                        atlas.rois['z']
                        )).T
    coords = coords[:4]
    time_series, excluded_rois = extract_time_series(masker_pars, fmri_file, None, coords)
    time_series1, excluded_rois1 = extract_time_series_empty(masker_pars, fmri_file, None, coords)

    assert excluded_rois == {}
    assert excluded_rois1 == {}
    assert time_series.shape == (168, 4)
    assert time_series1.shape == (168, 4)
    assert np.allclose(time_series, time_series1)


def test_extract_time_series_empty_rois():
    data = datasets.fetch_development_fmri(1)
    fmri_file = data.func[0]
    mask_img = compute_epi_mask(fmri_file)
    masker_pars = {"mask_img": mask_img,
                   "detrend": True,
                   "standardize": True,
                   "smoothing_fwhm": 6,
                   "radius": 5,
                   "allow_overlap": True
                   }
    atlas = datasets.fetch_coords_seitzman_2018()
    coords = np.vstack((atlas.rois['x'],
                        atlas.rois['y'],
                        atlas.rois['z']
                        )).T
    coords = coords[:4]
    coords[0] = [-90, -90, -90]
    coords[1] = [-90, -90, -90]
    time_series, excluded_rois = extract_time_series(masker_pars, fmri_file, None, coords)

    assert excluded_rois == {0: [-90., -90., -90.], 1: [-90., -90., -90.]}
    assert (time_series[:, 0] == 0).all()
    assert (time_series[:, 1] == 0).all()
    assert not (time_series[0] == 0).all()
    assert not (time_series[1] == 0).all()


def test_connectivity_matrix_to_df():
    raw_mat = np.array([(1, .3, .7, .2),
                        (.3, 1, .6, .5),
                        (.7, .6, 1, .9),
                        (.2, .5, .9, 1),
                        ])
    roi_names = ["r1", "r2", "r3", "r4"]

    df = connectivity_matrix_to_df(raw_mat, roi_names)

    assert df.shape == (4, 4), "shape issue"
    assert (df.columns == roi_names).all(), "name issue"
    assert isinstance(df, pd.DataFrame), "type issue"


def test_matrix_long_utri():
    raw_mat = np.array([(1, .3, .7, .2),
                        (.3, 1, .6, .5),
                        (.7, .6, 1, .9),
                        (.2, .5, .9, 1),
                        ])
    roi_names = ["r1", "r2", "r3", "r4"]
    conmat_df_orig = connectivity_matrix_to_df(raw_mat, roi_names)
    conmat_df_long = get_df_utri_long(conmat_df_orig)

    expected_df = pd.DataFrame([['r1', 'r2', 0.3],
                                ['r1', 'r3', 0.7],
                                ['r1', 'r4', 0.2],
                                ['r2', 'r3', 0.6],
                                ['r2', 'r4', 0.5],
                                ['r3', 'r4', 0.9]],
                               columns=['roi1', 'roi2', 'fc'])
    pd.testing.assert_frame_equal(expected_df, conmat_df_long)


def test_matrix_long_utri_nan():
    raw_mat = np.array([(1, np.nan, .7, .2),
                        (np.nan, np.nan, np.nan, np.nan),
                        (.7, np.nan, 1, .9),
                        (.2, np.nan, .9, 1),
                        ])
    roi_names = ["r1", "r2", "r3", "r4"]
    conmat_df_orig = connectivity_matrix_to_df(raw_mat, roi_names)
    conmat_df_long = get_df_utri_long(conmat_df_orig)

    expected_df = pd.DataFrame([['r1', 'r2', np.nan],
                                ['r1', 'r3', 0.7],
                                ['r1', 'r4', 0.2],
                                ['r2', 'r3', np.nan],
                                ['r2', 'r4', np.nan],
                                ['r3', 'r4', 0.9]],
                               columns=['roi1', 'roi2', 'fc'])
    pd.testing.assert_frame_equal(expected_df, conmat_df_long)


def test_matrix_long_utri_sorting():
    # roi names are sorted
    raw_mat = np.array([(1, .3, .7, .2),
                        (.3, 1, .6, .5),
                        (.7, .6, 1, .9),
                        (.2, .5, .9, 1),
                        ])
    roi_names = ["r1", "r2", "r3", "r1"]
    conmat_df_orig = connectivity_matrix_to_df(raw_mat, roi_names)
    conmat_df_long = get_df_utri_long(conmat_df_orig)

    expected_df = pd.DataFrame([['r1', 'r2', 0.3],
                                ['r1', 'r3', 0.7],
                                ['r1', 'r1', 0.2],
                                ['r2', 'r3', 0.6],
                                ['r1', 'r2', 0.5],
                                ['r1', 'r3', 0.9]],
                               columns=['roi1', 'roi2', 'fc'])
    pd.testing.assert_frame_equal(expected_df, conmat_df_long)


def test_matrix_long_utri_nonsym():
    # roi names are sorted
    raw_mat = np.array([(1, 999, .7, .2),
                        (.3, 1, .6, .5),
                        (.7, .6, 1, .9),
                        (.2, .5, .9, 1),
                        ])
    roi_names = ["r1", "r2", "r3", "r1"]
    conmat_df_orig = connectivity_matrix_to_df(raw_mat, roi_names)
    with pytest.raises(Exception):
        conmat_df_long = get_df_utri_long(conmat_df_orig)


def test_downsample_to_networks():
    raw_mat = np.array([(1, .3, .7, .2),
                        (.3, 1, .6, .5),
                        (.7, .6, 1, .9),
                        (.2, .5, .9, 1),
                        ])
    roi_names = ["r1", "r2", "r3", "r1"]
    conmat_df_orig = connectivity_matrix_to_df(raw_mat, roi_names)
    conmat_df_long = get_df_utri_long(conmat_df_orig)
    downsampled_df = downsample_to_networks(conmat_df_long)

    expected_df = pd.DataFrame([[0.2, 0.4, 0.8, 0.6]],
                               columns=['r1_r1', 'r1_r2', 'r1_r3', 'r2_r3'])
    df1, df2 = expected_df.align(downsampled_df)
    pd.testing.assert_frame_equal(df1, df2)
