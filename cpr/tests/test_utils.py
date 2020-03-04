import pytest
from pathlib import Path
from bids import BIDSLayout
from ..utils import (get_subject_in_data, check_subject, SubjectData, remove_sub_ses_prefixes,
                     get_baseline_session_mapping)
import pandas as pd
import tempfile
from time import time
import os

test_data_dir = BIDSLayout(Path(__file__).parent / "test_data/skeleton/bids", index_metadata=False)
test_data_out_dir_full = Path(__file__).parent / "test_data/skeleton/output"
test_data_out_dir_fs_missing = Path(__file__).parent / "test_data/skeleton/output_fs_missing"
test_data_out_dir_bold_missing = Path(__file__).parent / "test_data/skeleton/output_bold_missing"


@pytest.fixture
def get_test_layout():
    layout = test_data_dir
    return layout


# tests for get_subject_in_data()
def test_full_dataset(get_test_layout):
    layout = get_test_layout
    subject = "OAS30001"
    session = "d0129"
    subject_data = get_subject_in_data(layout, subject, session, check_duration=False)
    assert len(subject_data["t1w"]) == 2
    assert len(subject_data["func_runs"]) == 3


def test_missing_t1w_raise(get_test_layout):
    layout = get_test_layout
    subject = "OAS30002"
    session = "d2340"
    with pytest.raises(Exception) as e:
        subject_data = get_subject_in_data(layout, subject, session, check_duration=False)
    assert str(e.value).startswith("No T1w images found")


# tests for check_subject()
def test_missing_subject_ok(get_test_layout):
    layout = get_test_layout
    check_subject(layout, "OAS30001", "d0129")
    assert True


def test_missing_subject_raise(get_test_layout):
    layout = get_test_layout
    with pytest.raises(RuntimeError) as e:
        check_subject(layout, "OAS30099", "d0129")
    assert str(e.value).startswith("Subject not found in input directory OAS30099")


def test_missing_session_raise(get_test_layout):
    layout = get_test_layout
    with pytest.raises(RuntimeError) as e:
        check_subject(layout, "OAS30001", "d999")
    assert str(e.value).startswith("Subject OAS30001 does not have session d999")


# test SubjectData
@pytest.fixture
def get_subject():
    subject_data = SubjectData(test_data_dir, "OAS30001", "d0129", "/data/out", check_duration=False)
    return subject_data


def test_subject_init(get_subject):
    subject_data = get_subject
    assert subject_data.subject == "OAS30001"
    assert len(subject_data.preprocessed_files) == 2
    assert len(subject_data.preprocessed_files["fmri"]) == 3


def test_copy_subject(get_subject):
    subject_data = get_subject
    subject_data.create_fmriprep_tmp_outdir()

    in1 = {Path(f).name for f in subject_data.in_data["t1w"]}
    in2 = {Path(f).name for f in subject_data.in_data["t1w"]}
    assert in1 == in2

    in1 = {Path(f["fmri_file"]).name for f in subject_data.in_data["func_runs"].values()}
    in2 = {Path(f["fmri_file"]).name for f in subject_data.in_data["func_runs"].values()}
    assert in1 == in2


def test_copy_check_preprocessing():
    subject_data = SubjectData(test_data_dir, "OAS30001", "d0129", test_data_out_dir_full, False)
    found, missing = subject_data.check_preprocessing()
    assert len(found) == 10
    assert len(missing) == 0


def test_copy_check_preprocessing_raise_fs():
    subject_data = SubjectData(test_data_dir, "OAS30001", "d0129", test_data_out_dir_fs_missing, False)
    with pytest.raises(RuntimeError) as e:
        found, missing = subject_data.check_preprocessing()
    assert str(e.value).startswith(
        "Expected 10 files, but only 9 found with 1 missing")


def test_copy_check_preprocessing_raise_bold():
    subject_data = SubjectData(test_data_dir, "OAS30001", "d0129", test_data_out_dir_bold_missing, False)
    with pytest.raises(RuntimeError) as e:
        found, missing = subject_data.check_preprocessing()
    assert str(e.value).startswith(
        "Expected 10 files, but only 5 found with 5 missing")

    found, missing = subject_data.check_preprocessing(raise_on_missing=False)
    assert len(found) == 5
    assert len(missing) == 5


def test_copy_subject_teardown(get_subject):
    subject_data = get_subject
    subject_data.create_fmriprep_tmp_outdir()
    assert subject_data.tmp_output_dir.is_dir()

    subject_data.teardown_tmp_dirs()
    assert not subject_data.tmp_output_dir.is_dir()


def test_subject_fmriprep_cmd(get_subject):
    subject_data = get_subject
    subject_data.create_fmriprep_tmp_outdir()
    subject_data.compile_fmriprep_cmd("/fs_lic", test_run=False, n_cpus=2)
    assert subject_data.fmriprep_cmd == f"conda run -n fmriprep_env fmriprep {subject_data.tmp_bids_dir} " \
        f"{subject_data.tmp_output_dir}/fmriprep participant " \
        f"--resource-monitor --fs-license-file /fs_lic --n_cpus 2 --skip_bids_validation --notrack "

    subject_data.compile_fmriprep_cmd("/fs_lic", test_run=True, n_cpus=2)
    assert subject_data.fmriprep_cmd == f"conda run -n fmriprep_env fmriprep {subject_data.tmp_bids_dir} " \
        f"{subject_data.tmp_output_dir}/fmriprep participant " \
        f"--resource-monitor --fs-license-file /fs_lic --n_cpus 2 --skip_bids_validation --notrack " \
        f"--sloppy"


def test_remove_sub_ses_prefixes():
    df = pd.DataFrame({'participant_id': ['sub-xx1', 'sub-xx2'],
                       'session_id': ['ses-s1', 'ses-s2']})
    df_expected = pd.DataFrame({'participant_id': ['xx1', 'xx2'],
                                'session_id': ['s1', 's2']})
    df_out = remove_sub_ses_prefixes(df)
    pd.testing.assert_frame_equal(df_out, df_expected)


def test_remove_sub_ses_prefixes_noses():
    df = pd.DataFrame({'participant_id': ['sub-xx1', 'sub-xx2'],
                       'xxx': ['xxs1', 'xx-s2']})
    df_expected = pd.DataFrame({'participant_id': ['xx1', 'xx2'],
                                'xxx': ['xxs1', 'xx-s2']})
    df_out = remove_sub_ses_prefixes(df, session_col=None)
    pd.testing.assert_frame_equal(df_out, df_expected)


def test_remove_sub_ses_prefixes_noprefs():
    df = pd.DataFrame({'participant_id': ['xx1', 'xx2'],
                       'session_id': ['s1', 's2']})
    df_out = remove_sub_ses_prefixes(df)
    pd.testing.assert_frame_equal(df_out, df)


def test_get_baseline_session_mapping_single():
    session_mapping = get_baseline_session_mapping(subjects=['s1'], session='ses1', baseline_sessions_file=None)
    assert session_mapping == {'s1': 'ses1'}


def test_get_baseline_session_mapping_df():
    df = pd.DataFrame({'participant_id': ['sub-xx1', 'sub-xx2'],
                       'session_id': ['ses-s1', 'ses-s2']})
    temp_file = tempfile.NamedTemporaryFile(suffix='.tsv')
    df.to_csv(temp_file.name, sep="\t", index=False)

    session_mapping = get_baseline_session_mapping(subjects=None, session=None, baseline_sessions_file=temp_file.name)
    assert session_mapping == {'xx1': 's1', 'xx2': 's2'}
    temp_file.close()


def test_get_baseline_session_mapping_multisub_raise():
    with pytest.raises(AssertionError):
        session_mapping = get_baseline_session_mapping(subjects=['s1', 's2'], session='ses1',
                                                       baseline_sessions_file=None)
