from pathlib import Path

import pandas as pd
from tempfile import TemporaryDirectory
import os
import shutil
import subprocess
from bids import BIDSLayout

import logging
import nibabel as nib
from dataclasses import dataclass, field
import pprint

logger = logging.getLogger(__name__)


def run_cmd(cmd, env={}, output_to_file=None):
    logger.info(f"RUNNING {cmd}")

    merged_env = os.environ
    merged_env.update(env)
    if output_to_file:
        logger.info(f"Writing output to {output_to_file} .stdout/.stderr")
        stdout_file = Path(output_to_file).with_suffix(".stdout")
        stderr_file = Path(output_to_file).with_suffix(".stderr")
        stdout_file.parent.mkdir(exist_ok=True, parents=True)
        with open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
            subprocess.run(cmd, shell=True, check=True, env=merged_env, stdout=stdout, stderr=stderr)

        # remove empty files
        for f in [stderr_file, stderr_file]:
            if f.stat().st_size == 0:
                f.unlink()

    else:
        try:
            subprocess.run(cmd, shell=True, check=True, env=merged_env, capture_output=True)
        except subprocess.CalledProcessError as err:
            raise Exception(err.stderr)


def get_subject_in_data(layout, subject, session, check_duration=True):
    """"
    Returns files for T1w and resting-state functional images.
    Raises if no T1w or functional images
    check_duration=False can be used for testing
    """
    MIN_FMRI_DURATION_SEC = 120
    # check for T1w
    t1w_files = layout.get(subject=subject,
                           session=session,
                           extension=["nii", "nii.gz"],
                           suffix='T1w',
                           return_type='file')
    t1w_files = [Path(f) for f in t1w_files]

    func_files_all = layout.get(subject=subject,
                                session=session,
                                extension=["nii", "nii.gz"],
                                task="rest",
                                suffix='bold',
                                return_type='file')

    # check that fmri run duration in sufficient and return filename with tr
    func_runs = {}
    for f in func_files_all:
        # the entities in one line gives an error: sqlalchemy.exc.InvalidRequestError: stale association proxy,
        # parent object has gone out of scope
        fi = layout.get_file(str(f))
        if check_duration:
            tr = fi.entities['RepetitionTime']
            img = nib.load(str(f))
            n_volumes = img.shape[3]
            duration_sec = n_volumes * tr

            assert tr > 0, 'tr not > 0'
            assert n_volumes > 0, 'n_volumes not > 0'
        else:
            tr = None

        if check_duration:
            if duration_sec > MIN_FMRI_DURATION_SEC:
                # if run is not a key, there's just one scan in the session
                func_runs[fi.entities.get('run', 1)] = {'fmri_file': Path(f), 'tr': tr}
        else:
            func_runs[fi.entities.get('run', 1)] = {'fmri_file': Path(f), 'tr': tr}

    if not t1w_files:
        raise Exception(f"No T1w images found. Stopping. {subject} {session}")
    if not func_runs:
        raise Exception(f"No fmri images found. Stopping. {subject} {session}")
    return {"t1w": t1w_files, "func_runs": func_runs}


def check_subject(layout, subject, session):
    """
    Checks if subject is part of bids dir and if subject has session
    Raises if one is not the case
    """
    if subject not in layout.get_subjects():
        raise RuntimeError(f"Subject not found in input directory {subject}")

    if session not in layout.get_sessions(subject=subject):
        raise RuntimeError(f"Subject {subject} does not have session {session}")


def bids_to_tmp(bids_dir, subjects, session_mapping):
    """
    links bids-sessions into temp dir
    """
    tmp_dir_obj = TemporaryDirectory()
    tmp_bids_dir = Path(tmp_dir_obj.name)
    logger.debug(f"Linking selected bids sessions to {tmp_bids_dir}")

    dataset_file = bids_dir / "dataset_description.json"
    os.symlink(dataset_file, tmp_bids_dir / dataset_file.name)

    for subject in subjects:
        session = session_mapping[subject]
        target_subject_dir = tmp_bids_dir / f"sub-{subject}"
        target_subject_dir.mkdir()
        source_session_dir = bids_dir / f"sub-{subject}" / f"ses-{session}"
        logger.debug(f"copy {source_session_dir}")
        os.symlink(source_session_dir, target_subject_dir / source_session_dir.name, target_is_directory=True)

    return tmp_dir_obj, tmp_bids_dir


@dataclass
class SubjectData:
    layout: BIDSLayout
    subject: str
    session: str
    output_dir: Path
    check_duration: bool = True

    tmp_bids_dir: Path = field(init=False)
    in_data: dict = field(init=False)
    preprocessing_dir: Path = field(init=False)
    preprocessing_crashed_dir: Path = field(init=False)
    fmriprep_logfile_stub: Path = field(init=False)
    fmriprep_dir: Path = field(init=False)
    fmriprep_crashed_dir: Path = field(init=False)
    feature_dir: Path = field(init=False)
    preprocessed_files: dict = field(init=False)
    features_files: dict = field(init=False)

    def __post_init__(self):
        self.tmp_bids_dir = Path(self.layout.root)
        self.in_data = get_subject_in_data(self.layout, self.subject, self.session, self.check_duration)

        self.preprocessing_dir = Path(self.output_dir) / "preprocessing"
        self.preprocessing_crashed_dir = Path(self.output_dir) / "preprocessing_crashed"
        self.fmriprep_logfile_stub = self.preprocessing_dir / f"logs/sub-{self.subject}/sub-{self.subject}_fmriprep"
        self.fmriprep_dir = self.preprocessing_dir / f"sub-{self.subject}"
        self.fmriprep_crashed_dir = self.preprocessing_crashed_dir / f"sub-{self.subject}"
        self.feature_dir = Path(self.output_dir) / f"features/sub-{self.subject}/ses-{self.session}"
        self.get_expected_files()

    def __str__(self):
        d = self.__reduce__()[2]
        del d["layout"]
        return pprint.pformat(d, indent=4)

    def get_expected_files(self):
        # files that are expected after preprocessing
        # preprocessed_files = {"modality1":
        #                           {"run1":
        #                                {"file1": "filename1",
        #                                 "file2": "filename2",
        #                                 }
        #                            }
        #                       }
        self.preprocessed_files = {"fs": {"runx": {}}, "fmri": {}}
        self.preprocessed_files["fs"]["runx"] = {"done": (self.fmriprep_dir /
                                                          f"freesurfer/sub-{self.subject}/scripts/recon-all.done")}

        for run, func_info in self.in_data["func_runs"].items():
            # preprocessed functional files
            f = func_info["fmri_file"]
            out_filename = f.name[:f.name.find("_bold.nii")] + "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            out_filepath_fmri = (self.fmriprep_dir /
                                 f"fmriprep/sub-{self.subject}/ses-{self.session}/func/" / out_filename)

            # mask file
            out_filename = f.name[:f.name.find("_bold.nii")] + "_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
            out_filepath_mask = (self.fmriprep_dir /
                                 f"fmriprep/sub-{self.subject}/ses-{self.session}/func/" / out_filename)

            # confounds file
            out_filename = f.name[:f.name.find("_bold.nii")] + "_desc-confounds_regressors.tsv"
            out_filepath_confounds = (self.fmriprep_dir /
                                      f"fmriprep/sub-{self.subject}/ses-{self.session}/func/" / out_filename)

            self.preprocessed_files["fmri"][run] = {"fmri_file": out_filepath_fmri,
                                                    "mask_file": out_filepath_mask,
                                                    "confounds_file": out_filepath_confounds}

        # files that are expected after feature extraction
        self.features_files = {"fs": [self.feature_dir / f"sub-{self.subject}_ses-{self.session}_desc-fsFeatures.pkl"],
                               "fmri": [
                                   (self.feature_dir / f"sub-{self.subject}_ses-{self.session}_desc-fmriFeatures.pkl"),
                                   (self.feature_dir /
                                    f"sub-{self.subject}_ses-{self.session}_desc-fmriFullFeatures.pkl")
                               ]}

    def create_fmriprep_tmp_outdir(self):
        """
        creates tmpdir and links specified session into tempdir (since fmriprep does not allow
        to select sessions)
        """
        # create local output folder
        self.tmp_output_obj = TemporaryDirectory()  # self.tmp_output_obj.cleanup for tear down
        self.tmp_output_dir = Path(self.tmp_output_obj.name)
        self.tmp_fmriprep_dir = self.tmp_output_dir / "fmriprep"

    def check_preprocessing(self, raise_on_missing=True):
        preprocessed_files = []
        for mod in self.preprocessed_files.keys():
            for run in self.preprocessed_files[mod].keys():
                for file in self.preprocessed_files[mod][run].keys():
                    preprocessed_files.append(self.preprocessed_files[mod][run][file])

        found = []
        missing = []
        for f in preprocessed_files:
            if f.is_file():
                found.append(f)
            else:
                missing.append(f)

        if raise_on_missing and missing:
            msg = f"Expected {len(preprocessed_files)} files, but only {len(found)} found with " \
                f"{len(missing)} missing. {self.subject}, {self.session}.\n" \
                f"Found: {found}\n" \
                f"Missing: {missing}."
            raise RuntimeError(msg)
        return found, missing

    def compile_fmriprep_cmd(self, fs_license_file, test_run, n_cpus):
        sloppy = "--sloppy" if test_run else ""
        self.fmriprep_cmd = (f"conda run -n fmriprep_env "
                             f"fmriprep "
                             f"{str(self.tmp_bids_dir)} "
                             f"{str(self.tmp_fmriprep_dir)} "
                             f"participant "
                             f"--resource-monitor "
                             f"--fs-license-file "
                             f"{fs_license_file} "
                             f"--n_cpus {n_cpus} "
                             f"--skip_bids_validation --notrack {sloppy}")

    def run_fmriprep(self, fs_license_file, test_run, n_cpus):
        """
        fmriprep and dependencies are in the fmriprep_env conda env
        raw data of the relevant sessions is linked into a tempfolder and the fmriprep output is saved locally
        during execution (see https://github.com/poldracklab/smriprep/issues/44) and copied to the mounted output
        folder after processing
        """
        self.create_fmriprep_tmp_outdir()
        logger.debug(f"Create tmp fmriprep output dir {self.tmp_output_dir}")

        self.compile_fmriprep_cmd(fs_license_file, test_run, n_cpus)

        try:
            run_cmd(self.fmriprep_cmd, output_to_file=self.fmriprep_logfile_stub)
        except:
            logger.warning(f"Fmriprep 1st run failed. Try again.")
            try:
                run_cmd(self.fmriprep_cmd, output_to_file=self.fmriprep_logfile_stub)
            except:
                logger.warning(f"Fmriprep 2nd run failed. Try again with 1 cpu")
                self.compile_fmriprep_cmd(fs_license_file, test_run, n_cpus=1)
                try:
                    run_cmd(self.fmriprep_cmd, output_to_file=self.fmriprep_logfile_stub)
                    logger.warning("Fmriprep 3rd run successful")
                except:
                    logger.warning("Fmriprep 3rd run failed as well")
                    self.save_fmriprep_outputs(failed=True)
                    raise Exception("Fmriprep failed.")

        logger.debug(f"Saving fmriprep output")
        self.save_fmriprep_outputs(failed=False)
        logger.info("Fmriprep done")

        _, _ = self.check_preprocessing()

    def save_fmriprep_outputs(self, failed=False):
        output_dir = self.fmriprep_crashed_dir if failed else self.fmriprep_dir
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        fsav = (self.tmp_fmriprep_dir / "freesurfer").glob("fsaverage*")
        for f in fsav:
            shutil.rmtree(f)

        # shutil.copytree gives permission error
        logger.info(f"Copy {self.tmp_fmriprep_dir} to {output_dir}")
        run_cmd(f"cp -r {self.tmp_fmriprep_dir} {output_dir}{os.sep}")
        logger.info(f"Copy done")

    def teardown_tmp_dirs(self):
        self.tmp_output_obj.cleanup()

    def check_features(self, raise_on_missing=True):
        found = []
        missing = []
        for mod in self.features_files.keys():
            for f in self.features_files[mod]:
                if f.is_file():
                    found.append(f)
                else:
                    missing.append(f)

        if raise_on_missing and missing:
            msg = f"Expected {self.features_files}, but only {found} found with " \
                f"{len(missing)} missing. {self.subject}, {self.session}.\n" \
                f"Found: {found}\n" \
                f"Missing: {missing}."
            raise RuntimeError(msg)
        return found, missing


def remove_sub_ses_prefixes(df, subject_col='participant_id', session_col='session_id'):
    df_ = df.copy()
    if subject_col:
        df_[subject_col] = df_[subject_col].str.replace("sub-", "")
    if session_col:
        df_[session_col] = df_[session_col].str.replace("ses-", "")
    return df_


def get_baseline_session_mapping(subjects, session, baseline_sessions_file):
    """
    :param subjects: list of subjects
    :param session: session (only works if one subject specified)
    :param baseline_sessions_file: tsv file that has a subjects session mapping like
          participant_id session_id
          sub-xx1     ses-s1
          sub-xx2     ses-s2
    :return: dict:
    """
    baseline_sessions_file = Path(baseline_sessions_file) if baseline_sessions_file else baseline_sessions_file

    if session:
        assert len(subjects) == 1, f"more than one subjects specified {subjects}"
        session_mapping = {subjects[0]: session}
    else:
        assert baseline_sessions_file.suffix == ".tsv", f"Not a .tsv file {baseline_sessions_file}"
        df = pd.read_csv(baseline_sessions_file, sep="\t")
        df = remove_sub_ses_prefixes(df)
        df.set_index("participant_id", inplace=True)
        session_mapping = df.to_dict()['session_id']
    return session_mapping


def get_subjects_in_dir(bids_dir):
    subjects = [p.name.lstrip("sub-") for p in bids_dir.glob("sub-*")]
    return subjects
