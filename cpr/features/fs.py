import tempfile
from pathlib import Path
import logging
import pandas as pd

from ..utils import run_cmd

logger = logging.getLogger(__name__)


def extract_fs_features(subject_data):
    fs_dir = subject_data.fmriprep_dir / "freesurfer"
    dfs = collect_fs_tables(subject_data.subject, fs_dir)
    dfs = [df for df in dfs.values()]
    fs_df = pd.concat(dfs, axis=1)
    # remove sub- prefix from index
    fs_df.index = fs_df.index.str.lstrip("sub-")

    subject_data.feature_dir.mkdir(parents=True, exist_ok=True)
    out_file = subject_data.feature_dir / f"sub-{subject_data.subject}_ses-{subject_data.session}_desc-fsFeatures.pkl"
    fs_df.to_pickle(out_file)


def collect_fs_tables(subject, fs_dir):
    """collect freesurfer tables into four output tables
    - global: global volumes, ventricles, mean thickness?
    - subcortical volume (incl HC)
    - thickness
    - volume

    returns dict with dict_keys(['subcortVolume', 'globalVolume', 'cortThickness', 'cortVolume'])
    """

    dfs = {}
    aseg_df = get_aseg_df(subject, fs_dir)
    thickness_dfs = get_aparc_df(subject, fs_dir, "thickness")
    volume_dfs = get_aparc_df(subject, fs_dir, "volume")

    # subcortical
    c = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper',
         'Left-Caudate', 'Right-Caudate',
         'Left-Putamen', 'Right-Putamen',
         'Left-Pallidum', 'Right-Pallidum',
         'Left-Hippocampus', 'Right-Hippocampus',
         'Left-Amygdala', 'Right-Amygdala',
         'Left-Accumbens-area', 'Right-Accumbens-area']
    dfs["subcortVolume"] = aseg_df[c]

    # global
    global_df = (thickness_dfs["lh"].
        join(thickness_dfs["rh"], lsuffix="ll", rsuffix="rr")[
        ['lh_MeanThickness_thickness', 'rh_MeanThickness_thickness']])

    c = ['3rd-Ventricle', '4th-Ventricle',
         'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle',
         'lhCortexVol', 'rhCortexVol',
         'lhCerebralWhiteMatterVol', 'rhCerebralWhiteMatterVol',
         'Left-Cerebellum-White-Matter', 'Right-Cerebellum-White-Matter',
         'Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex',
         'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior',
         'CC_Anterior',
         'SubCortGrayVol', 'TotalGrayVol']
    dfs["globalVolume"] = global_df.join(aseg_df[c])

    # cortical thickness
    dfs["cortThickness"] = (thickness_dfs["lh"].drop(
        columns=['lh_MeanThickness_thickness', 'BrainSegVolNotVent', 'eTIV']).join(thickness_dfs["rh"].drop(
        columns=['rh_MeanThickness_thickness', 'BrainSegVolNotVent', 'eTIV'])))

    # cortical volume
    dfs["cortVolume"] = (volume_dfs["lh"].drop(columns=['BrainSegVolNotVent', 'eTIV']).
                         join(volume_dfs["rh"].drop(columns=['BrainSegVolNotVent', 'eTIV'])))

    # format columns like 'fs__{metric}__{name}'
    for k in dfs.keys():
        cols = dfs[k].columns
        dfs[k].columns = [f'fs__{k}__{c}' for c in cols]

    return dfs


def get_aseg_df(subject, fs_dir):
    """
    Extracts aseg data (global & subcortical measures) for one subject.

    Values are scaled wrt eTIV.
    https://surfer.nmr.mgh.harvard.edu/fswiki/eTIV
    https://surfer.nmr.mgh.harvard.edu/fswiki/MorphometryStats

    :param subject: subject name (without "sub-" prefix)
    :param fs_dir: directory that contains the freesurfer subjects
    :return: data frame with subject as index
    """

    tmp_dir = tempfile.TemporaryDirectory()
    outfile = Path(tmp_dir.name) / "aseg.txt"
    cmd = f"python2 `which asegstats2table` " \
        f"-s sub-{subject} " \
        f"-t {outfile} --etiv"

    run_cmd(cmd, env={"SUBJECTS_DIR": str(fs_dir)})

    df = pd.read_csv(outfile, sep="\t")
    # format df
    df.rename(columns={"Measure:volume": "subject"}, inplace=True)
    df.set_index("subject", inplace=True)

    tmp_dir.cleanup()
    return df


def get_aparc_df(subject, fs_dir, measure):
    """
    Extracts aparc.a2009s data (cortical measures) for one subject.

    volume values are scaled wrt eTIV, thickness values are not
    https://surfer.nmr.mgh.harvard.edu/fswiki/eTIV

    :param subject: subject name (without "sub-" prefix)
    :param fs_dir: directory that contains the freesurfer subjects
    :param measure: ["thickness", "volume"]
    :return: dict of data frames with keys ["lh", "rh"]; with subject as index
    """

    if measure not in ["thickness", "volume"]:
        raise NotImplementedError(f"Measure not implemented: {measure}")

    tmp_dir = tempfile.TemporaryDirectory()
    dfs = {}
    for hemi in ["lh", "rh"]:
        outfile = Path(tmp_dir.name) / f"{hemi}_{measure}.txt"
        cmd = f"python2 `which aparcstats2table` " \
            f"-s sub-{subject} " \
            f"-t {outfile} " \
            f"-m {measure} " \
            f"--hemi {hemi} " \
            f"-p aparc.a2009s"

        # norm volume against etiv
        if measure == "volume":
            cmd += " --etiv"

        run_cmd(cmd, env={"SUBJECTS_DIR": str(fs_dir)})

        df = pd.read_csv(outfile, sep="\t")
        # format df
        df.rename(columns={f"{hemi}.aparc.a2009s.{measure}": "subject"}, inplace=True)
        df.set_index("subject", inplace=True)
        df.columns = [c.replace("&", "_") for c in df.columns]
        dfs[hemi] = df
    tmp_dir.cleanup()
    return dfs
