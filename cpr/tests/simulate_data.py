from pathlib import Path
import os

#
# c = Path.cwd()
# l = c.rglob("sub-OAS30001/**/*")
# l = [str(p.relative_to(c)) for p in l]


out_dir = Path("test_data/skeleton/bids")
out_dir.mkdir(exist_ok=True, parents=True)

# full subject
l = ['sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-02_bold.json',
     'sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-03_bold.json',
     'sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-01_bold.json',
     'sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-02_bold.nii.gz',
     'sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-03_bold.nii.gz',
     'sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-01_bold.nii.gz',
     'sub-OAS30001/ses-d0129/anat/sub-OAS30001_ses-d0129_run-01_T1w.json',
     'sub-OAS30001/ses-d0129/anat/sub-OAS30001_ses-d0129_run-01_T1w.nii.gz',
     'sub-OAS30001/ses-d0129/anat/sub-OAS30001_ses-d0129_T2w.json',
     'sub-OAS30001/ses-d0129/anat/sub-OAS30001_ses-d0129_run-02_T1w.json',
     'sub-OAS30001/ses-d0129/anat/sub-OAS30001_ses-d0129_T2w.nii.gz',
     'sub-OAS30001/ses-d0129/anat/sub-OAS30001_ses-d0129_run-02_T1w.nii.gz',
     'sub-OAS30001/ses-d3132/func/sub-OAS30001_ses-d3132_task-rest_run-01_bold.nii.gz',
     'sub-OAS30001/ses-d3132/func/sub-OAS30001_ses-d3132_task-rest_run-02_bold.json',
     'sub-OAS30001/ses-d3132/func/sub-OAS30001_ses-d3132_task-rest_run-01_bold.json',
     'sub-OAS30001/ses-d3132/func/sub-OAS30001_ses-d3132_task-rest_run-02_bold.nii.gz',
     'sub-OAS30001/ses-d3132/anat/sub-OAS30001_ses-d3132_T1w.nii.gz',
     'sub-OAS30001/ses-d3132/anat/sub-OAS30001_ses-d3132_T1w.json',
     'sub-OAS30001/ses-d3132/anat/sub-OAS30001_ses-d3132_T2w.nii.gz',
     'sub-OAS30001/ses-d3132/anat/sub-OAS30001_ses-d3132_T2w.json',
     'sub-OAS30001/ses-d0757/func/sub-OAS30001_ses-d0757_task-rest_run-01_bold.nii.gz',
     'sub-OAS30001/ses-d0757/func/sub-OAS30001_ses-d0757_task-rest_run-01_bold.json',
     'sub-OAS30001/ses-d0757/func/sub-OAS30001_ses-d0757_task-rest_run-02_bold.json',
     'sub-OAS30001/ses-d0757/func/sub-OAS30001_ses-d0757_task-rest_run-02_bold.nii.gz',
     'sub-OAS30001/ses-d0757/anat/sub-OAS30001_ses-d0757_T2w.json',
     'sub-OAS30001/ses-d0757/anat/sub-OAS30001_ses-d0757_T2w.nii.gz',
     'sub-OAS30001/ses-d0757/anat/sub-OAS30001_ses-d0757_run-01_T1w.json',
     'sub-OAS30001/ses-d0757/anat/sub-OAS30001_ses-d0757_run-01_T1w.nii.gz',
     'sub-OAS30001/ses-d0757/anat/sub-OAS30001_ses-d0757_run-02_T1w.json',
     'sub-OAS30001/ses-d0757/anat/sub-OAS30001_ses-d0757_run-02_T1w.nii.gz',
     'sub-OAS30001/ses-d2430/anat/sub-OAS30001_ses-d2430_T1w.nii.gz',
     'sub-OAS30001/ses-d2430/anat/sub-OAS30001_ses-d2430_T1w.json']

# missing T1w
l += ['sub-OAS30002/ses-d2340/func/sub-OAS30002_ses-d2340_task-rest_run-02_bold.nii.gz',
      'sub-OAS30002/ses-d2340/func/sub-OAS30002_ses-d2340_task-rest_run-01_bold.nii.gz',
      'sub-OAS30002/ses-d2340/func/sub-OAS30002_ses-d2340_task-rest_run-01_bold.json',
      'sub-OAS30002/ses-d2340/func/sub-OAS30002_ses-d2340_task-rest_run-02_bold.json']

for p in l:
    f = out_dir / Path(p)
    print(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch(exist_ok=True)

f = out_dir / Path("dataset_description.json")

text = """{
"Name" : "test",
"BIDSVersion": "1.0.1"
}
"""
f.write_text(text)

###### full output
out_dir = Path("test_data/skeleton/output/preprocessing/sub-OAS30001")
out_dir.mkdir(exist_ok=True, parents=True)

l = ['freesurfer/sub-OAS30001/scripts/recon-all.done',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-01_desc-confounds_regressors.tsv',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-02_desc-confounds_regressors.tsv',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-03_desc-confounds_regressors.tsv',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-02_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-03_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-02_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-03_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
     ]

for p in l:
    f = out_dir / Path(p)
    print(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch(exist_ok=True)

###### fs_missing output
out_dir = Path("test_data/skeleton/output_fs_missing/preprocessing/sub-OAS30001")
out_dir.mkdir(exist_ok=True, parents=True)

l = ['fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-01_desc-confounds_regressors.tsv',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-02_desc-confounds_regressors.tsv',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-03_desc-confounds_regressors.tsv',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-02_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-03_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-02_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-03_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
     ]

for p in l:
    f = out_dir / Path(p)
    print(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch(exist_ok=True)

###### bold missing output
out_dir = Path("test_data/skeleton/output_bold_missing/preprocessing/sub-OAS30001")
out_dir.mkdir(exist_ok=True, parents=True)

l = ['freesurfer/sub-OAS30001/scripts/recon-all.done',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-01_desc-confounds_regressors.tsv',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-02_desc-confounds_regressors.tsv',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-03_desc-confounds_regressors.tsv',
     'fmriprep/sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz']

for p in l:
    f = out_dir / Path(p)
    print(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch(exist_ok=True)
