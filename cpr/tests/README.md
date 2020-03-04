
# Test data for unit tests

`test_data/skeleton` contains empty files used to test
query functions and file-exist checks.
The data is created via `simulate_data.py`.

* `test_data/skeleton/bids` contains input data
* `test_data/skeleton/output*` contains simulated output from
preprocessing

---

`test_data/preprocessed` has barebone preprocessed data. 
`copy_fs_files.py` creates this tree.


* `test_data/preprocessed/freesurfer` contains the stats folder of a
freesurfer subject.

---

`feature_columns.py` contains the expected columns of the output text files
that contain freesurfer features.
