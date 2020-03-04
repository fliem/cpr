"""
This script produces test data. It takes ds114_test2 and
- renames task-fingerfootlips to task-rest and selects 5 volumes
- renames test and retest ses to s1, s2

"""

import os
from pathlib import Path
import shutil
from nilearn import image
import json

in_dir = Path("/Users/franzliem/Desktop/ds114/ds114_test2")
out_dir = Path("/Users/franzliem/Desktop/dpr_tests/ds114_test2_mini")
out_dir.mkdir(exist_ok=True)

func_sidecar = {"EchoTime": 0.05,
                "FlipAngle": 90,
                "RepetitionTime": 2.5,
                "SliceTiming": [
                    0.0,
                    1.2499999999999998,
                    0.08333333333333333,
                    1.333333333333333,
                    0.16666666666666666,
                    1.4166666666666663,
                    0.25,
                    1.4999999999999996,
                    0.3333333333333333,
                    1.5833333333333328,
                    0.41666666666666663,
                    1.666666666666666,
                    0.5,
                    1.7499999999999993,
                    0.5833333333333333,
                    1.8333333333333326,
                    0.6666666666666666,
                    1.9166666666666659,
                    0.75,
                    1.9999999999999991,
                    0.8333333333333333,
                    2.083333333333332,
                    0.9166666666666666,
                    2.1666666666666656,
                    1.0,
                    2.249999999999999,
                    1.0833333333333333,
                    2.333333333333332,
                    1.1666666666666665,
                    2.416666666666665
                ],
                "TaskName": "finger_foot_lips"
                }
# collect files
g = list(in_dir.rglob("**/*"))
use_files = []
print(g)
for f in g:
    fs = str(f)
    if ("dataset_description" in fs) or ("anat" in fs) or (("task-fingerfootlips" in fs) and not ("events.tsv" in fs)):
        use_files.append(f)

for f in use_files:
    out_filename = str(f.relative_to(in_dir))
    out_filename = out_filename.replace("task-fingerfootlips_", "task-rest_")
    out_filename = out_filename.replace("ses-test", "ses-s1")
    out_filename = out_filename.replace("ses-retest", "ses-s2")

    full_out_path = out_dir / out_filename
    if f.is_file():
        full_out_path.parent.mkdir(exist_ok=True, parents=True)
        if "task-rest_bold.nii.gz" in str(full_out_path):
            # select images 10...15
            selected_img = image.index_img(str(f), range(10, 15))
            selected_img.to_filename(str(full_out_path))
            print("SELECT", f, full_out_path)

            json_file = str(full_out_path).strip(".nii.gz") + ".json"
            with open(json_file, "w") as fi:
                json.dump(func_sidecar, fi)
        else:
            print("copy", f, full_out_path)
            shutil.copy(str(f), str(full_out_path))
