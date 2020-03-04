from pathlib import Path
import shutil

preprocessed_dir = Path("/Users/franzliem/Desktop/dpr_tests/out")
out_dir = Path("test_data/preprocessed/freesurfer")

stats_in_dir = preprocessed_dir / "preprocessing/fmriprep/freesurfer/sub-01/stats"
stats_out_dir = out_dir / "preprocessing/fmriprep/freesurfer/sub-01/stats"

stats_out_dir.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(stats_in_dir, stats_out_dir)