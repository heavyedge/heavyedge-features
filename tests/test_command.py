import os
import subprocess


def test_shapefeatures(tmp_data_path, tmp_path):
    profiles_path, wt_path, prob_path = tmp_data_path
    out_path = tmp_path / "shape-features.csv"
    subprocess.run(
        [
            "heavyedge",
            "--log-level=INFO",
            "shape-features",
            profiles_path,
            wt_path,
            prob_path,
            "--sigma",
            "1.0",
            "--type1-indices",
            "0",
            "--type2-indices",
            "1",
            "--type3-indices",
            "2",
            "--target-indices",
            "0",
            "-o",
            out_path,
        ],
        check=True,
    )
    assert os.path.exists(out_path)


def test_shapefeatures_parallel(tmp_data_path, tmp_path):
    profiles_path, wt_path, prob_path = tmp_data_path
    out_path = tmp_path / "shape-features.csv"
    subprocess.run(
        [
            "heavyedge",
            "--log-level=INFO",
            "shape-features",
            profiles_path,
            wt_path,
            prob_path,
            "--sigma",
            "1.0",
            "--type1-indices",
            "0",
            "--type2-indices",
            "1",
            "--type3-indices",
            "2",
            "--target-indices",
            "0",
            "--n-jobs",
            "2",
            "--n-chunks",
            "4",
            "-o",
            out_path,
        ],
        check=True,
    )
    assert os.path.exists(out_path)
