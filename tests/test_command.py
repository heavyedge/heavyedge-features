import os
import subprocess


def test_globalfeatures_label_csv(tmp_data_path, tmp_path):
    _, (_, label_csv_path), _ = tmp_data_path
    out_path = tmp_path / "global-features.csv"
    subprocess.run(
        [
            "heavyedge",
            "--log-level=INFO",
            "features-global",
            label_csv_path,
            "--target-indices",
            "0",
            "-o",
            out_path,
        ],
        check=True,
    )
    assert os.path.exists(out_path)


def test_localfeatures_label_csv(tmp_data_path, tmp_path):
    profiles_path, (_, label_csv_path), wt_path = tmp_data_path
    out_path = tmp_path / "local-features.csv"
    subprocess.run(
        [
            "heavyedge",
            "--log-level=INFO",
            "features-local",
            profiles_path,
            label_csv_path,
            wt_path,
            "--sigma",
            "1.0",
            "--type1-indices",
            "0",
            "--type2-indices",
            "1",
            "--type3-indices",
            "2",
            "-o",
            out_path,
        ],
        check=True,
    )
