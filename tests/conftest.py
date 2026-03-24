import csv
import subprocess

import numpy as np
import pytest

np.random.seed(0)


def profile_type2(b0, b1, b2, b3, data_size=None, random_scale=0):
    """Generate artificial Type 2 profile.

    Parameters
    ----------
    b0 : scalar
        Plateau region height.
    b1, b2, b3 : scalar
        Heavy edge region y = -b1 * (x - b2)^2 + b3.
    data_size : int, optional
        If passed, profile is padded with "bare substrate region" to this size.
    random_scale : scalar, default=0
        Scale for standard normal noise.

    Raises
    ------
    ValueError
        If profile lenth is larger than *data_size*.
    """
    x = np.arange(np.ceil(np.sqrt(b3 / b1) + b2).astype(int))
    x_bp = np.ceil(-np.sqrt((b3 - b0) / b1) + b2).astype(int)

    y1 = np.full(x.shape, b0)
    y1[x_bp:] = 0
    y2 = -b1 * ((x - b2) ** 2) + b3
    y2[:x_bp] = 0
    ret = (y1 + y2).astype(float)

    if data_size is not None:
        if len(ret) > data_size:
            raise ValueError("data_size is too small.")
        ret = np.pad(ret, (0, data_size - len(ret)))
    ret += np.random.standard_normal(ret.shape) * random_scale
    return ret


class RawDataFactory:
    def __init__(self, path):
        self.path = path

    def mkrawdir(self, dirname):
        path = self.path / dirname
        path.mkdir(parents=True)
        return path

    def mkrawfile(self, rawdir, filename, data):
        path = rawdir / filename
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for y in data:
                writer.writerow([y])
        return path


@pytest.fixture(scope="session")
def tmp_data_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("RawData-")
    rawdata_factory = RawDataFactory(path)

    N_PROFILES = 15
    DATA_SIZE = 150
    RANDOM_SCALE = 5

    rawdir = rawdata_factory.mkrawdir("Type2-00")
    for i in range(N_PROFILES):
        rawdata_factory.mkrawfile(
            rawdir,
            f"{str(i).zfill(2)}.csv",
            profile_type2(
                700, 1, 50, 800, data_size=DATA_SIZE, random_scale=RANDOM_SCALE
            ),
        )

    data_dir = tmp_path_factory.mktemp("Data-")
    profile_path = data_dir / "Type2.h5"
    subprocess.run(
        [
            "heavyedge",
            "prep",
            "--type",
            "csvs",
            "--res=1",
            "--sigma=1",
            "--std-thres=40",
            "--fill-value=0",
            rawdir,
            "-o",
            profile_path,
        ],
        capture_output=True,
        check=True,
    )

    softlabels_shape = (N_PROFILES, 3)
    softlabels = np.random.uniform(0, 1, softlabels_shape)
    softlabels /= softlabels.sum(axis=1, keepdims=True)
    label_npy_path = data_dir / "labels.npy"
    np.save(label_npy_path, softlabels)
    label_csv_path = data_dir / "labels.csv"
    with open(label_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type 1", "Type 2", "Type 3"])
        for row in softlabels:
            writer.writerow(row)

    wet_thicknesses = np.full((N_PROFILES,), 700)
    wet_thickness_path = data_dir / "wet_thicknesses.npy"
    np.save(wet_thickness_path, wet_thicknesses)

    return profile_path, (label_npy_path, label_csv_path), wet_thickness_path
