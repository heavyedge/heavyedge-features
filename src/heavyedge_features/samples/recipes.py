"""Recipies to build sample data."""

import numpy as np
from heavyedge import ProfileData
from heavyedge import get_sample_path as heavyedge_sample
from heavyedge_classify.samples import get_sample_path as classify_sample

from . import get_sample_path


def save_wetthicknesses(path):
    N, _ = ProfileData(get_sample_path("Profiles.h5")).shape()
    wet_thicknesses = np.full(N, 0.25)
    np.save(path, wet_thicknesses)


RECIPES = {
    "Profiles.h5": lambda path: [
        "heavyedge",
        "merge",
        heavyedge_sample("Prep-Type1.h5"),
        heavyedge_sample("Prep-Type2.h5"),
        heavyedge_sample("Prep-Type3.h5"),
        "-o",
        path,
    ],
    "global-features.csv": lambda path: [
        "heavyedge",
        "features-global",
        classify_sample("labels-pred.npy"),
        "--target-indices",
        "0",
        "-o",
        path,
    ],
    "wet_thickness.npy": lambda path: save_wetthicknesses(path),
    "local-features.csv": lambda path: [
        "heavyedge",
        "features-local",
        get_sample_path("Profiles.h5"),
        classify_sample("labels-pred.npy"),
        get_sample_path("wet_thickness.npy"),
        "--sigma",
        "32",
        "--type1-indices",
        "0",
        "--type2-indices",
        "1",
        "--type3-indices",
        "2",
        "-o",
        path,
    ],
}
