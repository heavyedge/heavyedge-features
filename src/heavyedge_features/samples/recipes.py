"""Recipies to build sample data."""

from heavyedge import get_sample_path as heavyedge_sample
from heavyedge_classify.samples import get_sample_path as classify_sample

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
}
