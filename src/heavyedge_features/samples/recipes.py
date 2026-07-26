"""Recipies to build sample data."""

import csv

import numpy as np
from heavyedge import ProfileData
from heavyedge import get_sample_path as heavyedge_sample
from heavyedge_classify.samples import get_sample_path as classify_sample

from . import get_sample_path


def save_hw(path):
    N, _ = ProfileData(get_sample_path("Profiles.h5")).shape()
    h_w = np.full(N, 0.25)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wet_thickness"])
        for wt in h_w:
            writer.writerow([wt])


def save_classprob(path):
    # save npy as csv
    prob = np.load(classify_sample("labels-pred.npy"))
    _, C = prob.shape
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"prob_class_{c}" for c in range(C)])
        for row in prob:
            writer.writerow(row)


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
    "wet_thickness.csv": lambda path: save_hw(path),
    "class_probabilities.csv": lambda path: save_classprob(path),
    "shape-features.csv": lambda path: [
        "heavyedge",
        "shape-features",
        get_sample_path("Profiles.h5"),
        get_sample_path("wet_thickness.csv"),
        get_sample_path("class_probabilities.csv"),
        "--sigma",
        "32",
        "--type1-indices",
        "0",
        "--type2-indices",
        "1",
        "--type3-indices",
        "2",
        "--target-indices",
        "0",
        "-o",
        path,
    ],
}
