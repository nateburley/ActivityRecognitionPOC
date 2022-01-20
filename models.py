"""
Module containing functions pertaining to constructing models to classify activities.

Author: Nate Burley
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


# Probably unnecessary function, but... whatever.
def build_extra_trees_classifier() -> ExtraTreesClassifier:
    clf = ExtraTreesClassifier()  # Default parameters work best. Go figure.
    return clf