import numpy as np
import os
import pandas as pd
from configparser import ConfigParser


def get_sample_counts(output_dir, dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv(os.path.join(output_dir, f"{dataset}.csv"))
    # same strategy for dealing with missing/uncertain labels
    df = df.fillna(0)
    df = df.replace(to_replace=-1, value=1)

    total_count = df.shape[0]
    labels = df[class_names].values
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts


def preprocess_labels(df, label_handling, class_names):
    if label_handling["empty"] == "zeros":
        new_df = df.fillna(0)
    elif label_handling["empty"] == "ones":
        new_df = df.fillna(1)
    for class_name in class_names:
        if label_handling[class_name] == "zeros":
            new_df[class_name] = df[class_name].replace(to_replace=-1, value=0)
        elif label_handling[class_name] == "ones":
            new_df[class_name] = df[class_name].replace(to_replace=-1, value=1)
    return new_df
