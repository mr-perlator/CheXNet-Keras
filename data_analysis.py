import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from configparser import ConfigParser
import os
from utility import preprocess_labels


def main():
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    dataset_csv_dir = cp["TRAIN"].get("dataset_csv_dir")
    label_handling = {"empty": cp["LABEL"].get("empty")}
    class_names = cp["DEFAULT"].get("class_names").split(",")
    for class_name in class_names:
        label_handling[class_name] = cp["LABEL"].get(class_name)

    trainfile = os.path.join(dataset_csv_dir, "train.csv")
    # validationfile = os.path.join(dataset_csv_dir, "valid.csv")
    train_df = pd.read_csv(trainfile)
    # validation_df = pd.read_csv(validationfile)
    processed_train_df = preprocess_labels(train_df, label_handling, class_names)

    filenum = train_df.shape[0]
    for class_name in class_names:
        print(f"**Stats for {class_name}**")
        print(f"labeled: {str(train_df[class_name].count())}")
        print(f"NaN: {str(filenum - train_df[class_name].count())}")
        print(f"labeled counts: \n{str(train_df[class_name].value_counts())}")
        """print(f"-1: {str(train_df[class_name].count(-1))}")
        print(f"0: {str(train_df[class_name].count(0))}")
        print(f"1: {str(train_df[class_name].count(1))}")"""
        print(f"uncertain label policy: {label_handling[class_name]}")
        print(f"counts after preprocessing: \n{str(processed_train_df[class_name].value_counts())}")


if __name__ == "__main__":
    main()