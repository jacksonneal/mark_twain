import json
import os

import pandas as pd
from numerapi import NumerAPI

from constants import ERA_COL, DATA_TYPE_COL, TARGET_COL, DATA_DIR

napi = NumerAPI()


def download_current_data():
    """
    Download data files for the current round.
    """
    current_round = napi.get_current_round()
    dir_path = f'{DATA_DIR}/numerai_dataset_{current_round}'
    if os.path.isdir(dir_path):
        print(f"Data files are up to date! Current round is: {current_round}")
    else:
        os.makedirs(dir_path)
        print(f"Downloading data files for round: {current_round}!")
        # tournament data and example predictions change every week
        # training and validation data only change periodically
        files = [
            "numerai_tournament_data_int8.parquet",
            "numerai_training_data_int8.parquet",
            "numerai_validation_data_int8.parquet",
            "example_predictions.parquet",
            "example_validation_predictions.parquet",
            "example_predictions.csv",
            "features.json"
        ]
        for file in files:
            napi.download_dataset(file, dest_path=f"{dir_path}/{file}")


def load_data(mode: str, feature_set, aux_target_cols, sample_4th_era: bool = False) -> pd.DataFrame:
    """
    Load data from file for current round.
    :param mode: data file to load
    :param feature_set: feature set to use
    :param aux_target_cols: auxiliary targets to load
    :param sample_4th_era: avoid overlapping eras by sampling every 4th during training
    :return: A tuple containing the datasets (train, val, test)
    """
    if aux_target_cols is None:
        aux_target_cols = []
    print(f"Loading the data {mode}")
    current_round = napi.get_current_round()
    dir_path = f"{DATA_DIR}/numerai_dataset_{current_round}/"
    if mode == "test":
        file_path = f"{dir_path}/numerai_tournament_data_int8.parquet"
    elif mode == "train":
        file_path = f"{dir_path}/numerai_training_data_int8.parquet"
    elif mode == "val":
        file_path = f"{dir_path}/numerai_validation_data_int8.parquet"
    else:
        raise Exception(f"unsupported mode: {mode}")
    features_path = f"{dir_path}/features.json"
    custom_features_path = "custom_features.json"

    if feature_set is not None:
        print(f"Reading feature set {feature_set}")
        path = features_path if feature_set == "small" else custom_features_path
        with open(path, "r") as f:
            feature_metadata = json.load(f)
        if feature_set == "small":
            feature_metadata = feature_metadata["feature_sets"]
        features = feature_metadata[feature_set]
        # read in small feature set with meta, target, and aux target cols
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL] + aux_target_cols
        df = pd.read_parquet(file_path, columns=read_columns)
    else:
        print('Reading full feature set')
        df = pd.read_parquet(file_path)

    if sample_4th_era and mode == "train":
        print('Sampling every 4th training era')
        # parse down the number of eras to every 4th era
        every_4th_era = df[ERA_COL].unique()[::4]
        df = df[df[ERA_COL].isin(every_4th_era)]

    return df
