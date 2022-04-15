import itertools
import json
import os
from typing import Optional

import pandas as pd
from numerapi import NumerAPI
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from numerai.definitions import ERA_COL, DATA_TYPE_COL, TARGET_COL, DATA_DIR, CUSTOM_FEATURES_FILE, \
    PROVIDED_FEATURES_FILE, PROVIDED_FEATURE_SETS, PCA_DIR

napi = NumerAPI()


def download_current_data():
    """
    Download data files for the current round.
    """
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    current_round = napi.get_current_round()
    round_dir = os.path.join(DATA_DIR, f"numerai_dataset_{current_round}")
    if os.path.isdir(round_dir):
        print(f"Round data files are up to date! Current round is {current_round}")
    else:
        os.makedirs(round_dir)
        print(f"Downloading data files for round {current_round}!")
        # tournament data and example predictions change every week
        # training and validation data only change periodically
        round_files = [
            "numerai_tournament_data_int8.parquet",
            "example_predictions.csv",
            "example_predictions.parquet",
        ]
        for file in round_files:
            napi.download_dataset(file, dest_path=os.path.join(round_dir, file))
    files = [
        "numerai_training_data_int8.parquet",
        "numerai_validation_data_int8.parquet",
        "example_validation_predictions.parquet",
        "features.json",
    ]
    for file in files:
        path = os.path.join(DATA_DIR, file)
        if os.path.isfile(path):
            print(f"File {file} exists! not downloading")
        else:
            napi.download_dataset(file, dest_path=os.path.join(DATA_DIR, file))


def load_data(mode: str, feature_set="small", aux_target_cols=None, sample_4th_era=False, pca: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from file for current round.
    :param mode: data file to load
    :param feature_set: feature set to use
    :param aux_target_cols: auxiliary targets to load
    :param sample_4th_era: avoid overlapping eras by sampling every 4th during training
    :param pca: number of components
    :return: A tuple containing the datasets (train, val, test)
    """
    if aux_target_cols is None:
        aux_target_cols = []
    print(f"Loading {mode} data")
    current_round = napi.get_current_round()
    round_dir = os.path.join(DATA_DIR, f"numerai_dataset_{current_round}")
    if mode == "test":
        file_path = os.path.join(round_dir, "numerai_tournament_data_int8.parquet")
    elif mode == "train":
        file_path = os.path.join(DATA_DIR, "numerai_training_data_int8.parquet")
    elif mode == "val":
        file_path = os.path.join(DATA_DIR, "numerai_validation_data_int8.parquet")
    else:
        raise Exception(f"unsupported mode: {mode}")

    using_provided = feature_set in PROVIDED_FEATURE_SETS

    if feature_set != "full":
        print(f"Reading feature set {feature_set}")
        file = PROVIDED_FEATURES_FILE if using_provided else CUSTOM_FEATURES_FILE
        with open(file, "r") as f:
            feature_metadata = json.load(f)
        if using_provided:
            feature_metadata = feature_metadata["feature_sets"]
        features = feature_metadata[feature_set]
        # read in feature set with meta, target, and aux target cols
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL] + aux_target_cols
        df = pd.read_parquet(file_path, columns=read_columns)
    else:
        print('Reading full feature set')
        df = pd.read_parquet(file_path)

    if sample_4th_era and mode == "train":
        print('Sampling every 4th training era')
        # sample down to every 4th era
        every_4th_era = df[ERA_COL].unique()[::4]
        df = df[df[ERA_COL].isin(every_4th_era)]

    if pca is not None:
        df = to_pca(df, pca, os.path.join(PCA_DIR, f"{mode}-{feature_set}-{sample_4th_era}-{pca}.parquet"))

    return df


def create_feature_sets(corr=True, volatile=True):
    if os.path.isfile(CUSTOM_FEATURES_FILE):
        print("Custom feature sets already created!")
    else:
        print("Creating custom feature sets, this may take a while")
        df = load_data("train", feature_set="full", sample_4th_era=False, aux_target_cols=[])
        # extract feature col names
        feature_cols = [c for c in df if c.startswith("feature_")]
        feature_dict = dict()

        if corr:
            print("Creating correlation feature sets")
            feature_target_corr = df[feature_cols].corrwith(df[TARGET_COL])
            sorted_feature_target_corr = dict(
                sorted(feature_target_corr.items(), key=lambda item: item[1], reverse=True))
            top_250_features_with_target = dict(itertools.islice(sorted_feature_target_corr.items(), 250))
            top_500_features_with_target = dict(itertools.islice(sorted_feature_target_corr.items(), 500))
            top_750_features_with_target = dict(itertools.islice(sorted_feature_target_corr.items(), 750))
            feature_dict = dict()
            feature_dict['top_250_features_with_target'] = list(top_250_features_with_target.keys())
            feature_dict['top_500_features_with_target'] = list(top_500_features_with_target.keys())
            feature_dict['top_750_features_with_target'] = list(top_750_features_with_target.keys())

        if volatile:
            print("Creating volatility feature sets")
            feature_target_corr_by_era = df.groupby(ERA_COL)[feature_cols].corrwith(df[TARGET_COL])
            feature_target_corr_std_by_era = feature_target_corr_by_era[feature_cols].std()
            sorted_feature_target_corr_std_by_era = dict(
                sorted(feature_target_corr_std_by_era.items(), key=lambda item: item[1], reverse=True))
            feature_dict['top_100_most_volatile_features'] = list(dict(
                itertools.islice(sorted_feature_target_corr_std_by_era.items(), 100)).keys())
            feature_dict['top_250_most_volatile_features'] = list(dict(
                itertools.islice(sorted_feature_target_corr_std_by_era.items(), 250)).keys())
            feature_dict['top_500_most_volatile_features'] = list(dict(
                itertools.islice(sorted_feature_target_corr_std_by_era.items(), 500)).keys())
            feature_dict['top_750_most_volatile_features'] = list(dict(
                itertools.islice(sorted_feature_target_corr_std_by_era.items(), 750)).keys())
            sorted_feature_target_corr_std_by_era = dict(
                sorted(feature_target_corr_std_by_era.items(), key=lambda item: item[1], reverse=False))
            feature_dict['top_100_least_volatile_features'] = list(dict(
                itertools.islice(sorted_feature_target_corr_std_by_era.items(), 100)).keys())
            feature_dict['top_250_least_volatile_features'] = list(dict(
                itertools.islice(sorted_feature_target_corr_std_by_era.items(), 250)).keys())
            feature_dict['top_500_least_volatile_features'] = list(dict(
                itertools.islice(sorted_feature_target_corr_std_by_era.items(), 500)).keys())
            feature_dict['top_750_least_volatile_features'] = list(dict(
                itertools.islice(sorted_feature_target_corr_std_by_era.items(), 750)).keys())

        with open(CUSTOM_FEATURES_FILE, "w") as f:
            json.dump(feature_dict, f)
            print('Custom feature sets created!')


def to_pca(df: pd.DataFrame, num_feature_comp: int, fp: str) -> pd.DataFrame:
    if os.path.isfile(fp):
        print("PCA already computed, loading file!")
        temp = pd.read_parquet(fp)
    else:
        print(f"Computing PCA for {num_feature_comp} components")
        features = [c for c in df if c.startswith("feature_")]
        df_features = df[features]
        scalar = StandardScaler()
        scalar.fit(df_features)
        scaled_df = scalar.transform(df_features)
        principal = PCA(n_components=num_feature_comp)
        principal.fit(scaled_df)
        df_principal = principal.transform(scaled_df)
        temp = pd.DataFrame(data=df_principal, columns=[f"feature_{i}" for i in range(num_feature_comp)])

    targets = [c for c in df if c.startswith("target_")]
    meta_cols = [ERA_COL] + targets
    df_meta = df[meta_cols]
    df_meta.reset_index(inplace=True)
    df_pca = df_meta.join(temp)
    df_pca.set_index('id', inplace=True)

    # Create .parquet file for temp to load next time
    if not os.path.isdir(PCA_DIR):
        os.makedirs(PCA_DIR)
    temp.to_parquet(fp, index=True)

    return df_pca


def get_num_features(feature_set: str) -> int:
    if feature_set == "full":
        num_features = 1050
    else:
        using_provided = feature_set in PROVIDED_FEATURE_SETS
        file = PROVIDED_FEATURES_FILE if using_provided else CUSTOM_FEATURES_FILE
        with open(file, "r") as f:
            feature_metadata = json.load(f)
        if using_provided:
            feature_metadata = feature_metadata["feature_sets"]
        num_features = len(feature_metadata[feature_set])
    print(f"Feature set {feature_set} has {num_features} features")
    return num_features
