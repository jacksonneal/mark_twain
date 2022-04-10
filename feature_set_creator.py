import pandas as pd
import itertools
import numpy as np
from os.path import exists
import json


def create_feature_sets(df: pd.DataFrame):
    if exists('./feature_sets.json'):
        print('Feature set file exists.\n')
        pass
    else:
        features = list()
        for i in df.columns:
            if i.startswith('feature'):
                features.append(i)
        eras_not_to_keep = list()
        for i in df.era.unique():
            if (int(i) - 1) % 4 != 0:
                eras_not_to_keep.append(i)
        # Only keep every 4th era and drop others
        df = df[~df.era.isin(eras_not_to_keep)]
        cols = ['era'] + features + ['target']
        df_modified = df[cols]

        feature_target_corr = dict()
        for i in features:
            feature_target_corr[i] = df_modified[i].corr(df_modified['target'])
        target_feature_sorted_corr = dict(sorted(feature_target_corr.items(), key=lambda item: item[1], reverse=True))
        top_250_features_with_target = dict(itertools.islice(target_feature_sorted_corr.items(), 250))
        top_500_features_with_target = dict(itertools.islice(target_feature_sorted_corr.items(), 500))
        top_750_features_with_target = dict(itertools.islice(target_feature_sorted_corr.items(), 750))
        feature_dict = dict()
        feature_dict['top_250_features_with_target'] = list(top_250_features_with_target.keys())
        feature_dict['top_500_features_with_target'] = list(top_500_features_with_target.keys())
        feature_dict['top_750_features_with_target'] = list(top_750_features_with_target.keys())

        ''' 
            Create a dictionary called feature_all_eras_corr which has a feature name as the key and a dictionary as a value. 
            The value dictionary contains all eras (every 4th era) as keys and the correlation with target of that era for the 
            main key feature as the value field. 
            feature_all_corr = {"feature_name": {"era": "corr. with target"}}
            '''
        feature_all_eras_corr = dict()
        eras = list(df_modified.era.unique())
        for feature in features:
            era_dict = dict()
            for era in eras:
                era_df = df_modified[df_modified.era == era]
                corr_val = era_df[feature].corr(era_df['target'])
                era_dict[era] = corr_val
            feature_all_eras_corr[feature] = era_dict

        '''
            Next we calculate the standard deviation of corr. of eras for all features. This will help identify for which
            features the corr. with target is the most volatile across all the eras. The output will be a dictionary with 
            feature name as key and standard deviation as the value.
            '''
        feature_std_dev = dict()
        for feature, eras_dict in feature_all_eras_corr.items():
            feature_std_dev[feature] = np.std(list(eras_dict.values()))
        asc_feature_list = list(sorted(feature_std_dev.items(), key=lambda item: item[1]).keys())
        dsc_feature_list = list(sorted(feature_std_dev.items(), key=lambda item: item[1], reverse=True).keys())
        feature_dict['top_100_most_volatile_features'] = dsc_feature_list[:100]
        feature_dict['top_250_most_volatile_features'] = dsc_feature_list[:250]
        feature_dict['top_500_most_volatile_features'] = dsc_feature_list[:500]
        feature_dict['top_750_most_volatile_features'] = dsc_feature_list[:750]
        feature_dict['top_100_least_volatile_features'] = asc_feature_list[:100]
        feature_dict['top_250_least_volatile_features'] = asc_feature_list[:250]
        feature_dict['top_500_least_volatile_features'] = asc_feature_list[:500]
        feature_dict['top_750_least_volatile_features'] = asc_feature_list[:750]

        '''
            Create a json file 
            '''
        with open("feature_sets.json", "w") as outfile:
            json.dump(feature_dict, outfile)
        print('JSON file with feature sets created.\n')
