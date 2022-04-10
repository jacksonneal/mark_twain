import pandas as pd
from sklearn.decomposition import PCA
from os.path import exists


def pca_dataframe(df: pd.DataFrame, num_feature_comp: int, filename: str) -> pd.DataFrame:
    # check if file exists
    filepath = './' + filename
    if exists(filepath):
        pca_df = pd.read_parquet()
        return pca_df

    # Calculate PCA and return dataframe
    else:

        print('Calculating PCA...')
        principal = PCA(n_components=num_feature_comp)
        principal.fit(df)
        x = principal.transform(df)
        temp = pd.DataFrame(data=x, columns=['feature_{}'.format(i) for i in range(num_feature_comp)])
        era_target = df[['era', 'target']]
        era_target.reset_index(inplace=True)
        pca_df = era_target.join(temp)
        pca_df.set_index('id', inplace=True)

        # Create .parquet file
        pca_df.to_parquet(filename, index=True)
        print('File created!')

        return pca_df
