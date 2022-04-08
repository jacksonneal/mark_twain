import pandas as pd
from sklearn.decomposition import PCA
from os.path import exists


def pca_dataframe(df, num_feature_comp):
    # check if file exists
    filepath = './' + 'pca_{}_components.parquet'.format(num_feature_comp)
    if exists(filepath):
        pca_df = pd.read_parquet('pca_{}_components.parquet'.format(num_feature_comp))
        return pca_df

    # Calculate PCA and return dataframe
    else:

        print('Calculating PCA for {} components'.format(num_feature_comp))
        principal = PCA(n_components=num_feature_comp)
        principal.fit(df)
        x = principal.transform(df)
        temp = pd.DataFrame(data=x, columns=['feature_{}'.format(i) for i in range(num_feature_comp)])
        era_target = df[['era', 'target']]
        era_target.reset_index(inplace=True)
        pca_df = era_target.join(temp)
        pca_df.set_index('id', inplace=True)

        # Create .parquet file
        file_name = 'pca_{}_components.parquet'.format(num_feature_comp)
        pca_df.to_parquet(file_name, index=True)
        print('{} component file created!'.format(num_feature_comp))

        return pca_df
