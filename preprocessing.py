import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from etl import transfrom_attribute_map

from utils import categorize

def preprocess_data(df, attr_mapping_df, train=True, scaler=None, pca=None):
    '''Preprocesses data to to get ready for machine learning algortithms. The preprocessing steps are
    using one hot encoding to encode categorical features. Scaling the categorized data using standardization
    and applying PCA fore feature number reduction.

    There is a additional training swith implemented. This enables the standard scaler and PCA to be trained.
    Otherwise the the scaler and pca objects need to be provided and these provided objects will be applied to the data.

    ARGS
    ----
    df: (pandas.DataFrame) Cleaned dataframe to be preprocessed
    attr_mapping_df: (pandas.DataFrame) Dataframe that has a Attribute and Meaning column
    train: (boolean) Training switch when true, scaler and pca will be trained. Otherwise these input objects will
        be applied to the data
    scaler: (sklearn.preprocessing.StandardScaler) Scaler object to apply to the data. Required when train=False
    pca: (sklearn.decomposition.PCA) PCA object to apply to the data. Required when train=False

    RETURNS
    -------
    df_reduced: (numpy.Array) Preprocessed output of df. Onehot encoded, scaled and PCA applied
    df_preprocessed: (pandas.DataFrame) Copy of input df as categorized
    scaler: (sklearn.preprocessing.StandardScaler) Scaler object, same as input when train=False. Otherwise, the trained object
    pca: (sklearn.decomposition.PCA) PCA object, same as input when train=False. Otherwise, the trained object
    
    '''
    
    if not train:
        assert scaler is not None, 'scaler object must be provided when not in training mode'
        assert pca is not None, 'PCA object must be provided when not in training mode'
    
    df_preprocessed = df.copy()
    attr_mapping_clean, _ = transfrom_attribute_map(attr_mapping_df)

    print('Filtering qualitative features:')
    numeric_features = ['ANZ_HAUSHALTE_AKTIV', 'ANZ_HH_TITEL', 'ANZ_PERSONEN', 'ANZ_TITEL', 'GEBURTSJAHR', 'KBA13_ANZAHL_PKW', 'MIN_GEBAEUDEJAHR']
    qualitative_features = list(set(attr_mapping_clean['Attribute'].unique()).difference(set(numeric_features)))
    
    qualitative_features_used = [x for x in df.columns if x in(qualitative_features)]
    qualitative_features_mapping = attr_mapping_clean[attr_mapping_clean['Attribute'].isin(qualitative_features_used)]
    
    print('Removing qualitative features with granularity higher then 10')
    granularity = qualitative_features_mapping['Attribute'].value_counts().sort_values(ascending=False)
    df_preprocessed.drop(granularity[granularity>10].index, axis=1, inplace=True)
    qualitative_features_mapping = qualitative_features_mapping.drop(qualitative_features_mapping[qualitative_features_mapping['Attribute'].isin(granularity[granularity>10].index)].index)
    
    print('Categorizing qualitative features:')
    df_preprocessed = categorize(df_preprocessed, qualitative_features_mapping)
    
    if train:
        # in training mode
        print('Training StandardScaler...')
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_preprocessed)
        print('Starting PCA...')
        df_reduced, pca = dimentionality_reduction_PCA(df_scaled, train=True, n_comps=750, pca=None, use_subset=True, subset=60000)
    else:
        print('Standardizing data:')
        df_scaled = scaler.transform(df_preprocessed)
        print('Starting PCA...')
        df_reduced, pca = dimentionality_reduction_PCA(df_scaled, train=False, pca=pca, use_subset=False)
    
    print('Preprocessing complete')

    return df_reduced, df_preprocessed, scaler, pca


def get_subset(data, subset_size, sample_ind=None):
    '''Gets the random subset of data with a given size. Can also get the subset
    provided the indexes in the sample_ind.
    
    ARGS
    ----
    data: (np.Array) Data to get the random subset from
    subset_size: (int) Size of the output subset
    sample_ind: (list) Can return the items stored in the indices of the sample_ind list

    RETURNS
    -------
    subset: (np.Array) Random subset taken from the data with the provided options
    sample_ind: (list) Indices of the subset items in the Data
    '''

    assert type(subset_size)==int, 'subset_size must be of type int'
    assert subset_size<=data.shape[0], 'subset_size must be less then or equal to the row count of data'

    if sample_ind==None:
        sample_ind = np.random.choice(data.shape[0], size=subset_size, replace=False)

    subset = data.iloc[sample_ind, :]

    return subset, sample_ind



def dimentionality_reduction_PCA(x, train=True, n_comps=750, pca=None, use_subset=True, subset=5000):
    '''Applies PCA to the data. When training mode is false uses the input pca object. When in training mode,
    trains the PCA and then applies PCA.
    
    ARGS
    ----
    x: (np.Array) Data for pca application
    train: (boolean) Training switch when true, pca will be trained. Otherwise the input object will
        be applied to the data
    n_comps: (int) Number of components in PCA output
    pca: (sklearn.decomposition.PCA) PCA object to apply to the data. Required when train=False
    use_subset: (boolean) Subset usage switch. Uses subset when True
    subset: (int) Size of the subset

    RETURNS
    -------
    x_reduced: (np.Array) PCA output of the data
    pca: (sklearn.decomposition.PCA) PCA object, same as input when train=False. Otherwise, the trained object
    '''

    if train:
        pca = PCA(n_comps)

        if use_subset:
            print('Getting the subset with {} samples'.format(subset))
            x_train, sample_ind = get_subset(x, subset, sample_ind=None)
        else:
            x_train = x
        
        print('Training PCA...')
        pca.fit_transform(x_train)
        print('Fitting PCA...')
        x_reduced = pca.fit_transform(x)
    else:
        print('Fitting PCA...')
        x_reduced = pca.transform(x)

    return x_reduced, pca
