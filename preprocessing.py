import pandas as pd
from etl import transfrom_attribute_map

def preprocess_data(df, attr_mapping, train=True, scaler=None, pca=None):
    '''
    
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
        print('Training StandardScaler and standardizing data:')
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_preprocessed)
    else:
        print('Standardizing data:')
        df_scaled = scaler.transform(df_preprocessed)
    
    return df_reduced, df_preprocessed, scaler, pca


def get_subset(data, subset_size, sample_ind=None):
    '''Gets the random subset of data with a given size. Can also get the subset
    provided the indexes in the sample_ind.
    
    ARGS
    ----
    data: (pandas.DataFrame or np.Array) Data to get the random subset from
    subset_size: (int) Size of the output subset
    sample_ind: (list) Can return the items stored in the indices of the sample_ind list

    RETURNS
    -------
    subset: (pandas.DataFrame or np.Array) Random subset taken from the data with the provided options
    sample_ind: (list) Indices of the subset items in the Data
    '''

    assert type(subset_size)==int, 'subset_size must be of type int'
    assert subset_size<=data.shape[0], 'subset_size must be less then or equal to the row count of data)

    if sample_ind=None:
        sample_ind = np.random.choice(data.shape[0], size=subset_size, replace=False)

    subset = azdias_scaled[data]

    return subset, sample_ind



def dimentionality_reduction_PCA(x, train=True, subset=60000):
    pass