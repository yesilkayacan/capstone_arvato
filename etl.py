import pandas as pd


def transfrom_attribute_map(attr_mapping_df, top_level_attr_df):
    
    attr_mapping_clean = attr_mapping_df.copy()
    attr_mapping_clean.fillna(method='ffill', inplace=True)

    unknown_mapping = attr_mapping_clean[attr_mapping_clean['Meaning']=='unknown'].set_index('Attribute')['Value']
    attr_mapping_clean.drop(attr_mapping_clean[attr_mapping_clean['Meaning']=='unknown'].index, axis=0, inplace=True)

    

    return attr_mapping_clean, unknown_mapping


def get_all_attributes(attr_mapping_df, top_level_attr_df):
    '''Combines all the unique values in the 'Attribute' column of the input dataframes
    
    ARGS
    ----
    attr_mapping_df: (pandas.DataFrame) Dataframe that has a Attribute column
    top_level_attr_df: (pandas.DataFrame) Dataframe that has a Attribute column

    RETURNS
    -------
    all_attributes: (set) Set of unique values combined from the 'Attribute' columns of the input dataframes
    '''
    
    all_attributes = set(attr_mapping_df[attr_mapping_df['Attribute'].notnull()]['Attribute'].unique()).union(set(top_level_attr_df['Attribute'].unique()))

    return all_attributes



def transform_azdias(df, attr_mapping_df, top_level_attr_df, missing_cols_thresh=0.2, missing_rows_thresh=0.4):
    '''
    
    '''
    
    df_filtered = df.copy()
    attr_mapping_clean, unknown_mapping = transfrom_attribute_map(attr_mapping_df, top_level_attr_df)


    print('Setting LNR as index:')
    df_filtered.set_index('LNR', inplace=True)
    df_filtered.drop('EINGEFUEGT_AM', axis=1, inplace=True)
    
    print('Removing attriubtes that are not explained:')
    all_attributes = set(attr_mapping_df[attr_mapping_df['Attribute'].notnull()]['Attribute'].unique()).union(set(top_level_attr_df['Attribute'].unique()))
    attributes_not_explained = df_filtered.columns[~df_filtered.columns.isin(all_attributes)] # columns not in attribute explanation files
    df_filtered.drop(attributes_not_explained, axis=1, inplace=True)
    
    print('Correcting issues found in CAMEO_DEUG_2015:')
    df_filtered['CAMEO_DEUG_2015'] = np.where(df_filtered['CAMEO_DEUG_2015']=='X', np.NaN, df_filtered['CAMEO_DEUG_2015'])
    df_filtered['CAMEO_DEUG_2015'] = df_filtered['CAMEO_DEUG_2015'].astype(float)
    
    print('Decoding and converting missing values to NaN:')
    df_filtered = decode_missing_values(df_filtered, unknown_mapping=unknown_mapping)
    ratio_missing_rows, ratio_missing_cols = ratio_missing(df_filtered, plot=False)
    
    print('Removing missing values according to row and collumn thresholds:')
    df_filtered = remove_above_percent(df_filtered, ratio_missing_cols, missing_cols_thresh, axis=1)
    df_filtered = remove_above_percent(df_filtered, ratio_missing_rows, missing_rows_thresh, axis=0)
    
    print('Removing rows which are left with missing values:')
    df_filtered.dropna(axis=0, inplace=True)
    features_kept = df_filtered.columns
    
    return df_filtered, features_kept, attr_mapping_clean