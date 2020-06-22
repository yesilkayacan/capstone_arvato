import pandas as pd
import numpy as np
import pickle

from utils import decode_missing_values
from utils import remove_above_percent
from utils import ratio_missing


def transfrom_attribute_map(attr_mapping_df):
    '''Cleans the attr_mapping_df by filling the missing values. Also creates a dataframe of the 
    'Attribute'-'Meaning' couples where the 'Meaning' is 'unknown'
    
    ARGS
    ----
    attr_mapping_df: (pandas.DataFrame) Dataframe that has a Attribute and Meaning column

    RETURNS
    -------
    attr_mapping_clean: (pandas.DataFrame) Copy of attr_mapping_df where the nan values are filled with
        a forward fill and 'Meaning' matching 'unknown' rows have been removed
    unknown_mapping: (pandas.DataFrame) Dataframe of the 'Attribute'-'Meaning' couples where the 'Meaning'
        is 'unknown'
    '''
    
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


def transform_azdias(azdias_df, attr_mapping_df, top_level_attr_df, missing_cols_thresh=0.2, missing_rows_thresh=0.4):
    '''Clean the Udacity_AZDIAS dataset. Sets 'LNR' as index, removes 'EINGEFUEGT_AM' and 
    the features that have not been explained in attr_mapping_df or  top_level_attr_df.
    Cleans 'CAMEO_DEUG_2015' attribute where some data are falsely logged as 'X'.
    Decodes all the missing value encodings in the data as np.nan.
    Remove the features and rows where the missing values are above a given threshold.
    Finally remove all the rows that are left with a missing value.
    Returns the cleaned dataframe.

    ARGS
    ----
    azdias_df: (pandas.DataFrame) Udacity_AZDIAS dataframe to be cleaned
    attr_mapping_df: (pandas.DataFrame)  Dataframe that has a Attribute and Meaning column
    top_level_attr_df: (pandas.DataFrame) Dataframe that has a Attribute column
    missing_cols_thresh: (float) Threshold to remove features with missing value ratio above
    missing_rows_thresh: (float) Threshold to remove rows with missing value ratio above

    RETURNS
    -------
    azdias_df_filtered: (pandas.DataFrame) Cleaned copy of Udacity_AZDIAS dataframe
    features_kept: (list) List of features in the returned azdias_df_filtered
    '''
    
    azdias_df_filtered = azdias_df.copy()
    all_attributes = get_all_attributes(attr_mapping_df, top_level_attr_df)
    _, unknown_mapping = transfrom_attribute_map(attr_mapping_df)

    print('Setting LNR as index...')
    azdias_df_filtered.set_index('LNR', inplace=True)
    azdias_df_filtered.drop('EINGEFUEGT_AM', axis=1, inplace=True)
    
    print('Removing attriubtes that are not explained...')
    attributes_not_explained = azdias_df_filtered.columns[~azdias_df_filtered.columns.isin(all_attributes)] # columns not in attribute explanation files
    azdias_df_filtered.drop(attributes_not_explained, axis=1, inplace=True)
    
    print('Correcting issues found in CAMEO_DEUG_2015...')
    azdias_df_filtered['CAMEO_DEUG_2015'] = np.where(azdias_df_filtered['CAMEO_DEUG_2015']=='X', np.NaN, azdias_df_filtered['CAMEO_DEUG_2015'])
    azdias_df_filtered['CAMEO_DEUG_2015'] = azdias_df_filtered['CAMEO_DEUG_2015'].astype(float)
    
    print('Decoding and converting missing values to NaN...')
    azdias_df_filtered = decode_missing_values(azdias_df_filtered, unknown_mapping)
    ratio_missing_rows, ratio_missing_cols = ratio_missing(azdias_df_filtered, plot=False)
    
    print('Removing missing values according to row and collumn thresholds...')
    azdias_df_filtered = remove_above_percent(azdias_df_filtered, ratio_missing_cols, missing_cols_thresh, axis=1)
    azdias_df_filtered = remove_above_percent(azdias_df_filtered, ratio_missing_rows, missing_rows_thresh, axis=0)
    
    print('Removing rows which are left with missing values...')
    azdias_df_filtered.dropna(axis=0, inplace=True)

    print('Ratio of data used')
    print('features: %.2f' % (azdias_df_filtered.shape[1]/azdias_df.shape[1]*100))
    print('observations: %.2f' % (azdias_df_filtered.shape[0]/azdias_df.shape[0]*100))
    
    return azdias_df_filtered


def etl_transform(df, attributes_list, attr_mapping_df):
    '''Transform any data set taking into reference the attributes from the already transformed azdias dataset.
    The dataframe needs to have 'LNR' as one of the columns.
    Filter the data to have all the attributes listed in attributes_list.
    LNR will be set as index.
    Rows having incorrect data as 'X' values in featrue 'CAMEO_DEUG_2015' will be removed.
    Decodes all the missing value encodings in the data as np.nan.
    Finally remove all the rows that are left with a missing value.
    Returns the cleaned dataframe.

    ARGS
    ----
    df: (pandas.DataFrame) Udacity_AZDIAS dataframe to be cleaned
    attributes_list: (list) List of features in the reference data set (already transformed azdias dataset)

    RETURNS
    -------
    df_clean: (pandas.DataFrame) Cleaned copy of df dataframe
    '''
    
    print('Filtering features according to provided attribute list...')
    df_clean = df[['LNR', *attributes_list]].copy()
    _, unknown_mapping = transfrom_attribute_map(attr_mapping_df)

    df_clean.set_index('LNR', inplace=True)
    df_clean['CAMEO_DEUG_2015'] = np.where(df_clean['CAMEO_DEUG_2015']=='X', np.NaN, df_clean['CAMEO_DEUG_2015'])
    
    print('Decoding and converting missing values to NaN...')
    df_clean = decode_missing_values(df_clean, unknown_mapping)
    
    print('Dropping rows with NaN values...')
    df_clean.dropna(axis=0, inplace=True)
    
    return df_clean


def etl_save_data(obj_list, filenames_list):
    '''Generic save fucntion. All the files in the obj_list are saved as pickle as
    the the corresponding filenames_list items.

    Length of the obj_list must be the same as length of the filename_list.

    ARGS
    ----
    obj_list: (list) List of items to be saved
    filenames_list: (list) List of filenames that the corresponding items will be saved as
    '''
    
    assert len(obj_list)==len(filenames_list), 'Number of files to save and the names assigned do not match'
    
    for obj, file_name in zip(obj_list,filenames_list):
        with open(file_name + '.pkl','wb') as f:
            pickle.dump(obj, f, protocol=4)