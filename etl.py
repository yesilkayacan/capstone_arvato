import pandas as pd
import numpy as np
import pickle
import progressbar

from utils import decode_missing_values
from utils import remove_above_percent
from utils import ratio_missing

def get_unkown_mapping(attr_mapping_df):
    '''Creates a dataframe of the 'Attribute'-'Meaning' couples where the 'Meaning' is 'unknown'
    
    ARGS
    ----
    attr_mapping_df: (pandas.DataFrame) Dataframe that has a Attribute and Meaning column

    RETURNS
    -------
    unknown_mapping: (pandas.DataFrame) Dataframe of the 'Attribute'-'Meaning' couples where the 'Meaning'
        is 'unknown'
    '''
    
    attr_mapping_clean = attr_mapping_df.copy()
    attr_mapping_clean.fillna(method='ffill', inplace=True)

    unknown_mapping = attr_mapping_clean[attr_mapping_clean['Meaning']=='unknown'].set_index('Attribute')['Value']

    return unknown_mapping


def get_feature_types(df):
    '''Returns list of numeric features and the categorical features in the data. Numerical features
    have been extracted from DIAS Attributes - Values 2017.xlsx file manually. All other features are
    are assumed to be categorical (since all the left features in DIAS Attributes - Values 2017.xlsx 
    are categorical).

    ARGS
    ----
    df: (pandas.DataFrame) Dataframe whose features are clasified as categorical or numerical

    RETURNS
    -------
    qualitative_features_used: (list) Qualitative features in the dataframe df
    numeric_features_used: (list) Quantitative features in the dataframe df
    '''
    
    numeric_features = ['ANZ_HAUSHALTE_AKTIV', 'ANZ_HH_TITEL', 'ANZ_PERSONEN', 'ANZ_TITEL', 'GEBURTSJAHR', 'KBA13_ANZAHL_PKW', 'MIN_GEBAEUDEJAHR']
    
    # The numeric_features have been manually extracted from the DIAS Attributes - Values 2017.xlsx file
    numeric_features_used = [x for x in df.columns if x in(numeric_features)]
    
    # all other features are assumed to be categorical
    # qualitative_features = list(set(attr_mapping_clean['Attribute'].unique()).difference(set(numeric_features)))
    # qualitative_features_used = [x for x in df_impute.columns if x in(qualitative_features)]
    qualitative_features_used = np.setdiff1d(df.columns, numeric_features_used)
    
    return qualitative_features_used, numeric_features_used


def impute_na(df):
    '''Impute data inplace of missing values. Uses median for quantitative 
    data and most frequent for qualitative data.'
    
    ARGS
    ----
    df: (pandas.DataFrame) Dataframe where the missing values will be replaced

    RETURNS
    -------
    df_impute: (pandas.DataFrame) Copy of df where the missing values have been imputed
    '''
    
    df_impute = df.copy()
    
    qualitative_features_used, numeric_features_used = get_feature_types(df_impute)
    
    print('Imputing quantitative features...')
    cnter = 0
    bar = progressbar.ProgressBar(maxval=len(numeric_features_used)+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # impute median for missing values in quantitative features
    for feat in numeric_features_used:
        df_impute[feat] = df_impute[feat].fillna(df_impute[feat].median())
        cnter+=1 
        bar.update(cnter)
    bar.finish()
    
    print('Imputing qualitative features...')
    cnter = 0
    bar = progressbar.ProgressBar(maxval=len(qualitative_features_used)+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # impute mode (most frequent) for missing values in qualitative features
    for feat in qualitative_features_used:
        df_impute[feat] = df_impute[feat].fillna(df_impute[feat].mode().iloc[0])
        cnter+=1 
        bar.update(cnter)
    bar.finish()
    
    return df_impute


def transfrom_attribute_map(attr_mapping_df):
    '''Cleans the attr_mapping_df by filling the missing values. 
    
    ARGS
    ----
    attr_mapping_df: (pandas.DataFrame) Dataframe that has a Attribute and Meaning column

    RETURNS
    -------
    attr_mapping_clean: (pandas.DataFrame) Copy of attr_mapping_df where the nan values are filled with
        a forward fill and 'Meaning' matching 'unknown' rows have been removed
    '''
    
    attr_mapping_clean = attr_mapping_df.copy()
    attr_mapping_clean.fillna(method='ffill', inplace=True)

    attr_mapping_clean.drop(attr_mapping_clean[attr_mapping_clean['Meaning']=='unknown'].index, axis=0, inplace=True)

    return attr_mapping_clean


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


def transform_azdias(azdias_df, attr_mapping_df, top_level_attr_df, missing_cols_thresh=0.2):
    '''Clean the Udacity_AZDIAS dataset. Sets 'LNR' as index, removes 'EINGEFUEGT_AM' and 
    the features that have not been explained in attr_mapping_df or  top_level_attr_df.
    Cleans 'CAMEO_DEUG_2015' attribute where some data are falsely logged as 'X'.
    Decodes all the missing value encodings in the data as np.nan.
    Remove the features and rows where the missing values are above a given threshold.
    Finally impute missing values with most frequent if categorical or median if quantitative.
    Returns the cleaned dataframe.

    ARGS
    ----
    azdias_df: (pandas.DataFrame) Udacity_AZDIAS dataframe to be cleaned
    attr_mapping_df: (pandas.DataFrame)  Dataframe that has a Attribute and Meaning column
    top_level_attr_df: (pandas.DataFrame) Dataframe that has a Attribute column
    missing_cols_thresh: (float) Threshold to remove features with missing value ratio above

    RETURNS
    -------
    azdias_df_filtered: (pandas.DataFrame) Cleaned copy of Udacity_AZDIAS dataframe
    features_kept: (list) List of features in the returned azdias_df_filtered
    '''
    
    azdias_df_filtered = azdias_df.copy()
    all_attributes = get_all_attributes(attr_mapping_df, top_level_attr_df)
    unknown_mapping = get_unkown_mapping(attr_mapping_df)

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
    ratio_missing_cols = ratio_missing(azdias_df_filtered, axis=0)
    
    print('Removing missing values according to row and collumn thresholds...')
    azdias_df_filtered = remove_above_percent(azdias_df_filtered, ratio_missing_cols, missing_cols_thresh, axis=1)
    
    print('Imputing missing values...')
    azdias_df_filtered = impute_na(azdias_df_filtered)

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
    Finally impute missing values with most frequent if categorical or median if quantitative.
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
    unknown_mapping = get_unkown_mapping(attr_mapping_df)

    df_clean.set_index('LNR', inplace=True)
    df_clean['CAMEO_DEUG_2015'] = np.where(df_clean['CAMEO_DEUG_2015']=='X', np.NaN, df_clean['CAMEO_DEUG_2015'])
    
    print('Decoding and converting missing values to NaN...')
    df_clean = decode_missing_values(df_clean, unknown_mapping)
    
    print('Imputing missing values...')
    df_clean = impute_na(df_clean)
    
    print('Ratio of data used')
    print('features: %.2f' % (df_clean.shape[1]/df.shape[1]*100))
    print('observations: %.2f' % (df_clean.shape[0]/df.shape[0]*100))

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