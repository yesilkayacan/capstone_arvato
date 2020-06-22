import pandas as pd
import numpy as np
import progressbar

import matplotlib.pyplot as plt

def decode_missing_values(df, unknown_mapping):
    '''Replaces the values encoded as unkown in df as np.nan. The encoding for unkown values are 
    given in unknown_mapping

    ARGS
    ----
    df: (pandas.DataFrame) Dataframe where the encoded unkown values will be decoded as np.nan
    unkown_mapping: (pandas.Series) Series mapping the unknown encodings. Index is the featurea and 
    the value is the encoded value for unkowns. The encoding is a string of values seperated by ','. Eg. '-1, 9'
    
    RETURNS
    -------
    df_clean: (pandas.DataFrame) Copy of df where the unknown values are decoded as np.nan
    '''

    df_clean = df.copy()
    
    features = list(set(df_clean.columns).intersection(unknown_mapping.index.values))
    
    cnter = 0
    bar = progressbar.ProgressBar(maxval=len(features)+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for feat in features:
        # loop through features both in unknown maoppings and the dataframe
        for unknown in unknown_mapping[feat].split(','):
            df_clean[feat] = np.where(df_clean[feat]==int(unknown), np.NaN, df_clean[feat]) # find and replace all unknown encodings with nan
        
        cnter+=1 
        bar.update(cnter)
    
    bar.finish()
    
    return df_clean


def ratio_missing(df, axis):
    '''Calculate the ratio of missing values in specified axis. Returns the ratios with a descending order
    
    ARGS
    ----
    df: (Pandas DataFrame) DataFrame of interest to perform missing value analysis on
    axis: (integer) 1 for rows and 0 for columns
    
    RETURNS
    -------
    ratio_missing_rows: (pandas Serie) Series of sorted missing value ratios per
    '''
    
    n = df.shape[axis]
    
    n_missing = df.isnull().sum(axis=axis) #number of missing values per columns

    ratio_missing = n_missing/n #number of missing values per column divided by number of rows

    ratio_missing = ratio_missing.sort_values(ascending=False)
    
    return ratio_missing


def remove_above_percent(df, missing_ratios, percent, axis=1):
    '''Removes the row or column from the dataframe where the missing data ratio 
    is greater then a specified threshold
    
    ARGS
    ----
    df: (Pandas DataFrame) Dataframe to be cleaned
    missing_ratios: (List[float]) List of ratio for missing data
    percent: (float) Missing data threshold. Data below the threshold will be kept
    axis: (integer) 0 for rows and 1 for columns
    
    RETURNS
    -------
    cleaned_df: (Pandas DataFrame) Cleaned dataframe
    '''
    
    cleaned_df = df.drop(list(missing_ratios[missing_ratios > percent].index), axis=axis)
    
    return cleaned_df


def categorize(df, categorical_mapping):
    '''One hot encoder, encodes the dataframe categorical attributes and returns the encoded dataframe.
    
    ARGS
    ----
    df: (pandas.DataFrame) Dataframe which the features will be encoded
    categorical_mapping: (pandas.DataFrame) Dataframe with the encoding information. 
        The dataframe should have a 'Attribute' column consisting of feature names and 'Value' 
        column with individual categories of that attribute. Each category should be a individual row in the dataframe.

    RETURNS
    -------
    categorized_df: (pandas.DataFrame) Copy of df where all the categorical features have been one hot encoded.
    '''
    
    categorized_df = df.copy()
    
    feature_list = categorical_mapping['Attribute'].unique()
    
    cnter = 0
    bar = progressbar.ProgressBar(maxval=len(feature_list)+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for feat in feature_list:
        
        try:
            categories = categorical_mapping[categorical_mapping['Attribute']==feat]['Value'].values.astype(int)
            categorized_df[feat] = categorized_df[feat].astype(int)
        except:
            categories = categorical_mapping[categorical_mapping['Attribute']==feat]['Value'].values
        
        dummies = pd.get_dummies(categorized_df[feat], drop_first=False, prefix=feat, prefix_sep='_d_')
        
        not_categorized = np.setdiff1d(categories, categorized_df[feat].unique())
        if not_categorized is not None:
            for ncat in not_categorized:
                dummies[(feat+'_d_'+str(ncat))] = 0
            
        try:
            categorized_df = categorized_df.join(dummies)
            categorized_df.drop(feat, axis=1, inplace=True)
        except:
            print('Error during encoding')
            print('Error feature: {}'.format(feat))
            print('Encoded top 10 rows as:')
            print(dummies.head(10))
            break
    
        # Update the progress bar
        cnter+=1 
        bar.update(cnter)
        
    bar.finish()

    return categorized_df