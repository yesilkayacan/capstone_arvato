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


def ratio_missing(df, plot=False):
    '''Calculate the ratio of missing values in each row and column. Returns the ratios with a descending order
    
    ARGS
    ----
    df: (Pandas DataFrame) DataFrame of interest to perform missing value analysis on
    plot: (bool) It true the results are plotted as a histogram
    
    RETURNS
    -------
    ratio_missing_rows: (pandas Serie) Series of sorted missing value ratios per each row
    ratio_missing_cols: (pandas Serie) Series of sorted missing value ratios per each column
    '''
    
    n_rows = df.shape[0]
    n_cols = df.shape[1]
    
    n_missing_cols = df.isnull().sum(axis=0) #number of missing values per columns
    ratio_missing_cols = n_missing_cols/n_rows #number of missing values per column divided by number of rows
    
    n_missing_rows = df.isnull().sum(axis=1) #number of missing values per row
    ratio_missing_rows = n_missing_rows/n_cols #number of missing values per row divided by number of columns
    
    ratio_missing_cols = ratio_missing_cols.sort_values(ascending=False)
    ratio_missing_rows = ratio_missing_rows.sort_values(ascending=False)
    
    if plot:
        plt.figure(figsize=(14,10))
        plt.subplot(2,1,1)
        bins = np.arange(0, 1, step=0.02)
        ratio_missing_cols.hist(bins=bins)
        plt.xlabel('Ratio of Missing Values')
        plt.ylabel('Number of Features')
        plt.title('Ratio of Missing Value per Feature')
        plt.xlim((0, 1))
        
        plt.subplot(2,1,2)
        ratio_missing_rows.hist(bins=bins)
        plt.xlabel('Ratio of Missing Values')
        plt.ylabel('Number of Rows')
        plt.title('Ratio of Missing Value per Row')
        plt.xlim((0, 1))
    
    return ratio_missing_rows, ratio_missing_cols


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