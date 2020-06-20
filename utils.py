import pandas as pd
import numpy as np
import progressbar


def decode_missing_values(df, unknown_mapping):
    '''Replaces the values encoded as unkown in df as np.nan. The encoding for values are 
    given in unknown_mapping.

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