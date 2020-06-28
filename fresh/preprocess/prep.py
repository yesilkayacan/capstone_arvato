


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
    
    qualitative_features_used, numeric_features_used = AttributeMapping.get_feature_types(df_impute)
    
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