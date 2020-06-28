import numpy as np
import progressbar
import pandas as pd
import copy
import pickle


class Data_Correction():
    
    def __init__(self, mapping_obj):
        
        self.mapping = mapping_obj
        
    
    def scan_irregularities(self, df):
        '''

        '''
        
        known_mapping = copy.deepcopy(self.mapping.known_mapping)
        unknown_mapping = copy.deepcopy(self.mapping.unknown_mapping)

        check_dict = mergeDict(known_mapping, unknown_mapping)
        [check_dict[x].append(np.nan) for x in check_dict.keys()]
        
        all_keys = check_dict.keys()
        
        feature_investigate = set(df.columns).intersection(all_keys)
        
        cnter = 0
        bar = progressbar.ProgressBar(maxval=len(feature_investigate)+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        issues = dict([])
        for feature in feature_investigate:
            
            irregular_found = df[feature][~df[feature].isin(check_dict[feature])].unique()
            
            if len(irregular_found)!=0:
                issues[feature] = irregular_found
            
            cnter+=1 
            bar.update(cnter)

        bar.finish()
        
        return issues
    
    
    def decode_missing_values(self, df):
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

        features_mapped = list(set(df_clean.columns).intersection(self.mapping.unknown_mapping.keys()))

        cnter = 0
        bar = progressbar.ProgressBar(maxval=len(features_mapped)+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for feat in features_mapped:
            # loop through features both in unknown maoppings and the dataframe
            for val in self.mapping.unknown_mapping[feat]:
                df_clean[feat] = np.where(df_clean[feat]==val, np.NaN, df_clean[feat]) # find and replace all unknown encodings with nan

            cnter+=1 
            bar.update(cnter)

        bar.finish()

        return df_clean


    def fix_edge_cases(df):
        '''
        
        '''
        
        df['CAMEO_DEUG_2015'] = np.where(df['CAMEO_DEUG_2015'].isin(['X', 'XX']), np.NaN, df['CAMEO_DEUG_2015'])
        df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].astype(float)
        
        df['CAMEO_DEU_2015'] = np.where(df['CAMEO_DEU_2015'].isin(['X', 'XX']), np.NaN, df['CAMEO_DEU_2015'])


class AttributeMapping():
    '''
    
    '''
    
    UNKNOWN_DETECTION_KEYWORDS = ['unknown', 'unknown / no main age detectable', 'no transaction known']
    
    def __init__(self, attribute_map_file):
        
        self.attr_mapping_df = AttributeMapping._get_clean_df(attribute_map_file)
        self.defined_attributes = list(self.attr_mapping_df['Attribute'].unique())
        
        
        self.unknown_mapping = self.get_unkown_mapping(self.attr_mapping_df)
        self.known_mapping = self.get_known_mapping(self.attr_mapping_df)
        
        
    def _get_clean_df(attribute_map_file):
        '''
        
        '''
        
        attr_mapping_df = pd.read_excel(attribute_map_file, header=1, dtype=str)
        del attr_mapping_df['Unnamed: 0']
        
        attr_mapping_clean_df = AttributeMapping._transfrom_attribute_map(attr_mapping_df)
        
        return attr_mapping_clean_df
        
        
    def _transfrom_attribute_map(attr_mapping_df):
        '''Cleans the attr_mapping_df by filling the missing values. 

        ARGS
        ----
        attr_mapping_df: (pandas.DataFrame) Dataframe that has a Attribute and Meaning column

        RETURNS
        -------
        attr_mapping_clean: (pandas.DataFrame) Copy of attr_mapping_df where the nan values are filled with
            a forward fill
        '''

        attr_mapping_clean = attr_mapping_df.copy()
        attr_mapping_clean.fillna(method='ffill', inplace=True)

        return attr_mapping_clean

        
    def get_unkown_mapping(self, df):
        '''Creates a dataframe of the 'Attribute'-'Meaning' couples where the represents unknown

        ARGS
        ----
        df: (pandas.DataFrame) Dataframe that has a Attribute and Meaning column

        RETURNS
        -------
        unknown_mapping: (dict) Dictionary of the 'Attribute'-'Meaning' couples where the 'Meaning'
            represents unknown
        '''
        
        unknown_mapping = df[df['Meaning'].isin(self.UNKNOWN_DETECTION_KEYWORDS)].set_index('Attribute')['Value'].apply(lambda x: [int(x) for x in x.split(',')]).to_dict()
        
        return unknown_mapping
    
    
    def get_known_mapping(self, df):
        '''Creates a dataframe of the 'Attribute'-'Meaning' couples where the 'Meaning' is defined
        
        ARGS
        ----
        df: (pandas.DataFrame) Dataframe that has a Attribute and Meaning column

        RETURNS
        -------
        known_mapping: (dict) Dictionary of the 'Attribute'-'Meaning' couples where the 'Meaning'
            is defined
        '''
        
        known_mapping = df[~(df['Meaning'].isin(self.UNKNOWN_DETECTION_KEYWORDS) | df['Meaning'].str.contains('numeric'))].groupby('Attribute')['Value'].apply(list).to_dict()
        
        for key, val in known_mapping.items():
            
            try:
                known_mapping[key] = list(map(int, val))
            except:
                pass
        
        return known_mapping
    
    
    def add_to_unknown_mapping(self, addition):
        '''
        
        '''
        
        self.unknown_mapping = mergeDict(self.unknown_mapping, addition)


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


# source https://thispointer.com/how-to-merge-two-or-more-dictionaries-in-python/
def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = [*value , *dict1[key]]
            
    return dict3


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


def etl_transform(df, attr_mapping, ref_cols=None):
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
    
    df_clean = df.copy()
    
    print('Correcting issues on edge cases...')
    Data_Correction.fix_edge_cases(df_clean)
    
    print('Checking for irregular values...')
    corrector = Data_Correction(attr_mapping)
    irregular_values = corrector.scan_irregularities(df_clean)
    attr_mapping.add_to_unknown_mapping(irregular_values)
    
    print('Decoding missing or unknown values as NaN...')
    df_clean = corrector.decode_missing_values(df_clean)
    
    if ref_cols is None:
        print('Finding the features to remove...')
        attr_not_mapped = set(np.setdiff1d(df_clean.columns, attr_mapping.defined_attributes))
        drop_set = attr_not_mapped.copy()
        
        ratio_missing_cols = ratio_missing(df_clean, axis=0)
        above_missing_thresh = ratio_missing_cols[ratio_missing_cols>0.2]
        drop_set = drop_set.union(above_missing_thresh.index) 
        drop_set.remove('LNR')
        df_clean.drop(drop_set, axis=1, inplace=True)
    
    else:
        print('getting the subset of the data with the reference features...')
        df_clean = df_clean[ref_cols]
    
    print('Imputing missing values...')
    df_clean = impute_na(df_clean)

    print('Ratio of data used')
    print('features: %.2f' % (df_clean.shape[1]/df.shape[1]*100))

    return df_clean


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


def categorize(df, attribute_mapping):
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
    
    feature_list = list(set(categorized_df.columns).intersection(attribute_mapping.keys()))
    
    cnter = 0
    bar = progressbar.ProgressBar(maxval=len(feature_list)+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for feat in feature_list:
        
        try:
            categories = attribute_mapping[feat]
            categorized_df[feat] = categorized_df[feat].astype(int)
        except:
            categories = attribute_mapping[feat]
        
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