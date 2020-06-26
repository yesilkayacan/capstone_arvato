import numpy as np
import progressbar


class Missing_val():
    
    def __init__(self, mapping_obj):
        
        self.mapping = mapping_obj
        
    
    def scan_irregularities(self, df):
        '''

        '''
        
        check_dict = mergeDict(self.mapping.known_mapping, self.mapping.unknown_mapping)
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
    
    
# source https://thispointer.com/how-to-merge-two-or-more-dictionaries-in-python/
def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = [*value , *dict1[key]]
            
    return dict3