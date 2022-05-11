import pandas as pd 
import numpy as np
from collections import OrderedDict


def extract_history(df, quantity_name=str, apply_log= bool):
     
    """if apply_log==True:
        outputs: ID | log__history | history | 
    else:
        outputs: ID | history 
    """
    
    def fill_and_sort_series_of_dicts(S, complete=bool, replace_nan=bool, replace_nan_with=float):
        if len(S)!=0:
            
            if not complete:
                L = S.to_list()
                empty = dict.fromkeys(set().union(*L), np.nan)
                partial=[dict(empty, **d) for d in L]
                sorted_partial = [dict(sorted(d.items())) for d in partial]
                return sorted_partial
            
            if complete:
                L = S.to_list()
                max_key = max(max(d) for d in L)
                min_key = min(min(d) for d in L)
                keys=[x for x in range(int(min_key), int(max_key)+1)]
                keys=[str(x) for x in keys]
                empty=dict.fromkeys(keys, np.nan)
                partial=[dict(empty, **d) for d in L]
                sorted_partial = [dict(sorted(d.items())) for d in partial]
                
                return sorted_partial
        


    t_ls=[]
    full_year_list=list(range(np.nanmin(df['auct_year']), np.nanmax(df['auct_year'])))
    df_aux=df[['auct_year', quantity_name]]
    df_aux['auct_year']=df_aux['auct_year'].apply(str)
    auxil=pd.DataFrame(df_aux.groupby('artistId')['auct_year'].apply(list))
    auxil=auxil.merge(pd.DataFrame(df_aux.groupby('artistId')[quantity_name].apply(list)), on='artistId')
    
    # Calc
    if apply_log:
       
        auxil[f'log_{quantity_name}_history']=auxil.apply(
            lambda x: dict(zip(x['auct_year'], np.log(x[quantity_name]))), axis=1)
    
    auxil[f'{quantity_name}_history']=auxil.apply(
        lambda x: dict(zip(x['auct_year'], x[quantity_name])), axis=1)
    
    # sort/fill
    if apply_log:
    
        auxil[f'log_{quantity_name}_history']=pd.Series(auxil[f'log_{quantity_name}_history'].apply(
            lambda x: OrderedDict(sorted(x.items()))))

        auxil[f'log_{quantity_name}_history']=fill_and_sort_series_of_dicts(
            auxil[f'log_{quantity_name}_history'], complete=False, replace_nan=False)
    
    auxil[f'{quantity_name}_history']=pd.Series(auxil[f'{quantity_name}_history'].apply(
        lambda x: OrderedDict(sorted(x.items()))))
    
    auxil[f'{quantity_name}_history']=fill_and_sort_series_of_dicts(
        auxil[f'{quantity_name}_history'], complete=False, replace_nan=False)

    auxil=auxil.drop(['auct_year', quantity_name], axis=1) 
    #df=df.drop(t_ls, axis=1)
    del df_aux
    
    return auxil


def rolling_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def fill_and_sort_series_of_dicts(S, complete=bool, replace_nan=bool, replace_nan_with=float):
        if len(S)!=0:
            
            if not complete:
                L = S.to_list()
                empty = dict.fromkeys(set().union(*L), np.nan)
                partial=[dict(empty, **d) for d in L]
                sorted_partial = [dict(sorted(d.items())) for d in partial]
                return sorted_partial
            
            if complete:
                L = S.to_list()
                max_key = max(max(d) for d in L)
                min_key = min(min(d) for d in L)
                keys=[x for x in range(int(min_key), int(max_key)+1)]
                keys=[str(x) for x in keys]
                empty=dict.fromkeys(keys, np.nan)
                partial=[dict(empty, **d) for d in L]
                sorted_partial = [dict(sorted(d.items())) for d in partial]
                
                return sorted_partial



