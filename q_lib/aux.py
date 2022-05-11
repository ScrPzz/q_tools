from collections import Counter, OrderedDict
import numpy as np


def cfr_dict_with_value(d=dict, value=float):
    aux=dict(Counter(d.values()))
    if value in list(aux.keys()):
        if aux[value]== len(d):
                return True
        else:
            return False
    else: 
        return False
  
#------------------------------------------------------------------#
  
def cfr_dict_with_range_of_values(d=dict, value=float, semi_interval=float):
    aux=dict(Counter(d.values()))
    c=0
    if len(set([value-semi_interval, value, value+ semi_interval]) & set( list(aux.keys())))>0:
        for k, v in aux.items():
            if min(value-semi_interval, value+semi_interval) <= k <= max(value-semi_interval, value+semi_interval):
                c+=1
            else:
                continue
            pass
        
        if c==len(d):
            return True
        else:
            return False
 
 #------------------------------------------------------------------#

def extract_jump(d):
    k=list(d.keys())
    return d[k[1]] - d[k[0]]

#------------------------------------------------------------------#

def subset_dict(d, from_year, to_year):
    if type(list(d.keys())[0])==str:
        
        keys_subset=list(range(int(from_year), int(to_year) +1))
        keys_subset=[str(x) for x in keys_subset]
        d_subset={k: d[k] for k in keys_subset}
        
    if type(list(d.keys())[0])==int:
        keys_subset=list(range(from_year, to_year +1))
        keys_subset=[x for x in keys_subset]
        d_subset={k: d[k] for k in keys_subset}
    
    
    return d_subset

#------------------------------------------------------------------#

def round_dict_values(d):
    aux=d
    for k, v in d.items():
        if v is not None:
            aux[k]=np.abs(np.round(v))
        else:
            continue
    return aux

#------------------------------------------------------------------#

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

#------------------------------------------------------------------#

def extract_bins_to_ids_dict(data_w_id, year=int, bin_n=int, do_qtiles=bool, qtile=int ):
    
    if do_qtiles:
        bin_list=list(np.nanpercentile(data_w_id, np.arange(0, 100, int(100/bin_n))))
        bin_list.append(np.nanmax(data_w_id))
    else:
        bin_list = np.linspace(np.nanmin(data_w_id),np.nanmax(data_w_id), bin_n)
    intervs=list(zip(bin_list[::1],bin_list[1::1]))
    R={}
    for b, v in enumerate(intervs):
        L=[]
        for i in data_w_id.index:
            if v[0] <= float(data_w_id.loc[str(i)]) < v[1]:
                L.append(i)
        R[str(b)]=L
    return R