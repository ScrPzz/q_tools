import numpy as np
import pandas as pd


class list_tools:
    def count_not_none(l=list):
        return np.count_nonzero(~np.isnan(l))

    def get_last_not_nan_id(l=list):
        return np.argwhere(~np.isnan(np.array(l)))[-1][0]
    
    def get_first_not_nan_id(l=list):
        return np.argwhere(~np.isnan(np.array(l)))[0][0]

    def count_nan_from_tail(l=list, n=int):
        return pd.Series(l[::-1][:n]).isna().sum()
    
    def count_nan_after_last_not_nan(l=list):
        return len(l)-list_tools.get_last_not_nan_id(l)-1
    
    def count_nan(l=list):
        return np.count_nonzero(np.isnan(l))
    
    def find_max_consecutive_nans(l=list):
        mask = np.concatenate(([False],np.isnan(l),[False]))
        if ~mask.any():
            return 0
        else:
            idx = np.nonzero(mask[1:] != mask[:-1])[0]
            return (idx[1::2] - idx[::2]).max()
        
    def count_internal_max_consecutive_nans(l=list):
        from_id=list_tools.get_first_not_nan_id(l)
        to_id=list_tools.get_last_not_nan_id(l)
        return list_tools.find_max_consecutive_nans(l[from_id:to_id])
    
    def batcherer(lst, n):
        batches=[]
        """Split lst in batches of dim n."""
        for i in range(0, len(lst), n):
            batches.append(lst[i:i + n])
        return batches






def interpolate_dict(X=list, strategy={'inside': 'cubicspline', 'outside': 'fill_with_avg_last_3_values'}, ext_lim=5, int_lim=5, avg_interval=int, avg_weights='omogeneous', **kwargs):
    """-----------------------input---------------------#
    * strategy :
     - 'inside': interpolation style for None-valued 
           datapoints that are surrounded by actual data.
           Methods are inherited from pandas interpolate;

     - 'outside': interpolation style for None-valued 
           datapoints that come after the last actual data:

         + 'fill_with_last_not_nan': fill the nan values 
             at the tail of the time series using the more
                recent non nan value.
         + fill_with_exp_damp_last_not_nan: still not 
             developed;
             
         + fill_with_avg_last_3_values: fill the nans in 
             tail with the avg of last 3 valid values

    * ext_lim: max number of NaNs after last value modified
    * int_lim: max number of internal NaNs modified
    * avg_interval= number of values the avg id calculated on
    * avg_weights= weights policy for the weighted average:
        + 'omogenous': all weights are set to 1.0
        + 'ascending': weights are linspace(1.0, scale , avg_interval, endpoint=False),
            with scale set in kwargs ore default= 1.5
            
    kwargs: 
        * scale = scale for linear weighting of average;
        * isolated_peaks_weight = scale factore for isolated 
            values.
        * exp_constant = constant for the exponential damper
        
    -----------------------output--------------------#
    results: interpolated series of dicts"""


    years=['2000', '2001', '2002', '2003', '2004', '2005', 
         '2006', '2007', '2008', '2009', '2010', '2011',
         '2012', '2013', '2014', '2015', '2016', '2017', 
         '2018', '2019', '2020', '2021', '2022']
    # Weights preparation
    
    if avg_weights=='omogeneous':
        weights=np.ones(avg_interval)
    if avg_weights=='ascending':
        if 'scale'in kwargs:
            scale=kwargs.get('scale', None)
        else:
            scale=1.5
        weights=np.linspace(1.0, scale , avg_interval, endpoint=False)
    

    results={}
    if list_tools.count_not_none(X) >=2:
        if list_tools.count_internal_max_consecutive_nans(X) <= int_lim :
            
            R=pd.Series(X).interpolate(method=strategy['inside'] , limit_area='inside')
            X=list(R.values)
            last_not_nan_id=list_tools.get_last_not_nan_id(X)
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            if list_tools.count_nan_after_last_not_nan(X)<= ext_lim:
                
                if strategy['outside']=='fill_with_last_not_nan': # OK
                    X[last_not_nan_id+1:]=[X[last_not_nan_id] for x in range(0,list_tools.count_nan_after_last_not_nan(X))]

                if strategy['outside']=='fill_with_exp_damp_last_not_nan': #OK
                    if 'exp_constant'in kwargs:
                        exp_constant=kwargs.get('exp_constant', None)
                    else:
                        exp_constant=0.3
                        
                    X[last_not_nan_id+1:]= np.logspace(.9, 0.3 , num=len(X)-last_not_nan_id -1, endpoint=False, base=X[last_not_nan_id])

                if strategy['outside']=='fill_with_avg_last_n_values': #OK                    
                    X[last_not_nan_id+1:]=[np.nanmean(np.array(X[last_not_nan_id - avg_interval +1: last_not_nan_id +1])*np.array(weights)) for x in range(0,list_tools.count_nan_after_last_not_nan(X))]

            
        if list_tools.count_internal_max_consecutive_nans(X) > int_lim:
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            #print('L1')
            last_not_nan_id=list_tools.get_last_not_nan_id(X)
            if list_tools.count_nan_after_last_not_nan(X) <= ext_lim:
                
                if strategy['outside']=='fill_with_last_not_nan': # OK
                    X[last_not_nan_id+1:]=[X[last_not_nan_id] for x in range(0,list_tools.count_nan_after_last_not_nan(X))]

                if strategy['outside']=='fill_with_exp_damp_last_not_nan': #OK
                    if 'exp_constant'in kwargs:
                        exp_constant=kwargs.get('exp_constant', None)
                    else:
                        exp_constant=0.3
                    X[last_not_nan_id+1:]= np.logspace(.9, exp_constant , num=len(X)-last_not_nan_id -1, endpoint=False, base=X[last_not_nan_id])

                if strategy['outside']=='fill_with_avg_last_n_values': #OK
                    if 'isolated_peaks_weight' in kwargs:
                        p_weight=kwargs.get('isolated_peaks_weight', None)
                    else: 
                        p_weight=.75
                    X[last_not_nan_id+1:]=[p_weight*X[last_not_nan_id] for x in range(0,list_tools.count_nan_after_last_not_nan(X))]


            elif list_tools.count_nan_after_last_not_nan(X) > ext_lim:
                pass
        results=dict(zip(years, X))
        del X
        return results


def limit_dict_values_to_range(d=dict, accepted_range=tuple):
    m=accepted_range([0])
    M=accepted_range([1])
    
    for k, v in d.items():
        if (v is not None) and (v >M):
            d[k]=M
        if (v is not None) and (v <m):
            d[k]=m
        else:
            pass
    return d