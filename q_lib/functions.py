import aux
import pandas as pd
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import numpy as np

def extract_established_authors(quantiles_data=pd.DataFrame, from_year=int, to_year=int, value=float, **kwargs):
    
    """---------------------------------------------------------------------
    Locates authors that remains stationary in a quantile for a certain time interval. 
    Eventually locate the ones that after being stationary for a while are getting a jump
    
    Return 
    - list of ids of well established (static quantile across) authors from year to year;
    if tolerance is set the staticity will be calculated across the range [value-tolerance, value+tolerance]
    - lists of quantiles jumping authors in (to_year, to_year+1) interval:
    - Jumping up: jump > than jump_threshold
    - Jumping down: jump < than jump_threshold
        
    If not set by 'min_abs_jump_threshold' kwarg, default=2.0
    
    ----------------------------------------------------------------------"""
    
    # Established authors     
    quantiles_data_subset=quantiles_data.apply(lambda x: aux.subset_dict(d=x, from_year=from_year, to_year=to_year))
    
    if 'tolerance' in kwargs:
        tolerance=kwargs.get('tolerance', None)
        established_authors=quantiles_data_subset.apply(lambda x: aux.cfr_dict_with_range_of_values(x, value=value, semi_interval=tolerance))
    else:
        established_authors=quantiles_data_subset.apply(lambda x: aux.cfr_dict_with_value(d=x, value=value))

    established_authors=established_authors[established_authors==True].index
    del quantiles_data_subset

    # Jumps
    quantiles_data_subset=quantiles_data.loc[established_authors].apply(lambda x: aux.subset_dict(d=x, from_year=to_year, to_year=to_year+1))
    jumps=quantiles_data_subset.apply(lambda z: aux.extract_jump(z))
    
    if 'min_abs_jump_threshold' in kwargs:
        jump_thr=kwargs.get('min_abs_jump_threshold', None)
    else:
        jump_thr=2.0

    jumping_up=jumps[jumps>=jump_thr]
    jumping_down=jumps[jumps<0.0]
    
    if 'ALL' in kwargs:
        extract_established_authors(quantiles_data=pd.DataFrame, from_year=int, to_year=int, value=float)
        
    return established_authors, jumping_up, jumping_down

#------------------------------------------------------------------#


def add_qtiles_jump_over_years(q_tilized_data, from_year=int, to_year=int, column_name=str):
    """ 
    Add a "{from_year}_to_{to_year}_jump" column to the input dataframe, type int.
    """

    q_tilized_data[f'{column_name}_aux']=q_tilized_data[column_name].apply(
        lambda x: aux.subset_dict(x, str(from_year), str(to_year)))
    
    q_tilized_data[f'{from_year}_to_{to_year}_jump']=q_tilized_data[f'{column_name}_aux'].apply(lambda z: aux.extract_jump(z))
    
    q_tilized_data=q_tilized_data.drop(f'{column_name}_aux', axis=1)
    return q_tilized_data


#------------------------------------------------------------------#

def extract_qtiles_communities_over_time(q_tilized_data, from_year=int, to_year=int, threshold=float):
    
    interval=list(range(int(from_year), int(to_year)))
    Q_permanences=[]
    
    for i in q_tilized_data.index:
        D=q_tilized_data.loc[str(i)]

        if D is not None:
            D_subset=dict((str(k), D[str(k)]) for k in interval)
            value, count = Counter(D_subset.values()).most_common(1)[0]

            if float(count/len(interval)) > threshold:
                Q_permanences.append((value, str(i)))
            else:
                continue
        
                
    tile_to_ids = {}
    for x, y in Q_permanences:
        tile_to_ids.setdefault(x, []).append(y)
    tile_to_ids=dict(OrderedDict(sorted(tile_to_ids.items())))
    
    #Plot
    ll=[]
    for t in tile_to_ids.values():
        ll.append(len(t))
        
    f=plt.figure(figsize=(7,5))
    f.suptitle(f'{from_year}_to_{to_year}_quantile_permanences')
    plt.xlabel('Quantile')
    plt.ylabel('Permanences')
    plt.plot(tile_to_ids.keys(),ll)
    del Q_permanences, ll
    return tile_to_ids

#------------------------------------------------------------------#

def extract_q_tile_history(df, quantity_name=str, apply_log= bool, q_tile=int):
    """
    if apply_log==True:
        outputs: ID | log__history | history | log_qtiles_history	| qtiles_history 
    else:
        outputs: ID | history | qtiles_history 
    """
    
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

        auxil[f'log_{quantity_name}_history']=aux.fill_and_sort_series_of_dicts(
            auxil[f'log_{quantity_name}_history'], complete=False, replace_nan=False)
    
    auxil[f'{quantity_name}_history']=pd.Series(auxil[f'{quantity_name}_history'].apply(
        lambda x: OrderedDict(sorted(x.items()))))
    
    auxil[f'{quantity_name}_history']=aux.fill_and_sort_series_of_dicts(
        auxil[f'{quantity_name}_history'], complete=False, replace_nan=False)

    # Qs logarithm()
    if apply_log:
        A=auxil[f'log_{quantity_name}_history'].apply(pd.Series)
    else:
        A=auxil[f'{quantity_name}_history'].apply(pd.Series)
        
    A.replace(to_replace=0.0, value=np.nan, inplace=True)
    quantiles_list={}
    for y in full_year_list:
        if apply_log:
            df_aux[f'log_{quantity_name}_{y}_{q_tile}tile'], quantile_list =pd.Series(pd.qcut(A[f'{y}'], q=q_tile, labels=False, retbins=True))
            t_ls.append(f'log_{quantity_name}_{y}_{q_tile}tile')
            quantiles_list[f'{y}']=quantile_list
        else:
            df_aux[f'{quantity_name}_{y}_{q_tile}tile'], quantile_list =pd.Series(pd.qcut(A[f'{y}'], q=q_tile, labels=False, retbins=True))
            t_ls.append(f'{quantity_name}_{y}_{q_tile}tile')
            quantiles_list[f'{y}']=quantile_list

    tiles_hist_list=pd.Series(df_aux[t_ls].values.tolist())
    tiles_hist_list.index=df_aux.index
    tiles_hist_list=pd.DataFrame(tiles_hist_list, columns=['tiles_hist_list'])

    auxil=auxil.merge(tiles_hist_list, on='artistId')
    auxil = auxil[~auxil.index.duplicated(keep='first')]
    
    if apply_log:
        auxil[f'log_{quantity_name}_{q_tile}iles_history']=auxil.apply(lambda x: dict(zip(full_year_list, x['tiles_hist_list'])), axis=1)
        auxil[f'log_{quantity_name}_{q_tile}iles_history']=pd.Series(auxil[f'log_{quantity_name}_{q_tile}iles_history'].apply(
            lambda x: dict(OrderedDict(sorted(x.items())))))
        
        auxil[f'{quantity_name}_{q_tile}iles_history']=auxil.apply(lambda x: dict(zip(full_year_list, x['tiles_hist_list'])), axis=1)
        auxil[f'{quantity_name}_{q_tile}iles_history']=pd.Series(auxil[f'{quantity_name}_{q_tile}iles_history'].apply(
            lambda x: dict(OrderedDict(sorted(x.items())))))
    else:
        auxil[f'{quantity_name}_{q_tile}iles_history']=auxil.apply(lambda x: dict(zip(full_year_list, x['tiles_hist_list'])), axis=1)
        auxil[f'{quantity_name}_{q_tile}iles_history']=pd.Series(auxil[f'{quantity_name}_{q_tile}iles_history'].apply(
            lambda x: dict(OrderedDict(sorted(x.items())))))

    auxil=auxil.drop(['auct_year', quantity_name, 'tiles_hist_list'], axis=1) 
    #df=df.drop(t_ls, axis=1)
    del df_aux, tiles_hist_list
    
    return auxil, A, quantiles_list

#------------------------------------------------------------------#

def extract_history(df, quantity_name=str, apply_log= bool):
     
    """if apply_log==True:
        outputs: ID | log__history | history | 
    else:
        outputs: ID | history 
    """
    
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

        auxil[f'log_{quantity_name}_history']=aux.fill_and_sort_series_of_dicts(
            auxil[f'log_{quantity_name}_history'], complete=False, replace_nan=False)
    
    auxil[f'{quantity_name}_history']=pd.Series(auxil[f'{quantity_name}_history'].apply(
        lambda x: OrderedDict(sorted(x.items()))))
    
    auxil[f'{quantity_name}_history']=aux.fill_and_sort_series_of_dicts(
        auxil[f'{quantity_name}_history'], complete=False, replace_nan=False)

    auxil=auxil.drop(['auct_year', quantity_name], axis=1) 
    #df=df.drop(t_ls, axis=1)
    del df_aux
    
    return auxil

#------------------------------------------------------------------#

def quantile_transition_by_ids(df, from_year=int, to_year=int, alluvial_graph=bool, qtiles_n=int):
    
    from_distr=aux.extract_bins_to_ids_dict(np.log(df[df['auct_year']==from_year]['avg_estimate']), bin_n=20, do_qtiles=True, qtile=qtiles_n)
    to_distr=aux.extract_bins_to_ids_dict(np.log(df[df['auct_year']==to_year]['avg_estimate']), bin_n=20, do_qtiles=True, qtile=qtiles_n)
    alluvial_data={}
    aux={}
    V=[]
    
    for k, v in from_distr.items():
        for i in v:
            V.append(i)
    A=tuple(V)
    V=[]
    for k, v in to_distr.items():
        for i in v:
            V.append(i)
            
    B=tuple(V)
    full_shared_artist_ids=set(A) & set(B)
    lost_artists= list(set(A).difference(B))
    infos_loss=len(lost_artists)/len(A)
    del A,B,V
    print(f'The chosen years are missing shared infos for {round(infos_loss*100, 2)}% of the authors')
    
    for k_0, v_0 in from_distr.items():
        a={}
        for k_1, v_1 in to_distr.items():
            
            shared_ids=list(set(from_distr[k_0]) & set(to_distr[k_1]))
            aux[f'{k_0}_to_{k_1}']= {'shared_n': len(shared_ids), 'shared_ids': shared_ids}
            a[str(f'{k_1}')] = float( len(shared_ids) /len(from_distr[k_0]))
                            
        alluvial_data[str(f'_{k_0}')]=a
        
    # alluvial plot
    if alluvial_graph:
        import alluvial
        cmap = plt.cm.get_cmap('jet')
        ax = alluvial.plot(alluvial_data, alpha=0.7, color_side=1, rand_seed=7, wdisp_sep=' '*2, cmap=cmap, fontname='Monospace',
        labels=('starting_quantile', 'arriving_quantile'), label_shift=1)
        fig = ax.get_figure()
        fig.set_size_inches(9,9)
        ax.set_title(f'Transitions from {from_year} to {to_year}')
        plt.show()
    
    return aux, alluvial_data, infos_loss, lost_artists
