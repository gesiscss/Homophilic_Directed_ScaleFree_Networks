################################################################################
# System dependencies
################################################################################
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
from joblib import delayed
from joblib import Parallel
from org.gesis.lib import io
from org.gesis.lib import utils
from collections import defaultdict
from sklearn.metrics import mean_absolute_error

################################################################################
# Constants
################################################################################
PERCENTAGE_RANGE = np.append([5], np.arange(10, 100 + 10, 10)).astype(np.float)
VALID_METRICS = ['indegree','outdegree','pagerank','wtf'] 

################################################################################
# Functions
################################################################################

def _rank(rank_dict):
    x_list = []
    y_list = []
    N = len(rank_dict.keys())

    for percentage in PERCENTAGE_RANGE:
        rank = percentage / 100.

        x_list.append(percentage)
        count_min = 0
        count_maj = 0

        for rank_index in sorted(rank_dict.keys(), reverse=False):
            count_min += rank_dict[rank_index].count('min')
            count_maj += rank_dict[rank_index].count('maj')
            if int(rank_index) > (rank * N):
                break

        y_list.append(float(count_min) / (count_min + count_maj))
    return x_list, y_list

def _rank_function_matrix(df, metric):
    '''
    df['minority'] is a binary value. 1 represents a minority node, and 0 majority.
    '''
    sorted_rnk = df.sort_values([metric], ascending=False)

    rank_val = 0
    rank_index_dict = {}
    rank_dict = defaultdict(list)
    count_all_min = 0
    count_all_maj = 0

    for index, row in sorted_rnk.iterrows():

        _val = row[metric]

        if _val in rank_index_dict.keys():
            rank_val = rank_index_dict[_val]
        else:
            rank_val += 1
            rank_index_dict[_val] = rank_val

        node_color = row['minority']
        if node_color == 1:
            count_all_min += 1
            rank_dict[rank_val].append('min')
        else:
            count_all_maj += 1
            rank_dict[rank_val].append('maj')

    minority_fraction = float(count_all_min) / float(count_all_min + count_all_maj)
    return _rank(rank_dict), minority_fraction

def _fm_function_matrix_rank(df, metric):
    
    # from larger values to smaller values
    sorted_rnk = sorted(df[metric].unique(),reverse=True) 
    total = len(sorted_rnk)
    fmts = []
    ranks = []
    
    for rank in PERCENTAGE_RANGE:
        k = round(rank/100.,2)    # k%
        t = int(round(k*total))   # No. of unique ranks in top-k
        topkrnk = sorted_rnk[0:t] # first top-k ranks (list)
        topnodes = df.query("{} in @topkrnk".format(metric))
        
        if topnodes.shape[0]==0:
            continue
            #fmts.append(None)
        else:
            ranks.append(rank)
            fmts.append(topnodes.minority.sum() / topnodes.shape[0])

    return ranks, fmts, fmts[-1]

def _gini_function_matrix_rank(df, metric):
    
    # from smallest values to largest values
    sorted_rnk = sorted(df[metric].unique(),reverse=True) 
    total = len(sorted_rnk)
    ginis = []
    ranks = []
    
    for rank in PERCENTAGE_RANGE:
        k = round(rank/100.,2)    # k%
        t = int(round(k*total))   # No. of unique ranks in top-k
        topkrnk = sorted_rnk[0:t] # first top-k ranks (list)
        values = df.query("{} in @topkrnk".format(metric))[metric].astype(np.float).values
        
        if values.shape[0]==0:
            continue
            #ginis.append(None)
        else:
            ranks.append(rank)
            ginis.append(utils.gini(values))
            
    return ranks, ginis

def _horizontal_inequalities_parallel(path, dataset, fn):
    
    fn = os.path.join(path,dataset,fn) if dataset is not None else os.path.join(path,fn)
    
    try:
        epoch = int(fn.split("-ID")[-1].split(".csv")[0])
    except:
        epoch = None
        
    if epoch is None and dataset is not None:
        # empirical
        fn_final = fn.replace(".csv","_rank.csv")
    else:
        # fit & synthetic
        fn_final = fn.replace(".csv","_rank.csv")

    if os.path.exists(fn_final):
        print("{} already exists.".format(fn_final))
        return 0

    cols = ['kind', 'metric', 'dataset', 'epoch', 'rank', 'fmt', 'gt', 'gini', 'mae', 'me']
    df_rank = pd.DataFrame(columns=cols)
    kind = path.split("/")[-2 if path.endswith("/") else -1]
    df_metadata = io.read_csv(fn)

    
    for metric in VALID_METRICS:
        
        if metric not in df_metadata:
                continue

        rank, gt = _gini_function_matrix_rank(df_metadata, metric)
        rank, fmt, minority_fraction = _fm_function_matrix_rank(df_metadata, metric)
        
        g = utils.gini(df_metadata[metric].astype(np.float).values)     # global vertical ineq.
        mae = mean_absolute_error([minority_fraction] * len(fmt), fmt)  # global horizontal ineq. using mae
        me = utils.mean_error([minority_fraction] * len(fmt), fmt)      # global horizontal ineq. using me

        #rank: rank position 5, 10, 20, ..., 100
        #fmt : fraction of minorities in top-k
        #gt  : gini coefficient of rank distribution up to top-k rank
        #gini: gini coefficient of rank distribution (from 5% to 100% rank) (gt when rank=100)
        #mae : mean absolute error of fmt curve (from 5% to 100% rank)
        #me  : mean error of fmt curve (from 5% to 100% rank)

        tmp = pd.DataFrame({'kind': kind,
                            'metric': metric,
                            'dataset': dataset,
                            'epoch':epoch,
                            'rank': rank,
                            'fmt': fmt,
                            'gt': gt,
                            'gini':g,
                            'mae':mae,
                            'me':me,
                           }, columns=cols)
        df_rank = df_rank[cols].append(tmp[cols], ignore_index=True)
        
    io.save_csv(df_rank, fn_final)
    return 1

    
def horizontal_inequalities_parallel(path, dataset=None):

    ### If it does not exist, it generates it from node metadata
    files = [fn for fn in os.listdir(os.path.join(path,dataset) if dataset is not None else path) 
             if fn.endswith(".csv") and not fn.endswith("_rank.csv") and not fn.endswith("_netmeta.csv")]
             #and (fn == "nodes_metadata.csv" or '-ID' in fn )]
    
    n_jobs = multiprocessing.cpu_count()
    print(n_jobs, len(files))
    results = Parallel(n_jobs=n_jobs)(delayed(_horizontal_inequalities_parallel)(path, dataset, fn) for fn in files)
    print("{} done | {} already done".format(sum(results), len(results)-sum(results)))
    
    