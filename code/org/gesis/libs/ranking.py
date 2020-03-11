import os
import warnings
import operator
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from deprecated import deprecated
from collections import defaultdict
from sklearn.metrics import mean_absolute_error

from org.gesis.libs.utils import gini
from org.gesis.libs.io import save_csv
from org.gesis.libs.io import read_csv
from org.gesis.libs.utils import printf

PERCENTAGE_RANGE = np.append([5], np.arange(10, 100 + 10, 10)).astype(np.float)
VALID_METRICS = ['indegree','outdegree','pagerank', 'circle_of_trust', 'wtf']


def _rank(rank_dict):
    x_list = [];
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


def _rank_function_graph(G, metric, minority=None):
    if metric == 'degree':
        rank = dict(G.degree())
    elif metric == 'indegree':
        rank = dict(G.in_degree())
    elif metric == 'outdegree':
        rank = dict(G.out_degree())
    elif metric == 'pagerank':
        rank = nx.pagerank(G)
    else:
        raise ValueError('metric does not exist: {}'.format(metric))

    sorted_rnk = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
    del (rank)

    rank_val = 0
    rank_index_dict = {}
    rank_dict = defaultdict(list)
    count_all_min = 0;
    count_all_maj = 0

    for node_index, _val in sorted_rnk:

        if _val in rank_index_dict.keys():
            rank_val = rank_index_dict[_val]
        else:
            rank_val += 1
            rank_index_dict[_val] = rank_val

        try:
            node_color = G.node[node_index]['color']
        except:
            try:
                node_color = G.node[node_index][G.graph['class']]
            except Exception as e:
                print(ValueError("error: {}".format(e)))
                return (None, None), None

        node_color = str(node_color)

        if minority is None:
            if node_color in ['min', 'red', 'minority', '1', True, 'True']:
                count_all_min += 1
                rank_dict[rank_val].append('min')
            else:
                count_all_maj += 1
                rank_dict[rank_val].append('maj')
        else:
            if node_color == str(minority):
                count_all_min += 1
                rank_dict[rank_val].append('min')
            else:
                count_all_maj += 1
                rank_dict[rank_val].append('maj')

    minority_fraction = float(count_all_min) / float(count_all_min + count_all_maj)
    return _rank(rank_dict), minority_fraction



def _rank_function_matrix(df, metric):
    '''
    df['minority'] is a binary value. 1 represents a minority node, and 0 majority.
    '''
    sorted_rnk = df.sort_values([metric], ascending=False)

    rank_val = 0
    rank_index_dict = {}
    rank_dict = defaultdict(list)
    count_all_min = 0;
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

def _validate_metric(metric):
    if metric not in VALID_METRICS:
        raise ValueError('metric {} is not supported. Please choose from: {}'.format(metric, VALID_METRICS))

def rank_vh_inequalities_synthetic(model, output):

    fntr = 'rank_synthetic_{}.csv'.format(model)
    fntr = os.path.join(output, fntr)

    if os.path.exists(fntr):
        return read_csv(fntr)

    cols = ['kind', 'metric', 'N', 'kmin', 'fm', 'hMM', 'hmm', 'gm', 'gM', 'gini', 'mae', 'epoch', 'rank', 'fmt']
    df_rank = pd.DataFrame(columns=cols)

    path = os.path.join(output, 'synthetic', model)
    for fn in os.listdir(path):
        if fn.endswith('.csv'):
            #DHBA-N2000-kmin2-fm0.5-hMM1.0-hmm1.0-ID7.csv
            fn = os.path.join(path, fn)
            metadata = pd.read_csv(fn, index_col=0)
            N = int(fn.split("-N")[-1].split('-kmin')[0])
            kmin = int(fn.split("-kmin")[-1].split('-fm')[0])
            fm = float(fn.split("-fm")[-1].split('-hMM')[0])
            hMM = float(fn.split("-hMM")[-1].split('-hmm')[0])
            hmm = float(fn.split("-hmm")[-1].split('-ID')[0])
            epoch = int(fn.split('-ID')[-1].replace('.csv', ''))
            gM = 2.5  # @todo: take this from filename
            gm = 2.5 # @todo: take this from filename

            for metric in VALID_METRICS:
                if metric not in metadata.columns:
                    warnings.warn('metric {} does not exist for synthetic {}.'.format(metric, model), UserWarning)
                else:
                    (rank, fmt), minority_fraction = _rank_function_matrix(metadata, metric)

                    g = gini(metadata[metric].astype(np.float).values) # vertical ineq.
                    mae = mean_absolute_error(fmt, [fm]*len(fmt)) # horizontal ineq.

                    df_rank = df_rank.append(pd.DataFrame({'kind': model,
                                                           'metric': metric,
                                                           'N': N,
                                                           'kmin': kmin,
                                                           'fm': fm,
                                                           'hMM': hMM,
                                                           'hmm': hmm,
                                                           'gM': gM,
                                                           'gm': gm,
                                                           'gini': g,
                                                           'mae':mae,
                                                           'epoch': epoch,
                                                           'rank': rank,
                                                           'fmt': fmt}, columns=cols), ignore_index=True)
    save_csv(df_rank, fntr)
    return df_rank


def rank_fit(datasets, output):

    fntr = 'rank_fit.csv'
    fntr = os.path.join(output, fntr)

    if os.path.exists(fntr):
        return read_csv(fntr)

    cols = ['kind', 'metric', 'dataset', 'gini', 'mae', 'epoch', 'rank', 'fmt']
    df_rank = pd.DataFrame(columns=cols)

    for dataset in datasets:

        print()
        printf(dataset)

        path = os.path.join(output, dataset.lower())
        for model in os.listdir(path):
            for fn in os.listdir(os.path.join(path, model)):

                if fn.endswith('.csv'):
                    fn = os.path.join(path, model, fn)
                    metadata = pd.read_csv(fn, index_col=0)
                    epoch = int(fn.split('-ID')[-1].replace('.csv',''))

                    for metric in VALID_METRICS:
                        if metric not in metadata.columns:
                            warnings.warn('metric {} does not exist for {}.'.format(metric, dataset),  UserWarning)
                        else:
                            (rank, fmt), minority_fraction = _rank_function_matrix(metadata, metric)
                            g = gini(metadata[metric].astype(np.float).values)  # vertical ineq.
                            mae = mean_absolute_error(fmt, [minority_fraction] * len(fmt))  # horizontal ineq.

                            df_rank = df_rank.append(pd.DataFrame({'kind': model,
                                                                   'metric': metric,
                                                                   'dataset': dataset,
                                                                   'gini':g,
                                                                   'mae':mae,
                                                                   'epoch': epoch,
                                                                   'rank': rank,
                                                                   'fmt': fmt}, columns=cols), ignore_index=True)
    save_csv(df_rank, fntr)
    return df_rank


def rank_empirical(root, datasets, output):

    ### check final rank if exists
    fntr = 'rank_empirical.csv'
    fntr = os.path.join(output, fntr)

    if os.path.exists(fntr):
        return read_csv(fntr)

    ### if it does not exist, then create file
    cols = ['kind', 'metric', 'dataset', 'gini', 'mae', 'epoch', 'rank', 'fmt']
    df_rank = pd.DataFrame(columns=cols)

    ### for each dataset
    for dataset in datasets:

        print()
        printf(dataset)

        fndr = 'rank_empirical_{}.csv'.format(dataset)
        fndr = os.path.join(root, dataset.lower(), fndr)
        dataset_rank = None

        if os.path.exists(fndr):
            dataset_rank = read_csv(fndr)
        else:

            fn = os.path.join(root, dataset.lower(), 'nodes_metadata.csv')
            printf(os.path.exists(fn))

            # @ todo: remove this once pokec has node_metadata complete
            if not os.path.exists(fn):
                print('no metadata for {}'.format(dataset))
                fn = fn.replace(".csv","_incomplete.csv")
                #warnings.warn('{} does not exist.'.format(fn),  UserWarning)

            printf('loading...')
            metadata = read_csv(fn)
            printf('loaded')

            ### for each metric
            for metric in VALID_METRICS:

                if metric not in metadata.columns or metadata[metric].nunique() <= 1:
                    printf('{} not for {}'.format(metric,dataset))
                    #warnings.warn('metric {} does not exist for {}.'.format(metric, dataset),  UserWarning)
                else:
                    printf('ranking...')
                    (rank, fmt), minority_fraction = _rank_function_matrix(metadata, metric)

                    g = gini(metadata[metric].astype(np.float).values)  # vertical ineq.
                    mae = mean_absolute_error(fmt, [minority_fraction] * len(fmt))  # horizontal ineq.

                    printf('ranked!')
                    tmp = pd.DataFrame({'kind': 'empirical',
                                        'metric': metric,
                                        'dataset': dataset,
                                        'gini':g,
                                        'mae':mae,
                                        'epoch':0,
                                        'rank': rank,
                                        'fmt': fmt}, columns=cols)

                    if dataset_rank is None:
                        dataset_rank = tmp.copy()
                    else:
                        dataset_rank = dataset_rank[cols].append(tmp[cols], ignore_index=True)

            ### save all metrics of a dataset
            dataset_rank.to_csv(fndr)

        ### adding all metrics of a dataset to final
        df_rank = df_rank[cols].append(dataset_rank[cols], ignore_index=True)

    save_csv(df_rank, fntr)
    return df_rank

def best_ranking_empirical(df_distance):
    tmp_min = df_distance.groupby(['dataset']).distance.min().reset_index()
    tmp_min = tmp_min.set_index(['dataset', 'distance'])
    df_best = df_distance.set_index(['dataset', 'distance'])
    df_best = df_best.join(other=tmp_min, on=['dataset', 'distance'], how='right')
    df_best = df_best.reset_index()
    #df_best.drop(columns=['pvalue'], inplace=True)
    return df_best

def worst_ranking_empirical(df_distance):
    tmp_max = df_distance.groupby(['dataset']).distance.max().reset_index()
    tmp_max = tmp_max.set_index(['dataset', 'distance'])
    df_worst = df_distance.set_index(['dataset', 'distance'])
    df_worst = df_worst.join(other=tmp_max, on=['dataset', 'distance'], how='right')
    df_worst = df_worst.reset_index()
    #df_worst.drop(columns=['pvalue'], inplace=True)
    return df_worst

def best_ranking_fit(df_distance):
    tmp_min = df_distance.groupby(['dataset', 'metric']).distance.min().reset_index()
    tmp_min = tmp_min.set_index(['dataset', 'metric', 'distance'])
    df_best = df_distance.set_index(['dataset', 'metric', 'distance'])
    df_best = df_best.join(other=tmp_min, on=['dataset', 'metric', 'distance'], how='right')
    df_best = df_best.pivot_table(index='dataset', columns='metric', values='model', aggfunc='first', fill_value="-")
    return df_best

########################################################################################
# Divergence metrics
########################################################################################

def _divergence_empirical(df_rank, df_summary, fnc_distance):
    cols = ['dataset', 'metric', 'distance', 'pvalue']
    df = pd.DataFrame(columns=cols)

    for dataset in df_rank.dataset.unique():
        fm = df_summary.query("dataset==@dataset").fm.iloc[0]

        for metric in df_rank.metric.unique():
            tmp = df_rank.query("dataset==@dataset & metric==@metric").sort_values("rank").copy()

            try:
                if tmp.shape[0] == 0:
                    d = None
                    pv = None
                else:
                    x1 = tmp.fmt.values
                    x2 = [fm] * tmp.shape[0]
                    d, pv = fnc_distance(x1,x2)

                df = df.append(pd.DataFrame({'dataset': dataset,
                                             'metric': metric,
                                             'distance': d,
                                             'pvalue': pv},
                                            index=[0],
                                            columns=cols), ignore_index=True)
            except Exception as ex:
                pass

    return df

def _MAE_fnc(r1, r2):
    return mean_absolute_error(r1,r2),None

def MAE_empirical(df_rank, df_summary):
    '''
    Computes mean absolute error. 
    0 means perfect fit.
    :param df_rank: 
    :return: df
    '''
    return _divergence_empirical(df_rank, df_summary, _MAE_fnc)

def _divergence_fit(df_rank, fnc_distance):
    cols = ['dataset', 'metric', 'model', 'distance', 'pvalue']
    df = pd.DataFrame(columns=cols)

    for dataset in df_rank.dataset.unique():
        for metric in df_rank.metric.unique():
            if dataset.lower() in ['pokec', 'github'] and metric != 'pagerank':
                continue

            r1 = df_rank.query("dataset == @dataset & metric==@metric & kind=='empirical'").sort_values('rank').fmt

            for model in df_rank.kind.unique():
                if model == 'empirical':
                    continue

                r2 = df_rank.query("dataset == @dataset & metric==@metric & kind==@model")
                r2 = r2.groupby(['kind', 'metric', 'dataset', 'rank']).mean().reset_index().sort_values('rank').fmt

                try:
                    distance, pv = fnc_distance(r1,r2)
                    df = df.append(pd.DataFrame({'dataset': dataset,
                                                 'metric': metric,
                                                 'model': model,
                                                 'distance': distance,
                                                 'pvalue': pv},
                                                index=[0],
                                                columns=cols), ignore_index=True)
                except Exception as ex:
                    pass
    return df

def MAE_fit(df_rank):
    '''
    Computes mean absolute error 
    0 means perfect fit.
    :param df_rank: 
    :return: df
    '''
    return _divergence_fit(df_rank, _MAE_fnc)

@deprecated
def _JS_fnc(r1, r2):
    from scipy.spatial import distance
    d = distance.jensenshannon(r1, r2)
    return d, None
@deprecated
def _KS_fnc(r1, r2):
    return stats.ks_2samp(r1, r2)

@deprecated
def JS_divergence_empirical(df_rank, df_summary):
    '''
    Computes Jensen-Shannon divergence. 
    0 means perfect fit.
    :param df_rank: 
    :return: df
    '''
    return _divergence_empirical(df_rank, df_summary, _JS_fnc)

@deprecated
def KS_divergence_empirical(df_rank, df_summary):
    '''
    Computes the Kolmogorov-Smirnov statistic on 2 samples.
    This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution.
    If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis that the distributions of the two samples are the same.
    In other words:
    If the K-S statistic is large or the p-value is small, then we reject the hypothesis that the distributions of the two samples are the same (thus, they are not the same).

    :param df_rank:
    :return: df
    '''
    return _divergence_empirical(df_rank, df_summary, _KS_fnc)

@deprecated
def JS_divergence_fit(df_rank):
    '''
    Computes Jensen-Shannon divergence. 
    0 means perfect fit.
    :param df_rank: 
    :return: df
    '''
    return _divergence_fit(df_rank, _JS_fnc)

@deprecated
def KS_divergence_fit(df_rank):
    '''
    Computes the Kolmogorov-Smirnov statistic on 2 samples.
    This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution.
    If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis that the distributions of the two samples are the same.
    In other words:
    If the K-S statistic is large or the p-value is small, then we reject the hypothesis that the distributions of the two samples are the same (thus, they are not the same).

    :param df_rank:
    :return: df
    '''
    return _divergence_fit(df_rank, _KS_fnc)





