import numpy as np
import pandas as pd
import os
from collections import defaultdict
import warnings
from scipy import stats
import networkx as nx
import operator

PERCENTAGE_RANGE = np.append([5], np.arange(10, 100 + 10, 10)).astype(np.float)
VALID_METRICS = ['pagerank', 'circle_of_trust', 'wtf']


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

def rank_fit(datasets, output):
    from org.gesis.libs.io import read_csv
    from org.gesis.libs.utils import printf

    fntr = 'rank_fit.csv'
    fntr = os.path.join(output, fntr)

    if os.path.exists(fntr):
        return read_csv(fntr)

    cols = ['kind', 'metric', 'dataset', 'rank', 'fmt']
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
                            df_rank = df_rank.append(pd.DataFrame({'kind': model,
                                                                   'metric': metric,
                                                                   'dataset': dataset,
                                                                   'epoch': epoch,
                                                                   'rank': rank,
                                                                   'fmt': fmt}, columns=cols), ignore_index=True)
    df_rank.to_csv(fntr)
    return df_rank


def rank_empirical(root, datasets, output):
    from org.gesis.libs.utils import printf
    from org.gesis.libs.io import read_csv

    ### check final rank if exists
    fntr = 'rank_empirical.csv'
    fntr = os.path.join(output, fntr)

    if os.path.exists(fntr):
        return read_csv(fntr)

    ### if it does not exist, then create file
    cols = ['kind', 'metric', 'dataset', 'rank', 'fmt']
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

            if not os.path.exists(fn):
                print('no metadata for {}'.format(dataset))
                #warnings.warn('{} does not exist.'.format(fn),  UserWarning)

            printf('loading...')
            metadata = pd.read_csv(fn, index_col=0)
            printf('loaded')

            ### for each metric
            for metric in VALID_METRICS:

                if metric not in metadata.columns or metadata[metric].nunique() <= 1:
                    printf('{} not for {}'.format(metric,dataset))
                    #warnings.warn('metric {} does not exist for {}.'.format(metric, dataset),  UserWarning)
                else:
                    printf('ranking...')
                    (rank, fmt), minority_fraction = _rank_function_matrix(metadata, metric)
                    printf('ranked!')
                    tmp = pd.DataFrame({'kind': 'empirical',
                                        'metric': metric,
                                        'dataset': dataset,
                                        'rank': rank,
                                        'fmt': fmt}, columns=cols)

                    if dataset_rank is None:
                        dataset_rank = tmp.copy()
                    else:
                        dataset_rank = dataset_rank[cols].append(tmp[cols], ignore_index=True)

            ### save all metrics of a dataset
            dataset_rank.to_csv(fndr)

        ### adding all metrics of a dataset to final
        df_rank = df_rank[cols].append(dataset_rank[cols])

    df_rank.to_csv(fntr)
    return df_rank


def KS_divergence(df_rank):
    '''
    Computes the Kolmogorov-Smirnov statistic on 2 samples.
    This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution.
    If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis that the distributions of the two samples are the same.
    In other words:
    If the K-S statistic is large or the p-value is small, then we reject the hypothesis that the distributions of the two samples are the same (thus, they are not the same).

    :param df_rank:
    :return:
    '''
    cols = ['dataset','metric','model','ks','pvalue']
    df = pd.DataFrame(columns=cols)

    for dataset in df_rank.dataset.unique():
        for metric in df_rank.metric.unique():

            if dataset.lower() in ['pokec','github'] and metric != 'pagerank':
                continue

            r1 = df_rank.query("dataset == @dataset & metric==@metric & kind=='empirical'").sort_values('rank').fmt

            for model in df_rank.kind.unique():

                if model == 'empirical':
                    continue

                r2 = df_rank.query("dataset == @dataset & metric==@metric & kind==@model")
                r2 = r2.groupby(['kind','metric','dataset','rank']).mean().reset_index().sort_values('rank').fmt

                try:
                    ks,pv = stats.ks_2samp(r1, r2)
                    df = df.append(pd.DataFrame({'dataset':dataset,
                                                 'metric':metric,
                                                 'model':model,
                                                 'ks':ks,
                                                 'pvalue':pv},
                                                index=[0],
                                                columns=cols), ignore_index=True)
                except Exception as ex:
                    pass
    return df
