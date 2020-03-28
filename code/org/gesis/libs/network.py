##################################################################################################
# System's Dependencies
##################################################################################################
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import powerlaw
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel
from joblib import delayed
from collections import Counter
from fast_pagerank import pagerank_power

##################################################################################################
# Local Dependencies
##################################################################################################

from org.gesis.libs import utils
from org.gesis.libs import triads
from org.gesis.libs.utils import printf
from org.gesis.libs.io import save_csv
from org.gesis.libs.io import read_csv
from org.gesis.libs.io import read_pickle
from org.gesis.libs.io import save_pickle

##################################################################################################
# Constants
##################################################################################################

BIGNET = 11000

##################################################################################################
# Functions
##################################################################################################


###############################################################
# Distributions
###############################################################
def power_law_distribution(n, gamma, k_min=None, k_max=None):
    theoretical_distribution = powerlaw.Power_Law(xmin=k_min, xmax=k_max, parameters=[gamma])
    return theoretical_distribution.generate_random(n)

def random_draw(target_list, activity_distribution):
    prob = activity_distribution[target_list] / activity_distribution[target_list].sum()
    return np.random.choice(a=target_list, size=len(target_list), replace=False, p=prob)

###############################################################
# Network Summary
###############################################################

def get_all_datasets_summary(datasets, root, output, verbose=True):

    fn = os.path.join(output, 'summary_datasets.csv')
    if os.path.exists(fn):
        return read_csv(fn)

    df_summary = None
    for dataset in datasets:
        if verbose:  printf('{}...'.format(dataset))
        fng = os.path.join(root, dataset, '{}_attributed_network_anon.gpickle'.format(dataset))
        G = nx.read_gpickle(fng)
        row = get_network_summary(G)
        del (G)

        if df_summary is None:
            df_summary = row.copy()
        else:
            df_summary = df_summary.append(row, ignore_index=True)

    save_csv(df_summary, fn)
    return df_summary

def all_datasets_summary_as_latex(df_summary, output=None):
    df_latex_summary = df_summary.pivot_table(columns='dataset', aggfunc=lambda x: ' '.join(str(v) for v in x))
    columns = ['N', 'cc', 'class', 'M', 'm', 'fm', 'E', 'EMM', 'EMm', 'EmM', 'Emm', 'density']
    df_latex_summary = df_latex_summary.reindex(columns)

    if output is not None:
        fn = os.path.join(output, 'summary_datasets.tex')
        df_latex_summary.to_latex(fn, float_format=lambda x: '%.5f' % x)
        print('{} saved!'.format(fn))
        
    return df_latex_summary

def get_network_summary(G):
    from org.gesis.model.DHBA import estimate_homophily_empirical

    columns = ['dataset','N','cc','class','m','M','fm','E','Emm','EMM','EmM','EMm','density','gammaM','kminM','kmaxM','gammam','kminm','kmaxm','hMM','hmm','triadsratio','triadspdf']
    EMM, EMm, EmM, Emm = utils.get_edge_type_counts(G)
    E = G.number_of_edges()
    fm = utils.get_minority_fraction(G)
    N = G.number_of_nodes()
    m_counts = int(round(fm*N))
    M_counts = int(round((1-fm) * N))
    triads_count = triads.get_triads_from_edges(G,utils.CLASSNAME)
    triads_total = sum(triads_count.values())
    triads_ratio = triads_total / triads.get_total_possible_triads(m_counts, M_counts)
    if triads_total != 0:
        triads_pdf = [triads_count[key]/triads_total for key in triads.get_triads_ids()]
    else:
        triads_pdf = [1 / len(triads.get_triads_ids()) for key in triads.get_triads_ids()]

    fitM, fitm = utils.get_outdegree_powerlaw_exponents(G)
    gamma_M_out, xmin_M_out, xmax_M_out = fitM.power_law.alpha, fitM.power_law.xmin, fitM.power_law.xmax
    gamma_m_out, xmin_m_out, xmax_m_out = fitm.power_law.alpha, fitm.power_law.xmin, fitm.power_law.xmax

    fitM, fitm = utils.get_indegree_powerlaw_exponents(G)
    gamma_M_in, xmin_M_in, xmax_M_in = fitM.power_law.alpha, fitM.power_law.xmin, fitM.power_law.xmax
    gamma_m_in, xmin_m_in, xmax_m_in = fitm.power_law.alpha, fitm.power_law.xmin, fitm.power_law.xmax

    hMM, hmm = estimate_homophily_empirical(G, gammaM_in=gamma_M_in, gammam_in=gamma_m_in)

    return pd.DataFrame({'dataset':[utils.get_graph_metadata(G,'name')],
                         'N':[N],
                         'E':[E],
                         'cc':[nx.number_connected_components(nx.to_undirected(G))],
                         'density':[nx.density(G)],
                         'class':[utils.get_graph_metadata(G,'class')],
                         'm':[utils.get_graph_metadata(G,'labels')[1]],
                         'M':[utils.get_graph_metadata(G,'labels')[0]],
                         'fm':[fm],
                         'Emm': [Emm/E],
                         'EMM': [EMM/E],
                         'EmM': [EmM/E],
                         'EMm': [EMm/E],
                         'gammaM': [gamma_M_out],
                         'kminM': [xmin_M_out],  #outdegree
                         'kmaxM': [xmax_M_out],  # outdegree
                         'gammam': [gamma_m_out],
                         'kminm': [xmin_m_out],  # outdegree
                         'kmaxm': [xmax_m_out],  # outdegree
                         'hMM': [hMM],
                         'hmm': [hmm],
                         'triadsratio': [triads_ratio],
                         'triadspdf': [triads_pdf],
                         },
                        index=[1],
                        columns=columns)

###############################################################
# Nodes metadata
###############################################################

def _ppr(node_index, A, p, top):
    pp = np.zeros(A.shape[0])
    pp[node_index] = A.shape[0]
    pr = pagerank_power(A, p=p, personalize=pp)
    pr = pr.argsort()[-top-1:][::-1]
    #time.sleep(0.01)
    return pr[pr!=node_index][:top]

def get_circle_of_trust_per_node(A, p=0.85, top=10, num_cores=40):
    return Parallel(n_jobs=num_cores)(delayed(_ppr)(node_index, A, p, top) for node_index in np.arange(A.shape[0]))

def frequency_by_circle_of_trust(A, cot_per_node=None, p=0.85, top=10, num_cores=40):
    results = cot_per_node if cot_per_node is not None else get_circle_of_trust_per_node(A, p, top, num_cores)
    unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
    del(results)
    return [ 0 if node_index not in unique_elements else counts_elements[np.argwhere(unique_elements == node_index)[0, 0]] for node_index in np.arange(A.shape[0])]

def _salsa(node_index, cot, A, top=10):
    BG = nx.Graph()
    BG.add_nodes_from(['h{}'.format(vi) for vi in cot], bipartite=0)  # hubs
    edges = [('h{}'.format(vi), int(vj)) for vi in cot for vj in np.argwhere(A[vi,:] != 0 )[:,1]]
    BG.add_nodes_from(set([e[1] for e in edges]), bipartite=1)  # authorities
    BG.add_edges_from(edges)
    centrality = Counter({n: c for n, c in nx.eigenvector_centrality_numpy(BG).items() if type(n) == int
                                                                                       and n not in cot
                                                                                       and n != node_index
                                                                                       and n not in np.argwhere(A[node_index,:] != 0 )[:,1] })
    del(BG)
    #time.sleep(0.01)
    return np.asarray([n for n, pev in centrality.most_common(top)])[:top]

def frequency_by_who_to_follow(A, cot_per_node=None, p=0.85, top=10, num_cores=40):
    cot_per_node = cot_per_node if cot_per_node is not None else get_circle_of_trust_per_node(A, p, top, num_cores)
    results = Parallel(n_jobs=num_cores)(delayed(_salsa)(node_index, cot, A, top) for node_index, cot in enumerate(cot_per_node))
    unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
    del(results)
    return [0 if node_index not in unique_elements else counts_elements[np.argwhere(unique_elements == node_index)[0, 0]] for node_index in np.arange(A.shape[0])]

def get_nodes_metadata(graph, num_cores=10):
    nodes = list(graph.nodes())
    A = nx.adjacency_matrix(graph, nodes).astype(np.int8) #csr
    ind = A.sum(axis=0).flatten().tolist()[0]
    outd = A.sum(axis=1).flatten().tolist()[0]
    pr = pagerank_power(A, p=0.85).tolist()
    minoriy = [graph.node[n][graph.graph['label']] for n in nodes]

    if graph.number_of_nodes() < BIGNET:
        printf('cot_per_node...')
        cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=10, num_cores=num_cores)

        printf('cot...')
        cot = frequency_by_circle_of_trust(A, cot_per_node=cot_per_node, p=0.85, top=10, num_cores=num_cores)

        printf('wtf...')
        wtf = frequency_by_who_to_follow(A, cot_per_node=cot_per_node, p=0.85, top=10, num_cores=num_cores)

    else:
        cot = None
        wtf = None

    df = pd.DataFrame({'dataset':graph.graph['name'],
                       'node':nodes,
                       'minority':minoriy,
                       'indegree': ind,
                       'outdegree': outd,
                       'pagerank': pr,
                       'circle_of_trust': cot,
                       'wtf': wtf,
                       }, columns=['dataset','node','minority','indegree','outdegree','pagerank','circle_of_trust','wtf'])

    return df

def get_nodes_metadata_big(graph, path, original=False, num_cores=10):
    from sklearn.model_selection import StratifiedShuffleSplit

    fn_original = os.path.join(path, 'nodes_metadata.csv')
    fn_merge = os.path.join(path, 'nodes_metadata_merge.csv')

    ### stratified split (allowing for same fm in sample)
    Nold = graph.number_of_nodes()
    Nnew = 100000 # number of nodes in each sample
    n_splits = int(np.ceil(Nold / float(Nnew)))

    ### if already exist, return
    total = len([fn for fn in os.listdir(path) if fn.endswith('.csv') and fn.startswith('nodes_metadata_') and not fn.endswith('merge.csv')])
    if os.path.exists(fn_merge):
        df_merge = read_csv(fn_merge)

    elif total == n_splits and not os.path.exists(fn_merge):
        printf('Merging all splits into one file...')
        df_merge = _merge_metadata_big(path, graph)
    # elif os.path.exists(fn_original):
    #     printf('Nothing to do; all your results already exist. Bye!')
    #     return
    elif total < n_splits:
        ### All nodes and class value
        X = np.array(list(graph.nodes())) # nodes
        y = np.array([graph.node[n][graph.graph['label']] for n in X]) # classes

        ### Inferring the metadata for each split
        split = 1
        for N, nsplits in [(Nnew,n_splits-1), (Nold - ((n_splits-1) * Nnew) , 1)]:
            sss = StratifiedShuffleSplit(n_splits=nsplits, test_size=N, random_state=0)
            for train_index, test_index in sss.split(X, y):
                printf('')
                print('=============================')
                print('Split #{} of {}'.format(split, n_splits))
                print('- {} nodes ({}% of {})'.format(N, round(N*100. / float(Nold), 2), Nold))
                nodes = X[test_index]
                classes = y[test_index]
                _do_metadata_big(graph, nodes, classes, path, split, num_cores=num_cores)
                split += 1

        del(sss)
        del(X)
        del(y)

        ### merge into 1 file (only for cot and wtf)
        df_merge = _merge_metadata_big(path, graph)

    print('samples')
    print(df_merge.shape)
    print(df_merge.groupby('minority').size())

    ### for indegree, outdegree and pagerank: main file
    if original:

        if os.path.exists(fn_original):
            print('loading original...')
            df_original = read_csv(fn_original)
        else:
            print('computing original...')
            df_original = get_nodes_metadata(graph, num_cores=num_cores)
            df_original.loc[:,'dataset'] = graph.graph['name'].lower()

        print('original')
        print(df_original.shape)
        print(df_original.groupby('minority').size())

        if df_merge.shape[0] != df_original.shape[0]:
            print('different shapes')
            return
        else:
            if df_merge.groupby('minority').size()[0] > df_original.groupby('minority').size()[0]:
                nchanges = df_merge.groupby('minority').size()[0] - df_original.groupby('minority').size()[0]
                tmp = df_merge.query('minority==1').copy()
                idmin = tmp.indegree.min()
                idmax = tmp.indegree.max()
                odmin = tmp.indegree.min()
                odmax = tmp.indegree.max()
                counter = 0
                for i, row in df_merge.query("minority == 0 & indegree >= @idmin & indegree <= @idmax & outdegree >= @odmin & outdegree <= @odmax").iterrows():
                    df_merge.loc[i,'minority'] = 1
                    counter += 1
                    if counter == nchanges:
                        break
                print('samples')
                print(df_merge.shape)
                print(df_merge.groupby('minority').size())

                if df_merge.groupby('minority').size()[0] == df_original.groupby('minority').size()[0]:
                    save_csv(df_merge, fn_merge)
                    print('perfect!')

        print('updating...')
        df_merge.sort_values('minority', ascending=True, inplace=True)
        df_original.sort_values('minority', ascending=True, inplace=True)
        df_original.loc[:, 'wtf'] = df_merge.loc[:, 'wtf']
        df_original.loc[:, 'circle_of_trust'] = df_merge.loc[:, 'circle_of_trust']

        save_csv(df_original, fn_original)



def _merge_metadata_big(path, graph):
    files = [os.path.join(path,fn) for fn in os.listdir(path) if fn.endswith('.csv') and fn.startswith('nodes_metadata_') and not fn.endswith('merge.csv')]
    print('Merging {} files...'.format(len(files)))

    df = None
    for fn in files:
        tmp = read_csv(fn)

        if 'dataset' not in tmp.columns:
            tmp.loc[:,'dataset'] = graph.graph['name'].lower()
            tmp = tmp[['dataset','node','minority','indegree','outdegree','pagerank','circle_of_trust','wtf']]
            save_csv(tmp, fn)

        if df is None:
            df = tmp.copy()
        else:
            df = df.append(tmp, ignore_index=True)

    fn = os.path.join(path, 'nodes_metadata_merge.csv')
    save_csv(df, fn)
    return df

def _do_metadata_big(graph, nodes, classes, path, split, num_cores=1):

    ### 0. already exists?
    fn = os.path.join(path, 'nodes_metadata_{}.csv'.format(split))
    if os.path.exists(fn):
        print("- split #{} already exist.".format(split))
        return

    ### 1.
    printf('nodes')
    fncpn = fn.replace(".csv", "_nodes.pickle")
    if os.path.exists(fncpn):
        print('loading...')
        nodes = read_pickle(fncpn)
    else:
        print('saving...')
        save_pickle(nodes, fncpn)
    print('- class distribution: {}'.format(Counter(classes)))

    ### 2.
    A = nx.adjacency_matrix(graph, nodes).astype(np.int8) #csr
    printf('adj')

    ## 3.
    printf('cot_per_node')

    fncpn = fn.replace(".csv","_cpn.pickle")
    if os.path.exists(fncpn):
        print('loading...')
        cot_per_node = read_pickle(fncpn)
    else:
        print('computing...')
        cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=10, num_cores=num_cores)
        print('saving...')
        save_pickle(cot_per_node, fncpn)

    ## 4.
    printf('cot...')

    fncpn = fn.replace(".csv", "_cot.pickle")
    if os.path.exists(fncpn):
        print('loading...')
        cot = read_pickle(fncpn)
    else:
        print('computing...')
        cot = frequency_by_circle_of_trust(A, cot_per_node=cot_per_node, p=0.85, top=10, num_cores=num_cores)
        print('saving...')
        save_pickle(cot, fncpn)

    ## 5.
    printf('wtf...')

    fncpn = fn.replace(".csv", "_wtf.pickle")
    if os.path.exists(fncpn):
        print('loading...')
        wtf = read_pickle(fncpn)
    else:
        print('computing...')
        wtf = frequency_by_who_to_follow(A, cot_per_node=cot_per_node, p=0.85, top=10, num_cores=num_cores)
        print('saving...')
        save_pickle(wtf, fncpn)

    ### 6.
    ind = A.sum(axis=0).flatten().tolist()[0]
    printf('ind')

    outd = A.sum(axis=1).flatten().tolist()[0]
    printf('outd')

    pr = pagerank_power(A, p=0.85).tolist()
    printf('pr')

    minoriy = [graph.node[n][graph.graph['label']] for n in nodes]
    printf('minority')

    df = pd.DataFrame({'dataset': graph.graph['name'],
                       'node':nodes,
                       'minority':minoriy,
                       'indegree': ind,
                       'outdegree': outd,
                       'pagerank': pr,
                       'circle_of_trust': cot,
                       'wtf': wtf,
                       }, columns=['dataset','node','minority','indegree','outdegree','pagerank','circle_of_trust','wtf'])

    if fn is not None:
        print('final saving...')
        save_csv(df, fn)

def load_all_datasets_node_metadata_fit(datasets, output):
    fna = os.path.join(output, 'all_datasets_metadata_fit.csv')
    if os.path.exists(fna):
        return read_csv(fna)

    df_metadata = None

    for dataset in datasets:
        printf('=== {} ==='.format(dataset))
        df = None

        ### loading dataset metadata
        fn = os.path.join(output, dataset, 'nodes_metadata.csv')
        if os.path.exists(fn):
            df = read_csv(fn)
            printf('loaded!')
        else:
            fn = os.path.join(output, dataset, 'nodes_metadata_incomplete.csv')
            if os.path.exists(fn):
                df = read_csv(fn)
                printf('loaded!')
            else:
                printf("You should compute nodes_metadata.csv for {} using batch_node_attributes.py !!!".format(dataset))

        ### df_metadata from all datasets (append)
        if df is not None:
            if df_metadata is None:
                df_metadata = df.copy()
            else:
                df_metadata = df_metadata.append(df, ignore_index=True)
            del (df)

    save_csv(df_metadata, fna)
    return df_metadata

def load_all_datasets_node_metadata_empirical(datasets, root):
    fna = os.path.join(root, 'all_datasets_metadata_empirical.csv')
    if os.path.exists(fna):
        return read_csv(fna)

    df_metadata = None

    for dataset in datasets:
        printf('=== {} ==='.format(dataset))
        df = None

        ### loading dataset metadata
        fn = os.path.join(root, dataset, 'nodes_metadata.csv')
        if os.path.exists(fn):
            df = read_csv(fn)
            printf('loaded!')
        else:
            fn = os.path.join(root, dataset, 'nodes_metadata_incomplete.csv')
            if os.path.exists(fn):
                df = read_csv(fn)
                printf('i-loaded!')
            else:
                printf("You should compute nodes_metadata.csv for {} using batch_node_attributes.py !!!".format(dataset))

        ### df_metadata from all datasets (append)
        if df is not None:
            if df_metadata is None:
                df_metadata = df.copy()
            else:
                df_metadata = df_metadata.append(df, ignore_index=True)
            del (df)

    save_csv(df_metadata, fna)
    return df_metadata

def load_all_datasets_node_metadata_fit(datasets, models, output):
    fna = os.path.join(output, 'all_datasets_metadata_fit.csv')
    if os.path.exists(fna):
        return read_csv(fna)

    df_metadata = None

    for dataset in datasets:
        for model in models:
            path = os.path.join(output, dataset, model)
            files = [fn for fn in os.listdir(path) if fn.endswith('.csv')]
            for fn in files:
                id = int(fn.split('-ID')[-1].replace(".csv",''))
                fn = os.path.join(path, fn)
                df = read_csv(fn)
                df.loc[:, 'dataset'] = dataset
                df.loc[:, 'model'] = model
                df.loc[:, 'epoch'] = id

                ### df_metadata from all datasets (append)
                if df_metadata is None:
                    df_metadata = df.copy()
                else:
                    df_metadata = df_metadata.append(df, ignore_index=True)

                del (df)

    save_csv(df_metadata, fna)
    return df_metadata

def load_all_node_metadata_synthetic(model, output):
    fna = os.path.join(output, 'synthetic', '{}_all_metadata.csv'.format(model))
    if os.path.exists(fna):
        return read_csv(fna)

    path = os.path.join(output, 'synthetic', model)
    files = [os.path.join(path,fn) for fn in os.listdir(path) if fn.endswith('.csv')]

    df_metadata = None
    for fn in files:
        # DHBA-N2000-kmin2-fm0.5-hMM1.0-hmm1.0-ID1.csv
        # ,node,minority,indegree,outdegree,pagerank,circle_of_trust,wtf
        N = int(fn.split('/')[-1].split('-N')[-1].split('-kmin')[0])
        kmin = int(fn.split('/')[-1].split('-kmin')[-1].split('-fm')[0])
        fm = float(fn.split('/')[-1].split('-fm')[-1].split('-hMM')[0])
        hMM = float(fn.split('/')[-1].split('-hMM')[-1].split('-hmm')[0])
        hmm = float(fn.split('/')[-1].split('-hmm')[-1].split('-ID')[0]) #-gM
        # gM =
        # gm =
        epoch = int(fn.split('/')[-1].split('-ID')[-1].split('.csv')[0])

        df = read_csv(fn)
        df.loc[:, 'N'] = N
        df.loc[:, 'kmin'] = kmin
        df.loc[:, 'fm'] = fm
        df.loc[:, 'hMM'] = hMM
        df.loc[:, 'hmm'] = hmm
        df.loc[:, 'gm'] = 2.5
        df.loc[:, 'gM'] = 2.5
        df.loc[:,'epoch'] = epoch

        if df_metadata is None:
            df_metadata = df.copy()
        else:
            df_metadata = df_metadata.append(df, ignore_index=True)

    save_csv(df_metadata, fna)
    return df_metadata

def all_datasets_node_metadata_degrees_pivot(df_metadata, model=None):
    df_metadata_pivot = None

    for dataset in df_metadata.dataset.unique():
        tmp = df_metadata.query("dataset==@dataset").copy().sort_values('node')

        if model is not None and 'model' in tmp.columns:
            tmp = tmp.query("model==@model").copy()

        for metric in ['indegree', 'outdegree']:
            tmpnew = pd.DataFrame({'dataset': dataset,
                                   'node': tmp.node,
                                   'minority': tmp.minority,
                                   'metric': metric,
                                   'value': tmp[metric],
                                   'epoch':0 if model is None else tmp.epoch,
                                   })
            if df_metadata_pivot is None:
                df_metadata_pivot = tmpnew.copy()
            else:
                df_metadata_pivot = df_metadata_pivot.append(tmpnew, ignore_index=True)

        del (tmpnew)
        del (tmp)

    return df_metadata_pivot

# def mean_lorenz_curves_and_gini_fit(df, metrics):
#     from org.gesis.libs.utils import gini
#     from org.gesis.libs.utils import lorenz_curve
#
#     df_lorenz_curve = pd.DataFrame(columns=['x','y','model','dataset'])
#     df_gini_coef = None
#
#     for name,group in df.groupby(['dataset','model','epoch']):
#
#         for metric in metrics:
#             X = np.sort(group[metric].astype(np.float).values)
#             gc = gini(X)
#
#             y = lorenz_curve(X)
#             x = np.arange(y.size) / (y.size - 1)
    # return None, None