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

##################################################################################################
# Functions
##################################################################################################


###############################################################
# Distributions
###############################################################
def power_law_distribution(k_min, gamma, n):
    theoretical_distribution = powerlaw.Power_Law(xmin=k_min, parameters=[gamma])
    return theoretical_distribution.generate_random(n)

def random_draw(target_list, activity_distribution):
    prob = activity_distribution[target_list] / activity_distribution[target_list].sum()
    return np.random.choice(a=target_list, size=len(target_list), replace=False, p=prob)

###############################################################
# Network properties
###############################################################
def get_network_summary(G):
    from org.gesis.model.DHBA import estimate_homophily_empirical

    columns = ['dataset','N','cc','class','m','M','fm','E','Emm','EMM','EmM','EMm','density','gammaM','kminM','gammam','kminm','hMM','hmm','triadsratio','triadspdf']
    EMM, EMm, EmM, Emm = utils.get_edge_type_counts(G)
    E = G.number_of_edges()
    fm = utils.get_minority_fraction(G)
    N = G.number_of_nodes()
    m_counts = int(round(fm*N))
    M_counts = int(round((1-fm) * N))
    triads_count = triads.get_triads_from_edges(G,utils.CLASSNAME)
    triads_total = sum(triads_count.values())
    triads_ratio = triads_total / triads.get_total_possible_triads(m_counts, M_counts)
    triads_pdf = [triads_count[key]/triads_total for key in triads.get_triads_ids()]

    gamma_M_out, xmin_M_out, gamma_m_out, xmin_m_out = utils.get_outdegree_powerlaw_exponents(G)
    gamma_M_in, xmin_M_in, gamma_m_in, xmin_m_in = utils.get_indegree_powerlaw_exponents(G)
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
                         'gammam': [gamma_m_out],
                         'kminm': [xmin_m_out],  # outdegree
                         'hMM': [hMM],
                         'hmm': [hmm],
                         'triadsratio': [triads_ratio],
                         'triadspdf': [triads_pdf],
                         },
                        index=[1],
                        columns=columns)

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

    if graph.number_of_nodes() < 6000:
        printf('cot_per_node...')
        cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=10, num_cores=num_cores)

        printf('cot...')
        cot = frequency_by_circle_of_trust(A, cot_per_node=cot_per_node, p=0.85, top=10, num_cores=num_cores)

        printf('wtf...')
        wtf = frequency_by_who_to_follow(A, cot_per_node=cot_per_node, p=0.85, top=10, num_cores=num_cores)

    else:
        cot = None
        wtf = None

    df = pd.DataFrame({'node':nodes,
                       'minority':minoriy,
                       'indegree': ind,
                       'outdegree': outd,
                       'pagerank': pr,
                       'circle_of_trust': cot,
                       'wtf': wtf,
                       }, columns=['node','minority','indegree','outdegree','pagerank','circle_of_trust','wtf'])

    return df

def get_nodes_metadata_big(graph, fn=None, num_cores=10):
    nodes = list(graph.nodes())
    printf('nodes')

    A = nx.adjacency_matrix(graph, nodes).astype(np.int8) #csr
    printf('adj')

    ind = A.sum(axis=0).flatten().tolist()[0]
    printf('ind')

    outd = A.sum(axis=1).flatten().tolist()[0]
    printf('outd')

    pr = pagerank_power(A, p=0.85).tolist()
    printf('pr')

    minoriy = [graph.node[n][graph.graph['label']] for n in nodes]
    printf('minority')

    printf('cot_per_node...')
    cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=10, num_cores=num_cores)

    printf('cot...')
    cot = frequency_by_circle_of_trust(A, cot_per_node=cot_per_node, p=0.85, top=10, num_cores=num_cores)

    printf('wtf...')
    wtf = frequency_by_who_to_follow(A, cot_per_node=cot_per_node, p=0.85, top=10, num_cores=num_cores)

    df = pd.DataFrame({'node':nodes,
                       'minority':minoriy,
                       'indegree': ind,
                       'outdegree': outd,
                       'pagerank': pr,
                       'circle_of_trust': cot,
                       'wtf': wtf,
                       }, columns=['node','minority','indegree','outdegree','pagerank','circle_of_trust','wtf'])

    if fn is not None:
        save_csv(df, fn)

def load_all_node_metadata_empirical(datasets, root):
    df_metadata = None

    for dataset in datasets:
        printf('=== {} ==='.format(dataset))

        ### loading dataset metadata
        fn = os.path.join(root, dataset, 'nodes_metadata.csv')
        if os.path.exists(fn):
            df = read_csv(fn)
            printf('loaded!')
        else:
            fn = os.path.join(root, dataset, 'nodes_metadata_incomplete.csv')
            if os.path.exists(fn):
                df = read_csv(fn)
                printf('loaded!')

        ### df_metadata from all datasets (append)
        if df_metadata is None:
            df_metadata = df.copy()
        else:
            df_metadata = df_metadata.append(df, ignore_index=True)

        del (df)

    return df_metadata

def load_all_node_metadata_fit(datasets, models, output):
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

    return df_metadata


    return

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