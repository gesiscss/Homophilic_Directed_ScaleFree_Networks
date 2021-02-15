################################################################################
# System dependencies
################################################################################
import os
import powerlaw
import numpy as np
import pandas as pd
import networkx as nx
from joblib import delayed
from joblib import Parallel
from collections import Counter
from fast_pagerank import pagerank_power

################################################################################
# Local dependencies
################################################################################
from org.gesis.lib import io
from org.gesis.lib import utils

################################################################################
# Constants
################################################################################
BIGNET = 11000
EXT = '.gpickle'
TOPK = 10

################################################################################
# I/O
################################################################################

def get_graph(path, dataset, fn=None):
    if fn is None:
        fn = io.get_files(os.path.join(path,dataset), prefix="{}".format(dataset), ext=EXT)
        if len(fn) == 0:
            raise Exception("Graph not found.")
        fn = os.path.join(path,dataset,fn[0])
    print(fn)
    return io.load_gpickle(fn)

def get_outdegrees_fit(path):
    files = io.get_files(path, ext=EXT)
    degrees = []
    for fn in files:
        od = get_outdegree(io.load_gpickle(os.path.join(path,fn)))
        degrees.extend(od)
        del(od)
    return degrees
    
def get_indegrees_fit(path):
    files = io.get_files(path, ext=EXT)
    degrees = []
    for fn in files:
        od = get_outdegree(io.load_gpickle(os.path.join(path,fn)))
        degrees.extend(od)
        del(od)
    return degrees

################################################################################
# Properties
################################################################################

def get_graph_metadata(graph, attribute):
    if attribute in graph.graph:
        return graph.graph[attribute]
    return None

def get_edge_type_counts(graph, fraction=False):
    counts = Counter(['{}{}'.format(graph.graph['groups'][graph.node[edge[0]][graph.graph['label']]],
                                    graph.graph['groups'][graph.node[edge[1]][graph.graph['label']]]) for edge in graph.edges()])
    if not fraction:
        return counts['MM'], counts['Mm'], counts['mM'], counts['mm']

    total = counts['MM'] + counts['Mm'] + counts['mM'] + counts['mm'] 
    total = total if total>0 else 1
    return counts['MM']/total, counts['Mm']/total, counts['mM']/total, counts['mm']/total

def get_minority_fraction(graph):
    b = Counter([graph.node[n][graph.graph['label']] for n in graph.nodes()]).most_common()[1][1] / graph.number_of_nodes()
    return b

def get_density(graph):
    return nx.density(graph)

def get_min_degree(graph):
    return min([d for n, d in graph.degree()])

def get_outdegree(g):
    return [d for n,d in g.out_degree()]

def get_indegree(g):
    return [d for n,d in g.in_degree()]
        
def get_node_metadata_as_dataframe(g, njobs=1):
    cols = ['node','minority','indegree','outdegree','pagerank','wtf']
    df = pd.DataFrame(columns=cols)
    nodes = g.nodes()
    minority = [g.node[n][g.graph['label']] for n in nodes]
    indegree = [g.in_degree(n) for n in nodes]
    outdegree = [g.out_degree(n) for n in nodes]
    A = nx.to_scipy_sparse_matrix(g,nodes)
    pagerank = pagerank_power(A, p=0.85, tol=1e-6)
    wtf = who_to_follow_rank(A, njobs)
    
    return pd.DataFrame({'node':nodes,
                        'minority':minority,
                        'indegree':indegree,
                        'outdegree':outdegree,
                        'pagerank':pagerank,
                        'wtf':wtf,
                        }, columns=cols)

################################################################################
# Power-law
################################################################################

def fit_power_law(data, discrete=True):
    return powerlaw.Fit(data,
                        discrete=discrete,
                        verbose=False)

def fit_power_law_force(data, discrete=True, xmin=None, xmax=None):
    return powerlaw.Fit(data,
                        discrete=discrete,
                        xmin=xmin if xmin is not None else min(data),
                        xmax=xmax if xmax is not None else max(data),
                        verbose=False)

def fit_theoretical_power_law(nobs, exp, xmin=None, xmax=None, discrete=True):
    if discrete:
        xmin = int(round(xmin)) if xmin is not None else xmin
        xmax = int(round(xmax)) if xmax is not None else xmax
    nobs = int(round(nobs))

    theoretical_distribution = powerlaw.Power_Law(xmin=xmin, xmax=xmax, discrete=discrete, parameters=[exp])
    simulated_data = theoretical_distribution.generate_random(nobs)
    return powerlaw.Fit(simulated_data, verbose=False)

def get_outdegree_powerlaw_exponents(graph):
    x = np.array([d for n, d in graph.out_degree() if graph.node[n][graph.graph['label']] == 0])
    fitM = fit_power_law(x)

    x = np.array([d for n, d in graph.out_degree() if graph.node[n][graph.graph['label']] == 1])
    fitm = fit_power_law(x)

    return fitM, fitm

def get_indegree_powerlaw_exponents(graph):
    x = np.array([d for n, d in graph.in_degree() if graph.node[n][graph.graph['label']] == 0])
    fitM = fit_power_law(x)

    x = np.array([d for n, d in graph.in_degree() if graph.node[n][graph.graph['label']] == 1])
    fitm = fit_power_law(x)

    return fitM, fitm


################################################################################
# Who-To-Follow
################################################################################

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

def who_to_follow_rank(A, njobs=1):
    if A.shape[0] < BIGNET:
        return wtf_small(A, njobs)
    else:
        # TODO: implement optimal (or faster) solution for big net
        return wtf_small(A, njobs)
        
def wtf_small(A, njobs):
    utils.printf('cot_per_node...')
    cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=TOPK, num_cores=njobs)

    utils.printf('cot...')
    cot = frequency_by_circle_of_trust(A, cot_per_node=cot_per_node, p=0.85, top=TOPK, num_cores=njobs)

    utils.printf('wtf...')
    wtf = frequency_by_who_to_follow(A, cot_per_node=cot_per_node, p=0.85, top=TOPK, num_cores=njobs)
    return wtf
