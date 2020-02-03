import powerlaw
import numpy as np
import pandas as pd
import networkx as nx
from org.gesis.libs import utils
from org.gesis.libs import triads
from fast_pagerank import pagerank_power

def power_law_distribution(k_min, gamma, n):
    theoretical_distribution = powerlaw.Power_Law(xmin=k_min, parameters=[gamma])
    return theoretical_distribution.generate_random(n)

def random_draw(target_list, activity_distribution):
    prob = activity_distribution[target_list] / activity_distribution[target_list].sum()
    return np.random.choice(a=target_list, size=len(target_list), replace=False, p=prob)

def get_network_summary(G):
    from org.gesis.model.DHBA import estimate_homophily_empirical

    columns = ['dataset','N','cc','class','m','M','fm','kmin','E','Emm','EMM','EmM','EMm','density','gammaM','gammam','hMM','hmm','triadsratio','triadspdf']
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
    gamma_M, sigma_M, gamma_m, sigma_m = utils.get_outdegree_powerlaw_exponents(G)

    beta_M, sig_M, beta_m, sig_m = utils.get_indegree_powerlaw_exponents(G)
    hMM, hmm = estimate_homophily_empirical(G, betaM=beta_M, betam=beta_m)

    return pd.DataFrame({'dataset':[utils.get_graph_metadata(G,'name')],
                         'N':[N],
                         'E':[E],
                         'cc':[nx.number_connected_components(nx.to_undirected(G))],
                         'kmin':[utils.get_min_degree(G)],
                         'density':[nx.density(G)],
                         'class':[utils.get_graph_metadata(G,'class')],
                         'm':[utils.get_graph_metadata(G,'labels')[1]],
                         'M':[utils.get_graph_metadata(G,'labels')[0]],
                         'fm':[fm],
                         'Emm': [Emm/E],
                         'EMM': [EMM/E],
                         'EmM': [EmM/E],
                         'EMm': [EMm/E],
                         'gammaM': gamma_M,
                         'gammam': gamma_m,
                         'hMM': hMM,
                         'hmm': hmm,
                         'triadsratio': [triads_ratio],
                         'triadspdf': [triads_pdf],
                         },
                        index=[1],
                        columns=columns)

def get_nodes_metadata(graph):
    nodes = list(graph.nodes())
    A = nx.adjacency_matrix(graph, nodes).astype(np.int8) #csr
    ind = A.sum(axis=0).flatten().tolist()[0]
    outd = A.sum(axis=1).flatten().tolist()[0]
    pr = pagerank_power(A, p=0.85).tolist()

    df = pd.DataFrame({'node':nodes,
                       'indegree': ind,
                       'outdegree': outd,
                       'pagerank': pr,
                       'adamic-adar-in': 0,
                       'adamic-adar-out': 0,
                       '2hoprw':0
                       }, columns=['node','indegree','outdegree','pagerank','adamic-adar-in','adamic-adar-out','2hoprw'])

    return df