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

def get_nodes_metadata(graph):
    nodes = list(graph.nodes())
    A = nx.adjacency_matrix(graph, nodes).astype(np.int8) #csr
    ind = A.sum(axis=0).flatten().tolist()[0]
    outd = A.sum(axis=1).flatten().tolist()[0]
    pr = pagerank_power(A, p=0.85).tolist()
    minoriy = [graph.node[n][graph.graph['label'][0]] for n in nodes]

    df = pd.DataFrame({'node':nodes,
                       'minority':minoriy,
                       'indegree': ind,
                       'outdegree': outd,
                       'pagerank': pr,
                       'adamic-adar-in': 0,
                       'adamic-adar-out': 0,
                       '2hoprw':0
                       }, columns=['node','minority','indegree','outdegree','pagerank','adamic-adar-in','adamic-adar-out','2hoprw'])

    return df