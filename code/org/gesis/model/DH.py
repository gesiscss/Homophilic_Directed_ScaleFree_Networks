############################################
# System dependencies
############################################
import networkx as nx
import numpy as np

############################################
# Local dependencies
############################################
from org.gesis.libs.network import power_law_distribution
from org.gesis.libs.network import random_draw

################################################################################
# Constants
################################################################################

CLASS_LABEL = 'color'
MAJORITY = 'blue'
MINORITY = 'red'
LABELS = [MAJORITY, MINORITY]
GROUPS = ['M', 'm']
LABEL = 'minority'

################################################################################
# Model
################################################################################

# def directed_homophilic_graph(N, kmin, density, minority_fraction, h_mm, h_MM, gamma_m, gamma_M, seed=None):
#
#     np.random.seed(seed)
#     m = kmin
#
#     ############################################################################
#     # 1. Initializing expected values
#     ############################################################################
#     EXPECTED_E = int(round(density * N * (N - 1)))
#     START = int(round(N * 0.01))
#     ADD_NEW_EDGES = max(1, int(round(((m * (N - m)) - EXPECTED_E) / (START - N))))
#
#     print("density: {}".format(density))
#     print("EXPECTED_E: {}".format(EXPECTED_E))
#     print("ADD_NEW_EDGES (y): {}".format(ADD_NEW_EDGES))
#     print("START (x): {}".format(START))
#
#     ############################################################################
#     # 2. Initializing graph
#     ############################################################################
#     G = nx.DiGraph()
#     G.graph['name'] = 'DH'
#     G.graph['class'] = CLASS_LABEL
#     G.graph['minority'] = MINORITY
#     G.graph['labels'] = LABELS
#     G.graph['groups'] = GROUPS
#
#     G.graph['N'] = N
#     G.graph['kmin'] = kmin
#     G.graph['density'] = density
#     G.graph['fm'] = minority_fraction
#     G.graph['h_MM'] = h_MM
#     G.graph['h_mm'] = h_mm
#     G.graph['gamma_M'] = gamma_M
#     G.graph['gamma_m'] = gamma_m
#     G.graph['seed'] = seed
#
#     ############################################################################
#     # 3. Adding nodes with their respective membership class
#     ############################################################################
#     minority = int(minority_fraction * N)
#     minority_nodes = np.random.choice(a=np.arange(N), size=minority, replace=False)
#     G.add_nodes_from([(n, {'color': 'red' if n in minority_nodes else 'blue',
#                            'minority': int(n in minority_nodes)}) for n in range(N)])
#
#     ############################################################################
#     # 4. Homophily values
#     ############################################################################
#     homophily_dic = {(1, 1): h_mm,
#                      (1, 0): 1 - h_mm,
#                      (0, 0): h_MM,
#                      (0, 1): 1 - h_MM}
#
#     ############################################################################
#     # 5. Activity (outdegree)
#     ############################################################################
#     k_min = 1.0
#     activity_distribution = np.zeros(N)
#     activity_distribution[minority_nodes] = power_law_distribution(k_min, gamma_m, minority)
#     activity_distribution[[n for n in G.nodes() if n not in minority_nodes]] = power_law_distribution(k_min, gamma_M,
#                                                                                                       N - minority)
#     activity_distribution = activity_distribution / activity_distribution.sum()
#
#     ############################################################################
#     # 6. Preferential Attachment Algorithm + Homophily + Direction
#     ############################################################################
#     target_list = np.arange(m).tolist()
#     total_edges = 0
#     existing_dist = []
#     counter = 0
#
#     for source in np.arange(m, N, 1):
#
#         # 6.1. Adding new m edges: new node --> existing node (min outdegree)
#         targets = _pick_targets(G, source, target_list, homophily_dic, min(m, len(target_list)))
#         if len(targets) > 0:
#             G.add_edges_from(zip([source] * len(targets), targets))
#             total_edges += len(targets)
#         target_list.append(source)
#
#         # 6.2. Adding new edge: existing node --> existing node
#         new_added = 0
#         if source > START and total_edges < EXPECTED_E:
#
#             # 6.2.1. Picking nodes based on activity_distribution (outdegree)
#             existing_nodes = random_draw(target_list, activity_distribution)
#             for existing_node in existing_nodes:
#                 existing_dist.append(existing_node)
#
#                 # 6.2.2. Picking target nodes based on homophily
#                 targets = _pick_targets(G, existing_node, target_list, homophily_dic, len(target_list))
#                 for target in targets:
#                     if (target != existing_node):
#                         if not G.has_edge(existing_node, target):
#                             G.add_edge(existing_node, target)
#                             total_edges += 1
#                             new_added += 1
#                             break
#                         else:
#                             counter += 1
#
#                 # 6.2.3. Only adding the necessary
#                 if new_added >= ADD_NEW_EDGES or total_edges >= EXPECTED_E:
#                     break
#
#     print("{} times, an edge was not inserted".format(counter))
#     return G, existing_dist



def directed_homophilic_graph(N, kmin, density, minority_fraction, h_mm, h_MM, gamma_m, gamma_M, seed=None):
    """
    Return random homophilic directed graph.

    A graph of N nodes is grown by attaching new nodes each with m
    edges that are attached to existing nodes based on their group class.
    The connections are established by linking probability which
    depends on the similitude (similarities).
    similitude varies ranges from 0 to 1.

    Parameters
    ----------
    N : int
        Number of nodes

    kmin : int
        Number of edges to attach from a new node to existing nodes

    density : float
        Density of the network: E / N(N-1)

    minority_fraction : float
        Fraction of minorities in the network

    h_mm: float
        Homophily (similarity) among minority nodes. Value between 0.0 to 1.0

    h_MM: float
        Homophily (similarity) among majority nodes. Value between 0.0 to 1.0

    gamma_m: float
        Exponent of power-law for outdegree distribution of minority nodes.

    gamma_M: float
        Exponent of power-law for outdegree distribution of majority nodes.


    Returns
    -------
    G : nx.DiGraph


    Notes
    -----
    The initialization is a DiGraph with with N nodes and no edges.
    Minorities are represented as group=1 (color=red), and majorities as group=0 (color=blue).
    - Homophily between minorty: h_mm
    - Homophily between majority: h_MM
    """

    np.random.seed(seed)

    ############################################################################
    # 1. Initializing expected values
    ############################################################################
    EXPECTED_E = int(round(density * N * (N - 1)))
    ORGANIC_E = N * kmin
    NEW_EDGES = EXPECTED_E - ORGANIC_E
    START = N - NEW_EDGES if NEW_EDGES < N and NEW_EDGES > 0 else int(round(N * 1 / 100))
    ADD_NEW_EDGES = 1 if NEW_EDGES < N else int(round(NEW_EDGES / ((N - START) * kmin)))

    # EXPECTED_E = int(round(density * N * (N - 1)))
    # START = int(round(N * 0.01))
    # ADD_NEW_EDGES = max(1, int(round(((kmin * (N - kmin)) - EXPECTED_E) / (START - N))))

    print("density: {}".format(density))
    print("EXPECTED_E: {}".format(EXPECTED_E))
    print("ADD_NEW_EDGES (y): {}".format(ADD_NEW_EDGES))
    print("START (x): {}".format(START))

    ############################################################################
    # 2. Initializing graph
    ############################################################################
    G = nx.DiGraph()
    G.graph['name'] = 'DH'
    G.graph['class'] = CLASS_LABEL
    G.graph['minority'] = MINORITY
    G.graph['labels'] = LABELS
    G.graph['groups'] = GROUPS

    G.graph['N'] = N
    G.graph['kmin'] = kmin
    G.graph['density'] = density
    G.graph['fm'] = minority_fraction
    G.graph['h_MM'] = h_MM
    G.graph['h_mm'] = h_mm
    G.graph['gamma_M'] = gamma_M
    G.graph['gamma_m'] = gamma_m
    G.graph['seed'] = seed

    ############################################################################
    # 3. Adding nodes with their respective membership class
    ############################################################################
    minority = int(minority_fraction * N)
    minority_nodes = np.random.choice(a=np.arange(N), size=minority, replace=False)
    G.add_nodes_from([(n, {'color': 'red' if n in minority_nodes else 'blue',
                           'minority': int(n in minority_nodes)}) for n in range(N)])

    ############################################################################
    # 4. Homophily values
    ############################################################################
    h_mm = 0.9999999999 if h_mm == 1.0 else 0.0000000001 if h_mm == 0 else h_mm
    h_MM = 0.9999999999 if h_MM == 1.0 else 0.0000000001 if h_MM == 0 else h_MM

    homophily_dic = {(1, 1): h_mm,
                     (1, 0): 1 - h_mm,
                     (0, 0): h_MM,
                     (0, 1): 1 - h_MM}

    ############################################################################
    # 5. Activity (outdegree)
    ############################################################################
    k_min_activity = 1.0
    activity_distribution = np.zeros(N)
    activity_distribution[minority_nodes] = power_law_distribution(k_min_activity, gamma_m, minority)
    activity_distribution[[n for n in G.nodes() if n not in minority_nodes]] = power_law_distribution(k_min_activity, gamma_M,N - minority)
    activity_distribution = activity_distribution / activity_distribution.sum()

    ############################################################################
    # 6. Preferential Attachment Algorithm + Homophily + Direction
    ############################################################################
    target_list = np.arange(kmin).tolist()
    total_edges = 0
    existing_dist = []
    counter = 0

    for source in np.arange(kmin, N, 1):

        # 6.1. Adding new kmin edges: new node --> existing node (min outdegree)
        targets = _pick_targets(G, source, target_list, homophily_dic, min(kmin, len(target_list)))
        if len(targets) > 0:
            G.add_edges_from(zip([source] * len(targets), targets))
            total_edges += len(targets)
        target_list.append(source)

        # 6.2. Adding new edge: existing node --> existing node
        new_added = 0
        if source > START and total_edges < EXPECTED_E:

            # 6.2.1. Picking nodes based on activity_distribution (outdegree)
            existing_nodes = random_draw(target_list, activity_distribution)
            for existing_node in existing_nodes:
                existing_dist.append(existing_node)

                # 6.2.2. Picking target nodes based on homophily
                targets = _pick_targets(G, existing_node, target_list, homophily_dic, len(target_list))
                for target in targets:
                    if (target != existing_node):
                        if not G.has_edge(existing_node, target):
                            G.add_edge(existing_node, target)
                            total_edges += 1
                            new_added += 1
                            break
                        else:
                            counter += 1

                # 6.2.3. Only adding the necessary
                if new_added >= ADD_NEW_EDGES or total_edges >= EXPECTED_E:
                    break

    print("{} times, an edge was not inserted".format(counter))
    return G, existing_dist

def _pick_targets(G, source, target_list, homophily_dic, kmin):
    label = 'minority'
    label_source = G.node[source][label]
    # only homphily governs
    prob = np.array([homophily_dic[(label_source, G.node[target][label])] for target in target_list])
    prob = prob / prob.sum()
    return np.random.choice(a=target_list, size=kmin, replace=False, p=prob)


def estimate_homophily_empirical(graph, fm=None, EMM=None, EMm=None, EmM=None, Emm=None, verbose=False):
    hMM = EMM / (EMM + EMm)
    hmm = Emm / (Emm + EmM)
    return hMM, hmm

################################################################################
# main
################################################################################

if __name__ == '__main__':
    graph = directed_homophilic_graph(N=1000,
                                      kmin=2,
                                      density=0.001,
                                      minority_fraction=0.1,
                                      h_MM=0.5,
                                      h_mm=0.5,
                                      gamma_m=3.0,
                                      gamma_M=3.0)
    print(nx.info(graph))



