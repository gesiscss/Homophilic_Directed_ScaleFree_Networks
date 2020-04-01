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

def directed_barabasi_albert_graph(N, density, minority_fraction, gamma_m, gamma_M, kmin_M=2.0, kmin_m=2.0, kmax_M=None, kmax_m=None, seed=None, verbose=True):
    """
    Return random directed graph using BA preferential attachment model.

    A graph of N nodes is grown by attaching new nodes each with m
    edges that are preferentially attached to existing nodes with high
    degree. The connections are established by linking probability which
    depends on the connectivity of sites .

    Parameters
    ----------
    N : int
        Number of nodes

    density : float
        Density of the network: E / N(N-1)

    minority_fraction : float
        Fraction of minorities in the network

    kmin_M : int
        Minimum number of edges to attach from a new node (majority) to existing nodes

    kmax_M : int
        Maximum number of edges to attach from a new node (majority) to existing nodes

    kmin_m : int
        Minimum number of edges to attach from a new node (minority) to existing nodes

    kmax_m : int
        Maximum number of edges to attach from a new node (minority) to existing nodes

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
    """

    np.random.seed(seed)

    ############################################################################
    # 1. Initializing expected values
    ############################################################################
    KMIN_ACTIVITY = 1
    EXPECTED_E = int(round(density * N * (N - 1)))
    INIT_E = N * KMIN_ACTIVITY
    PA_E = EXPECTED_E - INIT_E
    PA_ADD = 20
    PA_START = INIT_E - int(round(PA_E / PA_ADD))

    if verbose:
        print("N: {}".format(N))
        print("density: {}".format(density))
        print("EXPECTED_E: {}".format(EXPECTED_E))
        print("INIT_EDGES (y): {}".format(INIT_E))
        print("PREF_ATTA_EDGES (y): {}".format(PA_E))
        print('PREF ATTA STARTS: {}'.format(PA_START))
        print('PREF ATTA ADD: {}'.format(PA_ADD))

    ############################################################################
    # 2. Initializing graph
    ############################################################################
    G = nx.DiGraph()
    G.graph['name'] = 'DBA'
    G.graph['class'] = CLASS_LABEL
    G.graph['minority'] = MINORITY
    G.graph['labels'] = LABELS
    G.graph['groups'] = GROUPS

    G.graph['N'] = N
    G.graph['density'] = density
    G.graph['fm'] = minority_fraction
    G.graph['kmin_M'] = kmin_M
    G.graph['kmax_M'] = kmax_M
    G.graph['kmin_m'] = kmin_m
    G.graph['kmax_m'] = kmax_m
    G.graph['gamma_M'] = gamma_M
    G.graph['gamma_m'] = gamma_m
    G.graph['seed'] = seed
    G.graph['label'] = LABEL

    ############################################################################
    # 3. Adding nodes with their respective membership class
    ############################################################################
    minority = int(minority_fraction * N)
    nodes = np.arange(N)
    minority_nodes = np.random.choice(a=nodes, size=minority, replace=False)
    G.add_nodes_from([(n, {'color': 'red' if n in minority_nodes else 'blue',
                           'minority': int(n in minority_nodes)}) for n in nodes])

    ############################################################################
    # 5. Activity (outdegree)
    ############################################################################
    activity_distribution = np.zeros(N)
    activity_distribution[minority_nodes] = power_law_distribution(minority, gamma_m, kmin_m, kmax_m)
    activity_distribution[[n for n in G.nodes() if n not in minority_nodes]] = power_law_distribution(N - minority, gamma_M, kmin_M, kmax_M)
    activity_distribution = activity_distribution / activity_distribution.sum()


    ############################################################################
    # 6. Preferential Attachment Algorithm + Direction
    ############################################################################
    ntarget = kmin_m + kmin_M
    target_list = np.arange(ntarget).tolist()
    total_edges = 0

    for source in np.arange(ntarget, N, 1):

        # 6.1. Adding new edge: new node --> existing node
        targets = _pick_targets(G, source, target_list, min(KMIN_ACTIVITY, len(target_list)))
        if len(targets) > 0:
            G.add_edges_from(zip([source] * len(targets), targets))
            total_edges += len(targets)
        target_list.append(source)

        # 6.2. Adding new edge: existing node --> existing node
        if source >= PA_START:
            counter_paa = 0

            # 6.2.1. Picking nodes based on activity_distribution (outdegree)
            existing_nodes = random_draw(target_list, activity_distribution)
            for existing_node in existing_nodes:

                kmin = kmin_m if existing_node in minority_nodes else kmin_M

                # 6.2.2. Picking target nodes based pref. attachment
                targets = _pick_targets(G, existing_node, target_list, len(target_list))
                new_added = 0
                for target in targets:
                    if (target != existing_node):
                        if not G.has_edge(existing_node, target):
                            G.add_edge(existing_node, target)
                            total_edges += 1
                            new_added += 1
                            counter_paa += 1

                        if new_added == kmin:
                            break

                # 6.2.3. Only adding the necessary
                if counter_paa >= PA_ADD:
                    break

    return G

def _pick_targets(G, source, target_list, kmin):
    # Only preferential attachment governs
    prob = np.array([G.in_degree(target) + 1 for target in target_list])
    prob = prob / prob.sum()
    return np.random.choice(a=target_list, size=kmin, replace=False, p=prob)

################################################################################
# main
################################################################################

if __name__ == '__main__':
    graph = directed_barabasi_albert_graph(N=1000,
                                           density=0.001,
                                           minority_fraction=0.1,
                                           gamma_m=3.0,
                                           gamma_M=3.0)
    print(nx.info(graph))
