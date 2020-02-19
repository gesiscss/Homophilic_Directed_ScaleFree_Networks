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
from org.gesis.libs import triads as tri

################################################################################
# Constants
################################################################################

TRIADS_NTYPES = 12
CLASS_LABEL = 'color'
MINORITY = 'red'
MAJORITY = 'blue'
LABELS = [MAJORITY, MINORITY]
GROUPS = ['M', 'm']
LABEL = 'minority'
TRIAD_IDS = tri.get_triads_ids()

################################################################################
# Model
################################################################################

def directed_triadic_graph(N, kmin, density, minority_fraction, gamma_m, gamma_M,
                           triads_pdf=[1 / TRIADS_NTYPES] * TRIADS_NTYPES, seed=None):
    """
    Return random triadic directed graph.

    A graph of N nodes is grown by attaching new nodes each with m
    edges. The connections are established by creating triads.

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

    gamma_m: float
        Exponent of power-law for outdegree distribution of minority nodes.

    gamma_M: float
        Exponent of power-law for outdegree distribution of majority nodes.

    triads_pdf : list of floats
        Probability density function

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
    print('')

    ############################################################################
    # 2. Initializing graph
    ############################################################################
    G = nx.DiGraph()
    G.graph['name'] = 'DT'
    G.graph['class'] = CLASS_LABEL
    G.graph['minority'] = MINORITY
    G.graph['labels'] = LABELS
    G.graph['groups'] = GROUPS

    G.graph['N'] = N
    G.graph['kmin'] = kmin
    G.graph['density'] = density
    G.graph['fm'] = minority_fraction
    G.graph['gamma_M'] = gamma_M
    G.graph['gamma_m'] = gamma_m
    G.graph['triads_pdf'] = triads_pdf
    G.graph['seed'] = seed
    G.graph['label'] = LABEL

    ############################################################################
    # 3. Adding nodes with their respective membership class
    ############################################################################
    minority = int(minority_fraction * N)
    minority_nodes = np.random.choice(a=np.arange(N), size=minority, replace=False)
    G.add_nodes_from([(n, {CLASS_LABEL: MINORITY if n in minority_nodes else MAJORITY,
                           LABEL: int(n in minority_nodes),
                           'group': GROUPS[int(n in minority_nodes)],
                           }) for n in range(N)])

    ############################################################################
    # 5. Activity (outdegree)
    ############################################################################
    k_min_activity = 1.0
    activity_distribution = np.zeros(N)
    activity_distribution[minority_nodes] = power_law_distribution(k_min_activity, gamma_m, minority)
    activity_distribution[[n for n in G.nodes() if n not in minority_nodes]] = power_law_distribution(k_min_activity, gamma_M, N - minority)
    activity_distribution = activity_distribution / activity_distribution.sum()

    ############################################################################
    # 6. Preferential Attachment Algorithm + Homophily + Direction
    ############################################################################
    target_list = np.arange(kmin).tolist()
    total_edges = 0
    existing_dist = []
    counter = 0

    for source in np.arange(kmin, N, 1):

        tmp = _create_triads(G, source, target_list, min(kmin, len(target_list)), triads_pdf)

        if tmp > 0:
            total_edges += tmp
            target_list.append(source)

        # 6.2. Adding new edge: existing node --> existing node
        new_added = 0
        if source > START and total_edges < EXPECTED_E:

            # 6.2.1. Picking nodes based on activity_distribution (outdegree) - powerlaw
            existing_nodes = random_draw(target_list, activity_distribution)
            for existing_node in existing_nodes:
                existing_dist.append(existing_node)

                # 6.2.2. Picking target nodes based on homophily + pref attach. or triads
                targets = _pick_targets(G, existing_node, target_list, len(target_list), triads_pdf)
                if targets is None:
                    continue

                for target in targets:
                    if (target != existing_node):
                        if not G.has_edge(existing_node, target):
                            G.add_edge(existing_node, target)
                            total_edges += 1
                            new_added += 1
                            break  # so, only adds 1
                        else:
                            counter += 1

                # 6.2.3. Only adding the necessary
                if new_added >= ADD_NEW_EDGES or total_edges >= EXPECTED_E:
                    break

    print("{} times, an edge was not inserted".format(counter))
    return G, existing_dist

def _create_triads(G, source, target_list, kmin, triads_pdf):
    counts = 0
    triads_type = []
    for index in np.random.choice(a=len(triads_pdf), size=kmin, p=triads_pdf):
        triads_type.append(TRIAD_IDS[index])

    for i, target in enumerate(target_list):
        for middle in target_list[i + 1:]:
            for keycode in tri.get_labeled_potential_triads(G, 'group', source, middle, target):
                code = tri.get_code(keycode)

                if code in triads_type:
                    if code[0] in ['m', 'M']:
                        G.add_edge(source, middle)
                    else:
                        G.add_edge(middle, source)

                    if code[1] in ['m', 'M']:
                        G.add_edge(middle, target)
                    else:
                        G.add_edge(target, middle)

                    if code[2] in ['m', 'M']:
                        G.add_edge(target, source)
                    else:
                        G.add_edge(source, target)

                    counts += 1

                    if counts == kmin:
                        return counts

    return counts


def _pick_targets(G, source, target_list, kmin, triads_pdf):
    prob = np.array([0 for target in target_list])

    triads_type = []
    for index in np.random.choice(a=len(triads_pdf), size=kmin, p=triads_pdf):
        triads_type.append(TRIAD_IDS[index])

    source_neighbors = set(G.neighbors(source))

    for i, target in enumerate(target_list):
        target_neighbors = set(G.neighbors(target))

        for middle in source_neighbors.intersection(target_neighbors):
            for keycode in tri.get_labeled_triads(G, 'group', source, middle, target):
                code = tri.get_code(keycode)

                if code in triads_type:
                    prob[i] += 1

    if prob.sum() == 0:
        return None

    prob = prob / prob.sum()
    return np.random.choice(a=target_list, size=min([kmin, np.count_nonzero(prob)]), replace=False, p=prob)


################################################################################
# main
################################################################################

if __name__ == '__main__':
    graph = directed_triadic_graph(N=1000,
                                   kmin=2,
                                   density=0.001,
                                   minority_fraction=0.1,
                                   gamma_m=3.0,
                                   gamma_M=3.0,
                                   triads_pdf=[0.34, 0.2, 0.1, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
    print(nx.info(graph))