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

def directed_homophilic_barabasi_albert_graph(N, kmin, density, minority_fraction, h_mm, h_MM, gamma_m, gamma_M, seed=None):
    """
    Return random homophilic directed graph using BA preferential attachment model.

    A graph of N nodes is grown by attaching new nodes each with m
    edges that are preferentially attached to existing nodes with high
    degree. The connections are established by linking probability which
    depends on the connectivity of sites and the similitude (similarities).
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
    print('')

    ############################################################################
    # 2. Initializing graph
    ############################################################################
    G = nx.DiGraph()
    G.graph['name'] = 'DHBA'
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

                # 6.2.2. Picking target nodes based on homophily and pref. attachment
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
    smooth = 0.000001
    label = 'minority'
    label_source = G.node[source][label]

    # homphily and preferential attachment (indegree) commands
    prob = np.array([homophily_dic[(label_source, G.node[target][label])] * (G.in_degree(target) + 1) for target in target_list])
    prob += smooth
    prob = prob / prob.sum()
    return np.random.choice(a=target_list, size=kmin, replace=False, p=prob)


def estimate_homophily_empirical(graph, fm=None, EMM=None, EMm=None, EmM=None, Emm=None, gammaM_in=None, gammam_in=None, verbose=False):
    from org.gesis.libs.utils import get_edge_type_counts
    from org.gesis.libs.utils import get_minority_fraction

    hmm = []
    hMM = []
    diff = []

    if graph is not None and (fm is None or EMM is None or EMm is None or EmM is None or Emm is None):
        EMM, EMm, EmM, Emm = get_edge_type_counts(graph)
        fm = get_minority_fraction(graph)
    elif graph is None and (fm is None or EMM is None or EMm is None or EmM is None or Emm is None):
        raise Exception('Missing important parameters.')

    E = EMM + EMm + EmM + Emm
    min_min = Emm / E
    maj_maj = EMM / E
    min_maj = EmM / E
    maj_min = EMm / E
    fM = 1 - fm

    # calculating ca for directed
    K_m = min_min + maj_min
    K_M = maj_maj + min_maj
    K_all = K_m + K_M

    cm = (K_m) / K_all
    if verbose:
        print(cm)

    for h_mm_ in np.arange(0, 1.01, 0.01):
        for h_MM_ in np.arange(0, 1.01, 0.01):

            h_mm_analytical = h_mm_
            h_MM_analytical = h_MM_

            h_mM_analytical = 1 - h_mm_analytical
            h_Mm_analytical = 1 - h_MM_analytical

            if gammaM_in is None:
                try:
                    gamma_M = float(fM * h_MM_analytical) / (
                                (h_Mm_analytical * cm) + (h_MM_analytical * (2 - cm))) + float(
                        fm * h_mM_analytical) / ((h_mm_analytical * cm) + (h_mM_analytical * (2 - cm)))
                except RuntimeWarning:
                    if verbose:
                        print('break 2')
                    break
            else:
                gamma_M = gammaM_in


            if gammam_in is None:
                try:
                    gamma_m = float(fm * h_mm_analytical) / ((h_mm_analytical * cm) + (h_mM_analytical * (2 - cm))) + float(
                        fM * h_Mm_analytical) / ((h_Mm_analytical * cm) + (h_MM_analytical * (2 - cm)))
                except RuntimeWarning:
                    if verbose:
                        print('break 1')
                    break
            else:
                gamma_m = gammam_in


            K = 1 - gamma_m
            Z = 1 - gamma_M

            if ((fm * h_mm_analytical * Z) + ((1 - fm) * (1 - h_mm_analytical) * K) == 0 or (fM * h_MM_analytical * K) + (fm * (1 - h_MM_analytical) * Z)) == 0:
                break

            pmm_analytical = float(fm * h_mm_analytical * Z) / ((fm * h_mm_analytical * Z) + ((1 - fm) * (1 - h_mm_analytical) * K))
            pMM_analytical = float(fM * h_MM_analytical * K) / ((fM * h_MM_analytical * K) + (fm * (1 - h_MM_analytical) * Z))

            if min_min + min_maj + maj_maj == 0:
                # bipartite
                pmm_emp = 0.000001
                pMM_emp = 0.000001
            else:
                pmm_emp = float(min_min) / (min_min + min_maj)
                pMM_emp = float(maj_maj) / (maj_maj + maj_min)

            _diff = abs(pmm_emp - pmm_analytical) + abs(pMM_emp - pMM_analytical)
            diff.append(_diff)
            hmm.append(h_mm_analytical)
            hMM.append(h_MM_analytical)

            if verbose and _diff < 0.02:
                print()
                print('pmm_emp', pmm_emp, 'pmm_analytical', pmm_analytical)
                print('pMM_emp', pMM_emp, 'pMM_analytical', pMM_analytical)
                print('pMM_diff', abs(pmm_emp - pmm_analytical), 'pMM_diff', abs(pMM_emp - pMM_analytical) , 'diff' ,  _diff)
                print('h_mm_analytical', h_mm_analytical, 'h_MM_analytical', h_MM_analytical, 'cm_analytical', cm)

    best = np.argmin(diff)
    return hMM[best], hmm[best]

################################################################################
# main
################################################################################

if __name__ == '__main__':
    graph, _ = directed_homophilic_barabasi_albert_graph(N=1000,
                                                         kmin=2,
                                                         density=0.001,
                                                         minority_fraction=0.1,
                                                         h_MM=0.5,
                                                         h_mm=0.5,
                                                         gamma_m=3.0,
                                                         gamma_M=3.0)
    print(nx.info(graph))






