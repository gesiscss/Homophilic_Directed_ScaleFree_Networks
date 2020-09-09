################################################################
# Systems' dependencies
################################################################
import time
import powerlaw
import numpy as np
import networkx as nx
from collections import Counter

################################################################
# Constants
################################################################
CLASS = 'm'
LABELS = [0,1] # 0 majority, 1 minority
GROUPS = ['M', 'm']

################################################################
# Functions
################################################################

def DBAH2(N, fm, d, plo_M, plo_m, h_MM, h_mm, verbose=False, seed=None):
    '''
    Generates a Directed Barabasi-Albert Homophilic network.
    - param N: number of nodes
    - param fm: fraction of minorities
    - param plo_M: power-law outdegree distribution majority class
    - param plo_m: power-law outdegree distribution minority class
    - h_MM: homophily among majorities
    - h_mm: homophily among minorities
    - verbose: if True prints every steps in detail.
    - seed: randommness seed for reproducibility
    '''
    np.random.seed(seed)
    start_time = time.time()
    
    # 1. Init nodes
    nodes, labels, NM, Nm = _init_nodes(N,fm)

    # 2. Init Directed Graph
    G = nx.DiGraph()
    G.graph = {'name':'DBA-Homophily', 'label':CLASS, 'groups': GROUPS}
    G.add_nodes_from([(n, {CLASS:l}) for n,l in zip(*[nodes,labels])])
    
    # 3. Init edges and indegrees
    E = int(round(d * N * (N-1)))
    minE = N
    paE = E-minE
    newE = 5
    lastE = int(round(paE/newE))
    startE = minE-lastE
    indegrees = np.zeros(N)
    if paE <= 0:
        raise Exception("Not enough edges.")
        
    # 4. Init Activity (out-degree)
    act_M = powerlaw.Power_Law(parameters=[plo_M], discrete=True).generate_random(NM)
    act_m = powerlaw.Power_Law(parameters=[plo_m], discrete=True).generate_random(Nm)
    activity = np.append(act_M, act_m)
    activity /= activity.sum()
    
    # 5. Init homophily
    homophily = np.array([[h_MM, 1-h_MM],[1-h_mm, h_mm]])
    
    # INIT SUMMARY
    
    if verbose:
        print("Directed Graph:")
        print("N={} (M={}, m={})".format(N, NM, Nm))
        print("E={} (d={})".format(E, d))
        print("Activity Power-Law outdegree: M={}, m={}".format(plo_M, plo_m))
        print("Homophily: h_MM={}, h_mm={}".format(h_MM, h_mm))
        print(homophily)
        print('')
        
    # 5. Generative process (pref. attach.)
    for source in np.arange(N):
        ns = nodes[source]
        target = _pick_target(source, N, labels, indegrees, homophily)
        nt = nodes[target]
        
        if not G.has_edge(ns, nt):
            G.add_edge(ns, nt)
            indegrees[target] += 1

        if verbose:
            ls = labels[source]
            lt = labels[target]
            print("{}->{} ({}{}): {}".format(ns, nt, 'm' if ls else 'M', 'm' if lt else 'M', G.number_of_edges()))

        # 6. Activity model (out-degree) and pref. attach.
        if source >= startE:
            for k in np.arange(newE):
                source = _pick_source(N, activity)
                ns = nodes[source]
                target = _pick_target(source, N, labels, indegrees, homophily)
                nt = nodes[target]

                if not G.has_edge(ns, nt):
                    G.add_edge(ns, nt)
                    indegrees[target] += 1

                if verbose:
                    ls = labels[source]
                    lt = labels[target]
                    print("{}->{} ({}{}): {}".format(ns, nt, 'm' if ls else 'M', 'm' if lt else 'M', G.number_of_edges()))
    
    while G.number_of_edges() < E:
        source = _pick_source(N, activity)
        ns = nodes[source]
        target = _pick_target(source, N, labels, indegrees, homophily)
        nt = nodes[target]

        if not G.has_edge(ns, nt):
            G.add_edge(ns, nt)
            indegrees[target] += 1

        if verbose:
            ls = labels[source]
            lt = labels[target]
            print("{}->{} ({}{}): {}".format(ns, nt, 'm' if ls else 'M', 'm' if lt else 'M', G.number_of_edges()))
        
        
    duration = time.time() - start_time
    if verbose:
        print()
        print(G.graph)
        print(nx.info(G))
        degrees = [d for n,d in G.out_degree()]
        print("min degree={}, max degree={}".format(min(degrees), max(degrees)))
        print(Counter(degrees))
        print(Counter([data[1][CLASS] for data in G.nodes(data=True)]))
        print()
        for k in [0,1]:
            fit = powerlaw.Fit(data=[d for n,d in G.out_degree() if G.node[n][CLASS]==k], discrete=True)
            print("{}: alpha={}, sigma={}, min={}, max={}".format('m' if k else 'M',
                                                                  fit.power_law.alpha, 
                                                                  fit.power_law.sigma, 
                                                                  fit.power_law.xmin, 
                                                                  fit.power_law.xmax))
        print()
        print("--- %s seconds ---" % (duration))

    return G

def _init_nodes(N, fm):
    '''
    Generates random nodes, and assigns them a binary label.
    param N: number of nodes
    param fm: fraction of minorities
    '''
    nodes = np.arange(N)
    np.random.shuffle(nodes)
    majority = int(round(N*(1-fm)))
    labels = [LABELS[i >= majority] for i,n in enumerate(nodes)]
    return nodes, labels, majority, N-majority

def _pick_source(N,activity):
    '''
    Picks 1 (index) node as source (edge from) based on activity score.
    '''
    return np.random.choice(a=np.arange(N),size=1,replace=True,p=activity)[0]
    
def _pick_target(source, N, labels, indegrees, homophily):
    '''
    Given a (index) source node, it returns 1 (index) target node based on homophily and pref. attachment (indegree).
    '''
    targets = [n for n in np.arange(N) if n!=source] # add here that n not in [source, neighbors of source]?
    probs = np.array([ homophily[labels[source],labels[n]] * (indegrees[n]+1) for n in targets])
    probs /= probs.sum()
    return np.random.choice(a=targets,size=1,replace=True,p=probs)[0]

################################################################
# Main
################################################################

if __name__ == "__main__":
    
    G = DBAH2(N=1000, 
             fm=0.5, 
             d=0.01, 
             plo_M=2.5, 
             plo_m=2.5, 
             h_MM=0.5, 
             h_mm=0.5, 
             verbose=True)
    