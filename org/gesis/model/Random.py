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

def Random(N, fm, d, verbose=False, seed=None):
    '''
    Generates a Directed Random network.
    - param N: number of nodes
    - param fm: fraction of minorities
    - param d: edge density
    - param verbose: if True prints every steps in detail.
    - param seed: randommness seed for reproducibility
    '''
    np.random.seed(seed)
    start_time = time.time()
    
    # 1. Init nodes
    nodes, labels, NM, Nm = _init_nodes(N,fm)

    # 2. Init Directed Graph
    G = nx.DiGraph()
    G.graph = {'name':'Random', 'label':CLASS, 'groups': GROUPS}
    G.add_nodes_from([(n, {CLASS:l}) for n,l in zip(*[nodes,labels])])
    
    # 3. Init edges
    E = int(round(d * N * (N-1)))
        
    # INIT SUMMARY
    if verbose:
        print("Directed Graph:")
        print("N={} (M={}, m={})".format(N, NM, Nm))
        print("E={} (d={})".format(E, d))
        print('')
        
    # 5. Generative process
    while G.number_of_edges() < E:
        source = _pick_source(N)
        ns = nodes[source]
        target = _pick_target(source, N, labels)
        nt = nodes[target]
        
        if not G.has_edge(ns, nt):
            G.add_edge(ns, nt)

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

def _pick_source(N):
    '''
    Picks 1 (index) node as source (edge from) based on activity score.
    '''
    return np.random.choice(a=np.arange(N),size=1,replace=True)[0]
    
def _pick_target(source, N, labels):
    '''
    Given a (index) source node, it returns 1 (index) target node randomly.
    '''
    targets = [n for n in np.arange(N) if n!=source] # add here that n not in [source, neighbors of source]?
    return np.random.choice(a=targets,size=1,replace=True)[0]

################################################################
# Main
################################################################

if __name__ == "__main__":
    
    G = Random(N=1000, 
               fm=0.5, 
               d=0.01,
               verbose=True)
    