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

def Null(N, fm, verbose=False, seed=None):
    '''
    Generates a Null Random network (a graph wihtout edges).
    - param N: number of nodes
    - param fm: fraction of minorities
    - verbose: if True prints every steps in detail.
    - seed: randommness seed for reproducibility
    '''
    np.random.seed(seed)
    start_time = time.time()
    
    # 1. Init nodes
    nodes, labels, NM, Nm = _init_nodes(N,fm)

    # 2. Init Directed Graph
    G = nx.DiGraph()
    G.graph = {'name':'Null', 'label':CLASS, 'groups': GROUPS}
    G.add_nodes_from([(n, {CLASS:l}) for n,l in zip(*[nodes,labels])])
    
    # 3. Init edges
    E = 0
        
    # INIT SUMMARY
    if verbose:
        print("Directed Graph:")
        print("N={} (M={}, m={})".format(N, NM, Nm))
        print("E={}".format(E))
        print('')
    
    duration = time.time() - start_time
    if verbose:
        print()
        print(G.graph)
        print(nx.info(G))
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

################################################################
# Main
################################################################

if __name__ == "__main__":
    
    G = Null(N=1000, 
             fm=0.5, 
             verbose=True)
    