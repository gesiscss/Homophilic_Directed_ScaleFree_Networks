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
CLASS = 'block'
LABELS = [0,1] # 0 majority, 1 minority
GROUPS = ['M', 'm']

################################################################
# Functions
################################################################

def SBM(N, fm, h_MM, h_mm, verbose=False, seed=None):
    '''
    Generates a Directed Stockastic Block Model network.
    - param N: number of nodes
    - param fm: fraction of minorities
    - h_MM: homophily among majorities
    - h_mm: homophily among minorities
    - verbose: if True prints every steps in detail.
    - seed: randommness seed for reproducibility
    '''
    np.random.seed(seed)
    start_time = time.time()
    
    Nm = int(round(N*fm))
    NM = N-Nm
    sizes = [NM, Nm]
    probs = np.array([[h_MM,1-h_MM],[1-h_mm,h_mm]])
    G = nx.stochastic_block_model(sizes, probs, directed=True, seed=int(seed))
    G.graph = {'name':'SBM', 'label':CLASS, 'groups': GROUPS}
    E = G.number_of_edges()
    
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

################################################################
# Main
################################################################

if __name__ == "__main__":
    
    G = SBM(N=1000, 
             fm=0.5, 
             h_MM=0.5, 
             h_mm=0.5, 
             verbose=True)
    