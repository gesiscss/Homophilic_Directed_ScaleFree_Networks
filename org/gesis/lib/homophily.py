################################################################################
# System dependencies
################################################################################
import numpy as np
from joblib import Parallel
from joblib import delayed
from itertools import product

################################################################################
# Local dependencies
################################################################################
from org.gesis.model.DPAH import DPAH
from org.gesis.lib import graph

################################################################################
# Functions
################################################################################

def get_metadata(g, steps, njobs=1, verbose=False, seed=None):
    '''
    Retrieves information form the given graph.
    - N number of nodes
    - fm fraction of minorities (node attribute from graph.graph['label']
    - d density
    - plo_* power law out-degree distribution from (M) majority and (m) minority
    - pli_* power law in-degree distribution from (M) majority and (m) minority
    - E** fraction of types of edges, e.g., EMM between majorities
    - _N number of nodes used for synthetic network
    - _d degree used for synthetic network
    - _mindiff minimum difference found between E** from real network and synthetic network.
    '''
    N = g.number_of_nodes()
    fm = graph.get_minority_fraction(g)
    d = graph.get_density(g)
    
    plo_M, plo_m = graph.get_outdegree_powerlaw_exponents(g)
    plo_M = plo_M.power_law.alpha
    plo_m = plo_m.power_law.alpha
    
    pli_M, pli_m = graph.get_indegree_powerlaw_exponents(g)
    pli_M = pli_M.power_law.alpha
    pli_m = pli_m.power_law.alpha
    
    EMM, EMm, EmM, Emm = graph.get_edge_type_counts(g, True)
    _N = 500
    _d = 3/_N
    hMM, hmm, _mindiff = infer_homophily_MLE(_N, fm, _d, plo_M, plo_m, EMM, EMm, EmM, Emm, steps, njobs, verbose, seed)
    
    return N, fm, d, plo_M, plo_m, pli_M, pli_m, EMM, EMm, EmM, Emm, hMM, hmm, _N, _d, _mindiff
    
def infer_homophily_MLE(N, fm, d, plo_M, plo_m, EMM, EMm, EmM, Emm, steps=0.05, njobs=1, verbose=False, seed=None):
    '''
    Infers the homophily value of a network given the DPAH model.
    '''
    h = np.arange(0.0, 1.0+steps, steps)
    hval = []
    diff = []
    
    if N < (1/d):
        N = int(round(1/d))
        
    if verbose:
        print("N={} d={} fm={}".format(N, d, fm))
        
    results = Parallel(n_jobs=njobs)(delayed(_infer_homophily_MLE)(N, fm, d, plo_M, plo_m, 
                                                                   EMM, EMm, EmM, Emm, hMM, hmm, verbose, i) 
                                             for i,(hMM,hmm) in enumerate(product(h,h)) )
    
    hMM, hmm, diff = zip(*results)
    mindiff = min(diff)
    
    if verbose:
        print("Minimum difference: {}".format(mindiff))
        
    mi = diff.index(mindiff)
    return hMM[mi], hmm[mi], mindiff
    
def _infer_homophily_MLE(N, fm, d, plo_M, plo_m, EMM, EMm, EmM, Emm, hMM, hmm, verbose, seed):
    '''
    Handler for infer_homophily_MLE to work in parallel.
    '''
    hMM = round(hMM,2)
    hmm = round(hmm,2)
    
    g = DPAH(N=N, fm=fm, d=d, plo_M=plo_M, plo_m=plo_m, h_MM=hMM, h_mm=hmm, verbose=False, seed=seed)
    eMM, eMm, emM, emm = graph.get_edge_type_counts(g, True)

    diff = abs(eMM-EMM)+abs(eMm-EMm)+abs(emM-EmM)+abs(emm-Emm)
    
    if verbose:
        print("hMM:{} hmm:{} diff:{}".format(hMM, hmm, diff))
        
    return (hMM, hmm, diff)