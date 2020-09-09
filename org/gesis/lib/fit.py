import os
import networkx as nx
from collections import Counter
import powerlaw
import pandas as pd

from org.gesis.lib import graph
from org.gesis.lib import io

    
def get_metadata(g_emp, fitpath):
    
    # empirical
    df = graph.get_node_metadata_as_dataframe(g_emp)
    df.loc[:,'kind'] = 'empirical'
    df.loc[:,'epoch'] = None
    
    # synthetic fit
    files = io.get_files(fitpath, ext=graph.EXT)
    for fn in files:
        try:
            epoch = int(fn.split('_')[-1].split(graph.EXT)[0])
        except:
            epoch = int(fn.split('-ID')[-1].split(graph.EXT)[0])
            
        fn = io.load_gpickle(os.path.join(fitpath,fn))
        tmp = graph.get_node_metadata_as_dataframe(fn)
        tmp.loc[:,'kind'] = 'synthetic'
        tmp.loc[:,'epoch'] = epoch
        
        df = df.append(tmp, ignore_index=True, sort=False)
        
    return df

#import sys
#sys.path.append('../../../code')
#from org.gesis.libs import io


# def get_params(G):
#     N = G.number_of_nodes()
#     label = G.graph['label']
#     fm = sum([1 for data in G.nodes(data=True) if data[1][label]==1]) / N
#     d = nx.density(G)
    
#     plo_M = powerlaw.Fit([d for n,d in G.out_degree() if G.node[n][G.graph['label']]==0], discrete=True).power_law.alpha
#     plo_m = powerlaw.Fit([d for n,d in G.out_degree() if G.node[n][G.graph['label']]==1], discrete=True).power_law.alpha
    
#     #wikipedia
#     #h_MM = 0.67
#     #h_mm = 0.58
    
#     #APS
#     #h_MM = 0.95
#     #h_mm = 0.93
    
#     #APS gender 3
#     #h_MM = 0.82
#     #h_mm = 0.27
    
#     # APS gender 8
#     h_MM = 0.50
#     h_mm = 0.60
    
#     #github
#     #h_MM = 0.55
#     #h_mm = 0.61
    
#     return N, fm, d, plo_M, plo_m, h_MM, h_mm

# def get_edge_types(G):
#     label = G.graph['label']
#     E = G.number_of_edges()
#     edges = Counter('{}{}'.format(G.graph['groups'][G.node[e[0]][label]], 
#                                   G.graph['groups'][G.node[e[1]][label]]) for e in G.edges())
#     print('EMM: {}'.format(edges['MM']/E))
#     print('EMm: {}'.format(edges['Mm']/E))
#     print('Emm: {}'.format(edges['mm']/E))
#     print('EmM: {}'.format(edges['mM']/E))