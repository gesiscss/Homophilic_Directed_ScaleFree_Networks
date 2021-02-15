################################################################################
# System dependencies
################################################################################
import os
import powerlaw
import pandas as pd
import networkx as nx
from collections import Counter

################################################################################
# Local dependencies
################################################################################
from org.gesis.lib import io
from org.gesis.lib import graph

################################################################################
# Metadata (empirical and fit)
################################################################################

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
