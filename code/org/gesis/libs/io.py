import os
import pandas as pd
import networkx as nx
from org.gesis.libs.utils import printf

def create_subfolders(fn):
    path = os.path.dirname(fn)
    os.makedirs(path, exist_ok = True)

def save_gpickle(G, fn):
    try:
        create_subfolders(fn)
        nx.write_gpickle(G, fn)
        printf('{} saved!'.format(fn))
    except Exception as ex:
        printf(ex)

def save_csv(df, fn):
    try:
        df.to_csv(fn)
        printf('{} saved!'.format(fn))
    except Exception as ex:
        printf(ex)
        return False
    return True

def read_csv(fn):
    df = None
    try:
        df = pd.read_csv(fn, index_col=0)
    except Exception as ex:
        printf(ex)
    return df