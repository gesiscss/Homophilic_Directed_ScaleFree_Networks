import os
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
