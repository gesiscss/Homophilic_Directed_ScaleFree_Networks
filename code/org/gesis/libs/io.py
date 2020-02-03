import os
import networkx as nx

def create_subfolders(fn):
    path = os.path.dirname(fn)
    os.makedirs(path, exist_ok = True)

def save_gpickle(G, fn):
    create_subfolders(fn)
    nx.write_gpickle(G, fn)