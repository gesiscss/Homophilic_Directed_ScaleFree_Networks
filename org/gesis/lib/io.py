################################################################################
# System dependencies
################################################################################
import os
import pickle
import pandas as pd
import networkx as nx

################################################################################
# Local dependencies
################################################################################
from org.gesis.lib import utils

################################################################################
# Functions
################################################################################

def create_subfolders(fn):
    path = os.path.dirname(fn)
    os.makedirs(path, exist_ok = True)

def get_files(path, prefix=None, ext=None):
    if prefix is not None and ext is not None:
        return [fn for fn in os.listdir(path) if fn.endswith(ext) and fn.startswith(prefix)]
    elif prefix is not None:
        return [fn for fn in os.listdir(path) if fn.startswith(prefix)]
    return [fn for fn in os.listdir(path) if fn.endswith(ext)]

def load_gpickle(fn):
    g = None
    try:
        g = nx.read_gpickle(fn)
    except Exception as ex:
        utils.printf(ex)
    return g

def save_gpickle(G, fn):
    try:
        create_subfolders(fn)
        nx.write_gpickle(G, fn)
        utils.printf('{} saved!'.format(fn))
    except Exception as ex:
        utils.printf(ex)

def save_csv(df, fn):
    try:
        create_subfolders(fn)
        df.to_csv(fn)
        utils.printf('{} saved!'.format(fn))
    except Exception as ex:
        utils.printf(ex)
        return False
    return True

def read_csv(fn, index_col=0):
    df = None
    try:
        df = pd.read_csv(fn, index_col=index_col)
    except Exception as ex:
        utils.printf(ex)
    return df

def save_pickle(obj, fn):
    try:
        create_subfolders(fn)
        with open(fn,'wb') as f:
            pickle.dump(obj, f)
        utils.printf('{} saved!'.format(fn))
    except Exception as ex:
        utils.printf(ex)

def read_pickle(fn):
    obj = None
    try:
        with open(fn,'rb') as f:
            obj = pickle.load(f)
    except Exception as ex:
        utils.printf(ex)
    return obj

def save_text(txt, fn):
    try:
        create_subfolders(fn)
        with open(fn,'w') as f:
            f.write(txt)
        utils.printf('{} saved!'.format(fn))
    except Exception as ex:
        utils.printf(ex)
        return False
    return True

    