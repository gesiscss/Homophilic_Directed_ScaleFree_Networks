################################################################################
# System dependencies
################################################################################
import os
import time
import numpy as np
from scipy.stats import pearsonr

################################################################################
# Local dependencies
################################################################################
from org.gesis.lib import io

################################################################################
# Functions
################################################################################

def printf(txt):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print('{}\t{}'.format(ts,txt))

def lorenz_curve(X):
    X_lorenz = np.sort(X)
    X_lorenz = X_lorenz.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0)
    return X_lorenz

def gini(X):
    """Calculate the Gini coefficient of a numpy array."""
    # https://github.com/oliviaguest/gini/blob/master/gini.py
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    X = X.flatten()
    if np.amin(X) < 0:
        # Values cannot be negative:
        X -= np.amin(X)
    # Values cannot be 0:
    X += 0.0000001
    # Values must be sorted:
    X = np.sort(X)
    # Index per array element:
    index = np.arange(1, X.shape[0] + 1)
    # Number of array elements:
    n = X.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * X)) / (n * np.sum(X)))

def mean_error(y_true, y_pred):
    '''
    If the mean error is (+) then it has been over-estimated.
    Otherwise, it has been under-estimated.
    '''
    return np.mean(np.array(y_pred) - np.array(y_true))
    