from collections import Counter
import numpy as np
import powerlaw
import time


CLASSNAME = 'minority'

def printf(txt):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print('{}\t{}'.format(ts,txt))

def get_graph_metadata(graph, attribute):

    if attribute in graph.graph:
        return graph.graph[attribute]
    return None

def get_edge_type_counts(graph):
    counts = Counter(['{}{}'.format(graph.graph['groups'][graph.node[edge[0]][CLASSNAME]],
                                    graph.graph['groups'][graph.node[edge[1]][CLASSNAME]]) for edge in graph.edges()])
    return counts['MM'], counts['Mm'], counts['mM'], counts['mm']

def get_minority_fraction(graph):
    b = Counter([graph.node[n][CLASSNAME] for n in graph.nodes()]).most_common()[1][1] / graph.number_of_nodes()
    return b

def get_min_degree(graph):
    return min([d for n, d in graph.degree()])

def fit_power_law(data):
    return powerlaw.Fit(data, xmin=min(data), xmax=max(data), discrete=not (min(data) > 0 and min(data) < 1))

def get_outdegree_powerlaw_exponents(graph):
    x = np.array([d for n, d in graph.out_degree() if graph.node[n][CLASSNAME] == 0])
    fitM = fit_power_law(x)

    x = np.array([d for n, d in graph.out_degree() if graph.node[n][CLASSNAME] == 1])
    fitm = fit_power_law(x)

    return fitM.power_law.alpha, fitM.power_law.xmin, fitm.power_law.alpha, fitm.power_law.xmin

def get_indegree_powerlaw_exponents(graph):
    x = np.array([d for n, d in graph.in_degree() if graph.node[n][CLASSNAME] == 0])
    fitM = fit_power_law(x)

    x = np.array([d for n, d in graph.in_degree() if graph.node[n][CLASSNAME] == 1])
    fitm = fit_power_law(x)

    return fitM.power_law.alpha, fitM.power_law.xmin, fitm.power_law.alpha, fitm.power_law.xmin

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



    # x = np.array([d for n, d in graph.out_degree() if graph.node[n][CLASSNAME] == 0])
    # gamma_M, sigma_M = get_exponent(x)
    #
    # x = np.array([d for n, d in graph.out_degree() if graph.node[n][CLASSNAME] == 1])
    # gamma_m, sigma_m = get_exponent(x)
    #
    # return gamma_M, sigma_M, gamma_m, sigma_m


    #
    # x = np.array([d for n, d in graph.in_degree() if graph.node[n][CLASSNAME] == 0])
    # gamma_M, sigma_M = get_exponent(x)
    #
    # x = np.array([d for n, d in graph.in_degree() if graph.node[n][CLASSNAME] == 1])
    # gamma_m, sigma_m = get_exponent(x)
    #
    # return gamma_M, sigma_M, gamma_m, sigma_m

# def get_homophily(graph, smooth=1):
#     precision = 1
#     fm = round(get_minority_fraction(graph),precision)
#     EMM, EMm, EmM, Emm = get_edge_type_counts(graph)
#
#     Emm += smooth
#     EmM += smooth
#     EMM += smooth
#     EMm += smooth
#
#     fM = 1 - fm
#     precision = 5
#
#     eMM = round(float(EMM) / (Emm + EmM + EMm + EMM),precision)
#     hMM = float(-2 * eMM * fM * fm) / ((eMM * (fM ** 2)) - (2 * eMM * fM * fm) + (eMM * (fm ** 2) - (fM ** 2)))
#
#     emm = round(float(Emm) / (Emm + EmM + EMm + EMM), precision)
#     hmm = float(-2 * emm * fm * fM) / ((emm * (fm ** 2)) - (2 * emm * fm * fM) + (emm * (fM ** 2) - (fm ** 2)))
#
#     return hMM, hmm

# def PageRank(A, s = .85, maxerr = 1e-06):
#     """
#     https://gist.github.com/diogojc/1338222/84d767a68da711a154778fb1d00e772d65322187
#     Computes the pagerank for each of the n states
#     Parameters
#     ----------
#     AG: matrix representing state transitions
#        Aij is a binary value representing a transition from state i to j.
#     s: probability of following a transition. 1-s probability of teleporting
#        to another state.
#     maxerr: if the sum of pageranks between iterations is bellow this we will
#             have converged.
#     """
#     n = A.shape[0]
#
#     # transform G into markov matrix A
#     A = csc_matrix(A,dtype=np.float)
#     rsums = np.array(A.sum(1))[:,0]
#     ri, ci = A.nonzero()
#     A.data /= rsums[ri]
#
#     # bool array of sink states
#     sink = rsums==0
#
#     # Compute pagerank r until we converge
#     ro, r = np.zeros(n), np.ones(n)
#     while np.sum(np.abs(r-ro)) > maxerr:
#         ro = r.copy()
#         # calculate each pagerank at a time
#         for i in np.arange(0,n):
#             # inlinks of state i
#             Ai = np.array(A[:,i].todense())[:,0]
#             # account for sink states
#             Di = sink / float(n)
#             # account for teleportation to state i
#             Ei = np.ones(n) / float(n)
#
#             r[i] = ro.dot( Ai*s + Di*s + Ei*(1-s) )
#
#     # return normalized pagerank
#     return r/float(sum(r))