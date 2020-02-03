#################################################
# Dependencies
#################################################
import math
from collections import Counter

#################################################
# Constants
#################################################

opp = {'m': 'n', 'M': 'N'}

KEYS = {'mmm': 'mmm', 'nnn': 'mmm',
        'MMM': 'MMM', 'NNN': 'MMM',

        'mnm': 'mnm', 'nmm': 'mnm', 'mmn': 'mnm',
        'nnm': 'mnm', 'nmn': 'mnm', 'mnn': 'mnm',

        'MNM': 'MNM', 'NMM': 'MNM', 'MMN': 'MNM',
        'NNM': 'MNM', 'NMN': 'MNM', 'MNN': 'MNM',

        'mmN': 'mmN', 'mNm': 'mmN', 'Nmm': 'mmN',
        'mNn': 'mmN', 'Nnm': 'mmN', 'nmN': 'mmN',

        'mmM': 'mmM', 'mMm': 'mmM', 'Mmm': 'mmM',
        'nNn': 'mmM', 'Nnn': 'mmM', 'nnN': 'mmM',

        'mMn': 'mMn', 'Mnm': 'mMn', 'nmM': 'mMn',
        'mnN': 'mMn', 'nNm': 'mMn', 'Nmn': 'mMn',

        'nMm': 'nMm', 'Mmn': 'nMm', 'mnM': 'nMm',
        'nnM': 'nMm', 'nMn': 'nMm', 'Mnn': 'nMm',

        'nMM': 'nMM', 'MMn': 'nMM', 'MnM': 'nMM',
        'nNM': 'nMM', 'NMn': 'nMM', 'MnN': 'nMM',

        'mMM': 'mMM', 'MMm': 'mMM', 'MmM': 'mMM',
        'nNN': 'mMM', 'NNn': 'mMM', 'NnN': 'mMM',

        'mNM': 'mNM', 'NMm': 'mNM', 'MmN': 'mNM',
        'nMN': 'mNM', 'MNn': 'mNM', 'NnM': 'mNM',

        'mMN': 'mMN', 'MNm': 'mMN', 'NmM': 'mMN',
        'mNN': 'mMN', 'NNm': 'mMN', 'NmN': 'mMN',
        }


#################################################
# Functions
#################################################

def get_code(key):
    if key not in KEYS:
        raise Exception('code does not exist: {}'.format(key))

    return KEYS[key]


def get_triads_ids():
    return ['mmm', 'mnm', 'MMM', 'MNM',
            'mmN', 'mmM', 'mMn', 'nMm',
            'nMM', 'mMM', 'mNM', 'mMN']


def get_labeled_potential_triads(G, label, source, middle, target):
    triads = []

    s = 'm' if G.node[source][label] else 'M'
    m = 'm' if G.node[middle][label] else 'M'
    t = 'm' if G.node[target][label] else 'M'

    # circular
    triads.append('{}{}{}'.format(s, m, t))

    # non-circular
    triads.append('{}{}{}'.format(s, m, opp[t]))

    triads.append('{}{}{}'.format(s, opp[m], t))

    triads.append('{}{}{}'.format(s, opp[m], opp[t]))

    return triads


def get_labeled_triads(G, label, source, middle, target):

    s = 'm' if G.node[source][label] else 'M'
    m = 'm' if G.node[middle][label] else 'M'
    t = 'm' if G.node[target][label] else 'M'

    triads = []

    # circular
    if target in G[middle] and source in G[target]:
        triads.append('{}{}{}'.format(s, m, t))

    # non-circular
    if target in G[middle] and target in G[source]:
        triads.append('{}{}{}'.format(s, m, opp[t]))

    if middle in G[target] and source in G[target]:
        triads.append('{}{}{}'.format(s, opp[m], t))

    if middle in G[target] and target in G[source]:
        triads.append('{}{}{}'.format(s, opp[m], opp[t]))

    return triads


def get_triads_from_edges(G, label):
    triad_counts = {i: 0 for i in get_triads_ids()}

    done = {}

    for source, middle in G.edges():
        target_s = set(list(G.predecessors(source)) + list(G.successors(source)))
        target_m = set(list(G.predecessors(middle)) + list(G.successors(middle)))
        targets = target_s.intersection(target_m) - set([source]) - set([middle])

        for target in targets:

            k = str(sorted([source, middle, target]))

            labeled_triads = get_labeled_triads(G, label, source, middle, target)

            for labeled_triad in labeled_triads:
                code = get_code(labeled_triad)

                if k in done.keys():
                    if code in done[k]:
                        code = None
                else:
                    done[k] = []

                if code is not None:
                    done[k].append(code)
                    triad_counts[code] += 1

    return triad_counts


def get_total_possible_triads_from_graph(G, label):
    if G.number_of_nodes() <= 2:
        raise Exception('There should be at least 3 nodes')

    tmp = Counter([n[1][label] for n in G.nodes(data=True)])

    try:
        m_counts = tmp[1]
        M_counts = tmp[0]
    except:
        m_counts = tmp['m']
        M_counts = tmp['M']

    return get_total_possible_triads(m_counts, M_counts)


def get_total_possible_triads(m_counts, M_counts):
    if m_counts + M_counts <= 2:
        raise Exception('There should be at least 3 nodes')

    mmm, MMM, mmM, MMm = 0, 0, 0, 0

    if m_counts > 2:
        mmm = 2 * (math.factorial(m_counts) / (math.factorial(m_counts - 3) * math.factorial(3)))

    if m_counts >= 2:
        mmM = 4 * M_counts * (math.factorial(m_counts) / (math.factorial(m_counts - 2) * math.factorial(2)))

    if M_counts > 2:
        MMM = 2 * (math.factorial(M_counts) / (math.factorial(M_counts - 3) * math.factorial(3)))

    if M_counts >= 2:
        MMm = 4 * m_counts * (math.factorial(M_counts) / (math.factorial(M_counts - 2) * math.factorial(2)))

    return mmm + MMM + mmM + MMm


#################################################
# End functions
#################################################

if __name__ == "__main__":
    print("computing triads.")
