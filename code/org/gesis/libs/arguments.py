import argparse
from org.gesis.libs.utils import printf

def init_batch_generate_network():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model', action='store',
                        dest='model',
                        required=True,
                        choices=["DBA","DH","DT","DHBA","DHTBA"],
                        help='Network type model (synthetic).',
                        )

    parser.add_argument('-N', action='store',
                        dest='N',
                        required=True,
                        type=int,
                        default=200,
                        help='Number of nodes.',
                        )

    parser.add_argument('-density', action='store',
                        dest='density',
                        required=True,
                        type=float,
                        default=0.001,
                        help='Edge density.')

    parser.add_argument('-f', action='store',
                        dest='minority_fraction',
                        required=True,
                        type=float,
                        default=0.5,
                        help='Minority fraction.')

    parser.add_argument('-kminM', action='store',
                        dest='kmin_M',
                        required=True,
                        type=int,
                        default=2,
                        help='Minimum degree Majority.')

    parser.add_argument('-kmaxM', action='store',
                        dest='kmax_M',
                        type=int,
                        default=None,
                        help='Maximum degree Majority.')

    parser.add_argument('-kminm', action='store',
                        dest='kmin_m',
                        required=True,
                        type=int,
                        default=2,
                        help='Minimum degree minority.')

    parser.add_argument('-kmaxm', action='store',
                        dest='kmax_m',
                        type=int,
                        default=None,
                        help='Maximum degree minority.')

    parser.add_argument('-gm', action='store',
                        dest='gamma_m',
                        required=True,
                        type=float,
                        default=3.0,
                        help='Exponent of power-law for outdegree distribution of minority nodes.')

    parser.add_argument('-gM', action='store',
                        dest='gamma_M',
                        required=True,
                        type=float,
                        default=3.0,
                        help='Exponent of power-law for outdegree distribution of majority nodes.')

    parser.add_argument('-hmm', action='store',
                        dest='h_mm',
                        type=float,
                        required=True,
                        help='Homophily within minorities.')

    parser.add_argument('-hMM', action='store',
                        dest='h_MM',
                        type=float,
                        default=None,
                        help='Homophily within majorities.')

    parser.add_argument('-tr', action='store',
                        dest='triads_ratio',
                        type=float,
                        #default=0.0,
                        help='Fraction of triads among all possible triads.')

    parser.add_argument('-tpdf',
                        nargs="+",
                        dest='triads_pdf',
                        type=float,
                        #default=[1/12]*12,
                        help='Fraction of triads among all possible triads [mmm, mnm, MMM, MNM, mmN, mmM, mMn, nMm, nMM, mMM, mNM, mMN].')

    parser.add_argument('-epoch', action='store',
                        dest='epoch',
                        type=int,
                        default=1,
                        help='A specific iteration (epoch out of iter).',
                        )

    parser.add_argument('-metadata', action='store_true',
                        dest='metadata',
                        help='Whether or not compute metadata of nodes (in/out degree, pagerank, COT, WTF).')

    parser.add_argument('-output', action='store',
                        dest='output',
                        default=None,
                        required=True,
                        help='Directory where to store the networkx file.')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    results = parser.parse_args()

    if results.h_MM is None and results.h_mm is not None:
        results.h_MM = results.h_mm

    print("===================================================")
    print("= ARGUMENTS PASSED:                               =")
    print("===================================================")
    print('model .................. = ', results.model)
    print('N ...................... = ', results.N)
    print('density ................ = ', results.density)
    print('minority_fraction ...... = ', results.minority_fraction)
    print('kmin_M ................. = ', results.kmin_M)
    print('kmax_M ................. = ', results.kmax_M)
    print('kmin_m ................. = ', results.kmin_m)
    print('kmax_m ................. = ', results.kmax_m)
    print('h_MM ................... = ', results.h_MM)
    print('h_mm ................... = ', results.h_mm)
    print('gamma_M ................ = ', results.gamma_M)
    print('gamma_m ................ = ', results.gamma_m)
    print('triads_ratio ........... = ', results.triads_ratio)
    print('triads_pdf ............. = ', results.triads_pdf)
    print('epoch .................. = ', results.epoch)
    print('metadata ............... = ', results.metadata)
    print('output ................. = ', results.output)
    print("===================================================")
    printf("init_batch_generate_network")
    return results


def init_batch_model_fit():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model', action='store',
                        dest='model',
                        required=True,
                        choices=["DBA", "DH", "DHBA", "DHTBA", "DT"],
                        help='Network type model (synthetic).',
                        )

    parser.add_argument('-dataset', action='store',
                        dest='dataset',
                        required=True,
                        choices=["aps", "apsgender3", "apsgender8", "github", "pokec", "wikipedia"],
                        help='Dataset',
                        )

    parser.add_argument('-N', action='store',
                        dest='N',
                        type=int,
                        default=None,
                        help='Number of nodes.',
                        )

    parser.add_argument('-kminm', action='store',
                        dest='kminm',
                        type=int,
                        default=None,
                        help='Minimun degree for minorities.',
                        )

    parser.add_argument('-kmaxm', action='store',
                        dest='kmaxm',
                        type=int,
                        default=None,
                        help='Maximum degree for minorities.',
                        )

    parser.add_argument('-kminM', action='store',
                        dest='kminM',
                        type=int,
                        default=None,
                        help='Minimun degree for majorities.',
                        )

    parser.add_argument('-kmaxM', action='store',
                        dest='kmaxM',
                        type=int,
                        default=None,
                        help='Maximum degree for majorities.',
                        )

    parser.add_argument('-density', action='store',
                        dest='density',
                        type=float,
                        default=None,
                        help='Edge density.',
                        )

    parser.add_argument('-epoch', action='store',
                        dest='epoch',
                        type=int,
                        default=1,
                        help='A specific iteration (epoch out of iter).',
                        )

    parser.add_argument('-output', action='store',
                        dest='output',
                        required=True,
                        default='results',
                        help='Directory where to store the networkx file.')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    results = parser.parse_args()

    print("===================================================")
    print("= ARGUMENTS PASSED:                               =")
    print("===================================================")
    print('model .................. = ', results.model)
    print('dataset ................ = ', results.dataset)
    print('N ...................... = ', results.N)
    print('kminM .................. = ', results.kminM)
    print('kminm .................. = ', results.kminm)
    print('density ................ = ', results.density)
    print('epoch .................. = ', results.epoch)
    print('output ................. = ', results.output)
    print("===================================================")
    printf("init_batch_model_fit")
    return results

def init_batch_node_attributes():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', action='store',
                        dest='dataset',
                        required=True,
                        choices=["aps", "github", "pokec", "wikipedia"],
                        help='Dataset')

    parser.add_argument('-nc', action='store',
                        dest='numcores',
                        type=int,
                        help='Dataset')

    parser.add_argument('-root', action='store',
                        dest='root',
                        required=True,
                        default='results',
                        help='Directory where the datasets are.')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    results = parser.parse_args()

    print("===================================================")
    print("= ARGUMENTS PASSED:                               =")
    print("===================================================")
    print('dataset ................ = ', results.dataset)
    print('numcores ............... = ', results.numcores)
    print('root ................... = ', results.root)
    print("===================================================")
    printf("init_batch_generate_network")
    return results