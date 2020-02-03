import argparse
from org.gesis.libs.utils import printf

def init_batch_generate_network():
    parser = argparse.ArgumentParser()

    parser.add_argument('-kind', action='store',
                        dest='kind',
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

    parser.add_argument('-m', action='store',
                        dest='m',
                        required=True,
                        type=int,
                        default=2,
                        help='Minimum degree.',
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
                        #default=0.5,
                        help='Homophily within minorities.')

    parser.add_argument('-hMM', action='store',
                        dest='h_MM',
                        type=float,
                        #default=0.5,
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

    parser.add_argument('-output', action='store',
                        dest='output',
                        default=None,
                        help='Directory where to store the networkx file.')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    results = parser.parse_args()

    print("===================================================")
    print("= ARGUMENTS PASSED:                               =")
    print("===================================================")
    print('kind ................... = ', results.kind)
    print('N ...................... = ', results.N)
    print('m ...................... = ', results.m)
    print('density ................ = ', results.density)
    print('minority_fraction ...... = ', results.minority_fraction)
    print('h_mm ................... = ', results.h_mm)
    print('h_MM ................... = ', results.h_MM)
    print('gamma_m ................ = ', results.gamma_m)
    print('gamma_M ................ = ', results.gamma_M)
    print('triads_ratio ........... = ', results.triads_ratio)
    print('triads_pdf ............. = ', results.triads_pdf)
    print('output ................. = ', results.output)
    print("===================================================")
    printf("INIT")
    return results


def init_batch_model_fit():
    parser = argparse.ArgumentParser()

    parser.add_argument('-kind', action='store',
                        dest='kind',
                        required=True,
                        choices=["DBA", "DH", "DT", "DHBA", "DHTBA"],
                        help='Network type model (synthetic).',
                        )

    parser.add_argument('-dataset', action='store',
                        dest='dataset',
                        required=True,
                        choices=["aps", "github", "pokec", "wikipedia"],
                        help='Dataset',
                        )

    parser.add_argument('-N', action='store',
                        dest='N',
                        required=True,
                        type=int,
                        default=200,
                        help='Number of nodes.',
                        )

    parser.add_argument('-epoch', action='store',
                        dest='epoch',
                        type=int,
                        default=1,
                        help='A specific iteration (epoch out of iter).',
                        )

    parser.add_argument('-iter', action='store',
                        dest='iter',
                        type=int,
                        default=1,
                        help='Total number of iterations of the same kind.',
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
    print('kind ................... = ', results.kind)
    print('N ...................... = ', results.N)
    print('epoch .................. = ', results.epoch)
    print('iter ................... = ', results.iter)
    print('output ................. = ', results.output)
    print("===================================================")
    printf("INIT")
    return results