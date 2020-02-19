####################################################################
# System dependencies
####################################################################
import networkx as nx
import time
import os

####################################################################
# Local dependencies
####################################################################
from org.gesis.model import DBA
from org.gesis.model import DH
from org.gesis.model import DT
from org.gesis.model import DHBA
from org.gesis.model import DHTBA
from org.gesis.libs.io import save_gpickle

####################################################################
# Constants
####################################################################
kDBA = 'DBA'
kDH = 'DH'
kDT = 'DT'
kDHBA = 'DHBA'
kDHTBA = 'DHTBA'

####################################################################
# functions
####################################################################
def get_homophily(kind, fm, EMM, EMm, EmM, Emm, gammaM_in, gammam_in):
    if kind in [kDBA, kDT]:
        return 0.5, 0.5
    if kind == kDH:
        return DH.estimate_homophily_empirical(graph=None, fm=fm, EMM=EMM, EMm=EMm, EmM=EmM, Emm=Emm, verbose=False)
    if kind == kDHBA:
        return DHBA.estimate_homophily_empirical(graph=None, fm=fm, EMM=EMM, EMm=EMm, EmM=EmM, Emm=Emm, gammaM_in=gammaM_in, gammam_in=gammam_in, verbose=False)
    if kind == kDHTBA:
        raise NotImplementedError('Not implemented yet.')

####################################################################
# main class
####################################################################
class DirectedGraph(object):

    '''
    Cosntructor
    '''
    def __init__(self, model, N, kmin, density, minority_fraction, gamma_m, gamma_M, h_mm, h_MM, triads_ratio, triads_pdf):
        self.model = model
        self.G = None
        self.N = N
        self.kmin = kmin
        self.density = density
        self.minority_fraction = minority_fraction
        self.gamma_m = gamma_m
        self.gamma_M = gamma_M
        self.h_mm = h_mm
        self.h_MM = h_MM
        self.triads_ratio = triads_ratio
        self.triads_pdf = triads_pdf
        self.duration = 0
        self.fn = None

    '''
    Handler to create a new network given its kind
    '''
    def create_network(self, seed=None):

        self.duration = time.time()

        if self.model == kDBA:
            self.G,_ = DBA.directed_barabasi_albert_graph(N=self.N,
                                                          kmin=self.kmin,
                                                          density=self.density,
                                                          minority_fraction=self.minority_fraction,
                                                          gamma_m=self.gamma_m,
                                                          gamma_M=self.gamma_M,
                                                          seed=seed)
        elif self.model == kDH:
            self.G ,_ = DH.directed_homophilic_graph(N=self.N,
                                                    kmin=self.kmin,
                                                    density=self.density,
                                                    minority_fraction=self.minority_fraction,
                                                    h_mm = self.h_mm,
                                                    h_MM = self.h_MM,
                                                    gamma_m=self.gamma_m,
                                                    gamma_M=self.gamma_M,
                                                    seed=seed)
        elif self.model == kDT:
            self.G,_ = DT.directed_triadic_graph(N=self.N,
                                                 kmin=self.kmin,
                                                 density=self.density,
                                                 minority_fraction=self.minority_fraction,
                                                 gamma_m=self.gamma_m,
                                                 gamma_M=self.gamma_M,
                                                 triads_pdf=self.triads_pdf,
                                                 seed=seed)
        elif self.model == kDHBA:
            self.G,_ = DHBA.directed_homophilic_barabasi_albert_graph(N=self.N,
                                                                      kmin=self.kmin,
                                                                      density=self.density,
                                                                      minority_fraction=self.minority_fraction,
                                                                      h_mm=self.h_mm,
                                                                      h_MM=self.h_MM,
                                                                      gamma_m=self.gamma_m,
                                                                      gamma_M=self.gamma_M,
                                                                      seed=seed)
        elif self.model == kDHTBA:
            self.G,_ = DHTBA.directed_homophilic_triadic_barabasi_albert_graph(N=self.N,
                                                                               kmin=self.kmin,
                                                                               density=self.density,
                                                                               minority_fraction=self.minority_fraction,
                                                                               h_mm=self.h_mm,
                                                                               h_MM=self.h_MM,
                                                                               gamma_m=self.gamma_m,
                                                                               gamma_M=self.gamma_M,
                                                                               triads_ratio=self.triads_ratio,
                                                                               seed=seed)
        self.duration = time.time() - self.duration

        return self.G

    def info(self):
        print()
        print(nx.info(self.G))
        print(self.G.graph)
        print()
        print('created in {} seconds.'.format(self.duration))

    def save(self, output, prefix=None, epoch=None):
        self.fn = self.get_filename(prefix, epoch)
        self.fn = os.path.join(output,self.fn)
        save_gpickle(self.G,self.fn)

    def get_filename(self, prefix=None, epoch=None):
        return '{}{}-N{}-kmin{}-fm{}-hMM{}-hmm{}{}.gpickle'.format('{}-'.format(prefix) if prefix is not None else '',
                                                                 self.model,
                                                                 self.N,
                                                                 self.kmin,
                                                                 round(self.minority_fraction,1),
                                                                 round(self.h_MM,2),
                                                                 round(self.h_mm,2),
                                                                 '-ID{}'.format(epoch) if epoch is not None else '')


    '''
    Validates that the given model of network has the necessary parameters. 
    '''
    @staticmethod
    def validate_params(params):

        if params.model == kDBA:
            r = ['N','m','density','minority_fraction','gamma_m','gamma_M']
        elif params.model == kDH:
            r = ['N', 'm', 'density', 'minority_fraction', 'h_mm', 'h_MM', 'gamma_m', 'gamma_M']
        elif params.model == kDT:
            r = ['N', 'm', 'density', 'minority_fraction', 'gamma_m', 'gamma_M', 'triads_pdf']
        elif params.model == kDHBA:
            r = ['N', 'm', 'density', 'minority_fraction', 'h_mm', 'h_MM', 'gamma_m', 'gamma_M']
        elif params.model == kDHTBA:
            r = ['N', 'm', 'density', 'minority_fraction', 'h_mm', 'h_MM', 'gamma_m', 'gamma_M', 'triads_ratio', 'triads_pdf']

        g = 0
        w = 0
        for p in params.__dict__:

            if p in ['model','seed','output']:
                continue

            if p not in r and getattr(params,p) is not None:
                w += 1
                setattr(params, p, None)
            elif getattr(params,p) is not None:
                g += 1

            print('{} [{}]'.format(p, 'unnecesary' if p not in r else 'ok' if p in r and getattr(params,p) is not None else 'missing'))

        if len(r) == g:
            print('- All required parameters passed')
        else:
            print('- {} Missing parameters.'.format(len(r) - g))

        if w > 0:
            print('- {} Unnecesary parameters passed (set to None)'.format(w))

        print("===================================================")

        return len(r)==g
