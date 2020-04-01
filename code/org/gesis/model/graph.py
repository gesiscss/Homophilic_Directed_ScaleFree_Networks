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
from org.gesis.libs import utils

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
    def __init__(self, model, N, density, minority_fraction, kmin_M, kmax_M, kmin_m, kmax_m, gamma_M, gamma_m, h_MM, h_mm, triads_ratio, triads_pdf):
        self.model = model
        self.G = None
        self.N = N
        self.density = density
        self.minority_fraction = minority_fraction
        self.kmin_M = kmin_M
        self.kmax_M = kmax_M
        self.kmin_m = kmin_m
        self.kmax_m = kmax_m
        self.gamma_M = gamma_M
        self.gamma_m = gamma_m
        self.h_MM = h_MM
        self.h_mm = h_mm
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
            self.G = DBA.directed_barabasi_albert_graph(N=self.N,
                                                        density=self.density,
                                                        minority_fraction=self.minority_fraction,
                                                        kmin_M=self.kmin_M,
                                                        kmax_M=self.kmax_M,
                                                        kmin_m=self.kmin_m,
                                                        kmax_m=self.kmax_m,
                                                        gamma_m=self.gamma_m,
                                                        gamma_M=self.gamma_M,
                                                        seed=seed)
        elif self.model == kDH:
            self.G = DH.directed_homophilic_graph(N=self.N,
                                                  density=self.density,
                                                  minority_fraction=self.minority_fraction,
                                                  kmin_M=self.kmin_M,
                                                  kmax_M=self.kmax_M,
                                                  kmin_m=self.kmin_m,
                                                  kmax_m=self.kmax_m,
                                                  h_MM = self.h_MM,
                                                  h_mm = self.h_mm,
                                                  gamma_M=self.gamma_M,
                                                  gamma_m=self.gamma_m,
                                                  seed=seed)
        elif self.model == kDT:
            self.G,_ = DT.directed_triadic_graph(N=self.N,
                                                 density=self.density,
                                                 minority_fraction=self.minority_fraction,
                                                 kmin_M=self.kmin_M,
                                                 kmax_M=self.kmax_M,
                                                 kmin_m=self.kmin_m,
                                                 kmax_m=self.kmax_m,
                                                 gamma_m=self.gamma_m,
                                                 gamma_M=self.gamma_M,
                                                 triads_pdf=self.triads_pdf,
                                                 seed=seed)
        elif self.model == kDHBA:
            self.G = DHBA.directed_homophilic_barabasi_albert_graph(N=self.N,
                                                                    density=self.density,
                                                                    minority_fraction=self.minority_fraction,
                                                                    kmin_M=self.kmin_M,
                                                                    kmax_M=self.kmax_M,
                                                                    kmin_m=self.kmin_m,
                                                                    kmax_m=self.kmax_m,
                                                                    h_MM=self.h_MM,
                                                                    h_mm=self.h_mm,
                                                                    gamma_M=self.gamma_M,
                                                                    gamma_m=self.gamma_m,
                                                                    seed=seed)
        elif self.model == kDHTBA:
            self.G,_ = DHTBA.directed_homophilic_triadic_barabasi_albert_graph(N=self.N,
                                                                               density=self.density,
                                                                               minority_fraction=self.minority_fraction,
                                                                               kmin_M=self.kmin_M,
                                                                               kmax_M=self.kmax_M,
                                                                               kmin_m=self.kmin_m,
                                                                               kmax_m=self.kmax_m,
                                                                               h_mm=self.h_mm,
                                                                               h_MM=self.h_MM,
                                                                               gamma_m=self.gamma_m,
                                                                               gamma_M=self.gamma_M,
                                                                               triads_ratio=self.triads_ratio,
                                                                               seed=seed)
        self.duration = time.time() - self.duration
        self.update_properties()

        return self.G

    def update_properties(self):

        if self.h_mm is None or self.h_MM is None:
            if self.model == kDBA:
                self.h_MM = 0.5
                self.h_mm = 0.5
            elif self.model == kDH:
                EMM, EMm, EmM, Emm = utils.get_edge_type_counts(G)
                self.h_MM = EMM / (EMM + EMm)
                self.h_MM = Emm / (Emm + EmM)
            else:
                raise Exception('not implemented.')

    def info(self):
        print()
        print(nx.info(self.G))
        print(self.G.graph)
        print()
        print('created in {} seconds.'.format(self.duration))

    def already_exists(self, output, prefix=None, epoch=None):
        try:
            self.fn = self._get_fn(output, prefix=prefix, epoch=epoch)
            return os.path.exists(self.fn)
        except:
            return False


    def save(self, output, prefix=None, epoch=None):
        self.fn = self._get_fn(output, prefix=prefix, epoch=epoch)
        save_gpickle(self.G,self.fn)

    def _get_fn(self, output, prefix=None, epoch=None):
        fn = self.get_filename(prefix=prefix, epoch=epoch)
        return os.path.join(output,fn)

    def get_filename(self, prefix=None, epoch=None):

        if self.model == kDBA:
            fn = '<prefix><model>-N<N>-fm<fm>-d<d>-kminM<kminM>-kmaxM<kmaxM>-kminm<kminm>-kmaxm<kmaxm>-gM<gM>-gm<gm><ID>.gpickle'
        elif self.model == kDH:
            fn = '<prefix><model>-N<N>-fm<fm>-d<d>-kminM<kminM>-kmaxM<kmaxM>-kminm<kminm>-kmaxm<kmaxm>-hMM<hMM>-hmm<hmm>-gM<gM>-gm<gm><ID>.gpickle'
        elif self.model == kDT:
            fn = None
        elif self.model == kDHBA:
            fn = '<prefix><model>-N<N>-fm<fm>-d<d>-kminM<kminM>-kmaxM<kmaxM>-kminm<kminm>-kmaxm<kmaxm>-hMM<hMM>-hmm<hmm>-gM<gM>-gm<gm><ID>.gpickle'
        elif self.model == kDHTBA:
            fn = None

        fn = fn.replace('<prefix>','{}-'.format(prefix) if prefix is not None else '')
        fn = fn.replace('<model>',self.model)
        fn = fn.replace('<N>', str(self.N))
        fn = fn.replace('<fm>', str(round(self.minority_fraction,1)))
        fn = fn.replace('<d>', str(round(self.density, 5)))
        fn = fn.replace('<kminM>', str(self.kmin_M))
        fn = fn.replace('<kmaxM>', str(self.kmax_M))
        fn = fn.replace('<kminm>', str(self.kmin_m))
        fn = fn.replace('<kmaxm>', str(self.kmax_m))
        fn = fn.replace('<hMM>', str(round(self.h_MM,1)))
        fn = fn.replace('<hmm>', str(round(self.h_mm)))
        fn = fn.replace('<gM>', str(round(self.gamma_M)))
        fn = fn.replace('<gm>', str(round(self.gamma_m)))
        fn = fn.replace('<ID>', '-ID{}'.format(epoch) if epoch is not None else '')

        return fn


    '''
    Validates that the given model of network has the necessary parameters. 
    '''
    @staticmethod
    def validate_params(params):

        if params.model == kDBA:
            r = ['N','kmin','density','minority_fraction','gamma_m','gamma_M']
        elif params.model == kDH:
            r = ['N', 'kmin', 'density', 'minority_fraction', 'h_mm', 'h_MM', 'gamma_m', 'gamma_M']
        elif params.model == kDT:
            r = ['N', 'kmin', 'density', 'minority_fraction', 'gamma_m', 'gamma_M', 'triads_pdf']
        elif params.model == kDHBA:
            r = ['N', 'kmin', 'density', 'minority_fraction', 'h_mm', 'h_MM', 'gamma_m', 'gamma_M']
        elif params.model == kDHTBA:
            r = ['N', 'kmin', 'density', 'minority_fraction', 'h_mm', 'h_MM', 'gamma_m', 'gamma_M', 'triads_ratio', 'triads_pdf']

        g = 0
        w = 0
        for p in params.__dict__:

            if p in ['model','seed','output','epoch','metadata']:
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
