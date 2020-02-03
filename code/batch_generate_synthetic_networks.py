from org.gesis.libs.arguments import init_batch_generate_network
from org.gesis.model.network import DirectedNetwork

def run(params):

    if DirectedNetwork.validate_params(params):
        DN = DirectedNetwork(params.kind,
                             N=params.N,
                             m=params.m,
                             density=params.density,
                             minority_fraction=params.minority_fraction,
                             gamma_m=params.gamma_m,
                             gamma_M=params.gamma_M,
                             h_mm=params.h_mm,
                             h_MM=params.h_MM,
                             triads_ratio=params.triads_ratio,
                             triads_pdf=params.triads_pdf)
        DN.create_network()
        DN.info()

if __name__ == "__main__":
    params = init_batch_generate_network()
    run(params)