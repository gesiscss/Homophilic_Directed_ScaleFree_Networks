import os
from org.gesis.libs.arguments import init_batch_generate_network
from org.gesis.model.graph import DirectedGraph

def get_path(params):
    return os.path.join(params.output, 'synthetic', params.model)

def run(params):

    if DirectedGraph.validate_params(params):
        DG = DirectedGraph(params.model,
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
        DG.create_network()
        DG.info()

        if params.output is not None:
            path = get_path(params)
            DG.save(path, epoch=params.epoch)

if __name__ == "__main__":
    params = init_batch_generate_network()
    run(params)