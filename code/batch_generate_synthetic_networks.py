import os
from org.gesis.libs.arguments import init_batch_generate_network
from org.gesis.model.graph import DirectedGraph
from org.gesis.libs.network import get_nodes_metadata
from org.gesis.libs.io import save_csv
from org.gesis.libs.utils import printf

def get_path(params):
    return os.path.join(params.output, 'synthetic', params.model)

def run(params):

    if DirectedGraph.validate_params(params):
        DG = DirectedGraph(params.model,
                           N=params.N,
                           kmin=params.kmin,
                           density=params.density,
                           minority_fraction=params.minority_fraction,
                           gamma_m=params.gamma_m,
                           gamma_M=params.gamma_M,
                           h_mm=params.h_mm,
                           h_MM=params.h_MM,
                           triads_ratio=params.triads_ratio,
                           triads_pdf=params.triads_pdf)

        path = get_path(params)

        if DG.already_exists(path, epoch=params.epoch):
            printf('{} already done.'.format(DG.fn))
            return
        else:
            DG.create_network()
            DG.info()

            if params.output is not None:
                DG.save(path, epoch=params.epoch)

            ### NODE METADATA
            if params.metadata:
                df = get_nodes_metadata(DG.G, num_cores=10)
                fn = DG.fn.replace(".gpickle", '.csv')
                save_csv(df, fn)

if __name__ == "__main__":
    params = init_batch_generate_network()
    run(params)