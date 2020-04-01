import os
import numpy as np
import pandas as pd
from org.gesis.libs.arguments import init_batch_model_fit
from org.gesis.model.graph import DirectedGraph
from org.gesis.libs.network import get_nodes_metadata
from org.gesis.libs.io import save_csv

def load_summary(output):
    fn = os.path.join(output, 'summary_datasets.csv')
    df_summary = pd.read_csv(fn, index_col=0)
    return df_summary

def get_path(params):
    return os.path.join(params.output, params.dataset, params.model)

def run(params):
    dataset = params.dataset.lower()
    df_summary = load_summary(params.output)
    row = df_summary.query("dataset.str.lower() == @dataset")

    if row.shape[0] > 0:
        row = row.iloc[0]

        ### NETWORK GENERATION
        N = params.N if params.N is not None else row.N
        hmm = row.hmm if params.model == 'DHBA' else row.Emm / (row.Emm + row.EmM) if params.model == 'DH' else 0.5
        hMM = row.hMM if params.model == 'DHBA' else row.EMM / (row.EMM + row.EMm) if params.model == 'DH' else 0.5
        kminm = params.kminm if params.kminm is not None else int(np.ceil(row.kminm * N / row.N))
        kminM = params.kminM if params.kminM is not None else int(np.ceil(row.kminM * N / row.N))
        kmaxm = params.kmaxm if params.kmaxm is not None else int(np.ceil(row.kmaxm * N / row.N))
        kmaxM = params.kmaxM if params.kmaxM is not None else int(np.ceil(row.kmaxM * N / row.N))
        density = params.density if params.density is not None else row.density

        min_density = 1 / (N-1.)
        if density < min_density:
            print('Density is too low: {}'.format(density))
            density = 2 * min_density #min_density + eval("%.0e" % min_density) * 100
            print('New density is: {}'.format(density))

        DG = DirectedGraph(params.model,
                           N=N,
                           density=density,
                           minority_fraction=row.fm,
                           kmin_M=kminM, kmax_M=kmaxM,
                           kmin_m=kminm, kmax_m=kmaxm,
                           gamma_m=row.gammam,
                           gamma_M=row.gammaM,
                           h_mm=hmm,
                           h_MM=hMM,
                           triads_ratio=row.triadsratio,
                           triads_pdf=row.triadspdf)
        DG.create_network()
        DG.G.graph['dataset'] = params.dataset
        DG.info()

        if params.output is not None:
            path = get_path(params)
            DG.save(path, prefix=params.dataset, epoch=params.epoch)

        ### NODE METADATA
        df = get_nodes_metadata(DG.G, num_cores=10)
        fn = DG.fn.replace(".gpickle",'.csv')
        save_csv(df, fn)


if __name__ == "__main__":
    params = init_batch_model_fit()
    run(params)