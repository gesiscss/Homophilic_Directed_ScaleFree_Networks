from org.gesis.libs.arguments import init_batch_model_fit
from org.gesis.model.graph import DirectedGraph
import os
import pandas as pd

def load_summary(output):
    fn = os.path.join(output, 'summary_datasets.csv')
    df_summary = pd.read_csv(fn, index_col=0)
    return df_summary

def get_filename(params):
    return os.path.join(params.output, params.dataset, '{}_i{}_x{}.gpickle'.format(params.kind,params.epoch,params.iter))

def run(params):

    dataset = params.dataset.lower()
    df_summary = load_summary(params.output)
    row = df_summary.query("dataset.str.lower() == @dataset")

    if row.shape[0] > 0:
        row = row.iloc[0]

        density = params.N * row.density / row.N

        DG = DirectedGraph(params.kind,
                           N=params.N,
                           kmin=row.kmin,
                           density=density,
                           minority_fraction=row.fm,
                           gamma_m=row.gammam,
                           gamma_M=row.gammaM,
                           h_mm=row.hmm,
                           h_MM=row.hMM,
                           triads_ratio=row.triadsratio,
                           triads_pdf=row.triadspdf)
        DG.create_network()

        DG.G.graph['dataset'] = params.dataset

        DG.info()

        if params.output is not None:
            fn = get_filename(params)
            DG.save(fn)

if __name__ == "__main__":
    params = init_batch_model_fit()
    run(params)