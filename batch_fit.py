################################################################
# System's dependencies
################################################################
import os
import sys
import time
import argparse
import numpy as np
from joblib import Parallel
from joblib import delayed

################################################################
# Local dependencies
################################################################
from org.gesis.lib import io
from org.gesis.model.DH import DH
from org.gesis.model.DBA import DBA
from org.gesis.model.DBAH import DBAH
from org.gesis.model.Random import Random

################################################################
# Constants
################################################################
DATASETS = ['aps','hate','blogs','wikipedia']

################################################################
# Main
################################################################

def run(metadata, model, epochs, njobs, output):
    
    print(metadata, model, njobs)
    
    # 1. Load real network metadata
    df = io.read_csv(metadata, index_col=None)
    
    # 2. Generate #epochs synthetic networks
    _ = Parallel(n_jobs=njobs)(delayed(_create)(df, model, epoch, output, False, epoch) for epoch in np.arange(epochs) )

def _create(df, model, epoch, output, verbose, seed):

    N = 50000 if df.loc[0,'dataset'] == 'github' else \
        100000 if df.loc[0,'dataset'] == 'pokec' else \
        1500 if df.loc[0,'dataset'] == 'blogs' else \
        500 if df.loc[0,'dataset'] == 'seventh' else 3000
        
    N = 500 if model == 'SBM' else N
    hMM = df.loc[0,'hMM'] if model != 'DH' else round(df.loc[0,'EMM']/(df.loc[0,'EMm']+df.loc[0,'EMM']),2)
    hmm = df.loc[0,'hmm'] if model != 'DH' else round(df.loc[0,'Emm']/(df.loc[0,'EmM']+df.loc[0,'Emm']),2)
    
    g = _create_graph(model=model, 
                      N=N, fm=df.loc[0,'fm'], d=df.loc[0,'d'], 
                      plo_M=df.loc[0,'plo_M'], plo_m=df.loc[0,'plo_m'], 
                      hMM=hMM, hmm=hmm, 
                      verbose=verbose, seed=seed)
    
    filename = get_filename(df.loc[0,'dataset'], model, N, df.loc[0,'fm'], df.loc[0,'d'], 
                            df.loc[0,'plo_M'], df.loc[0,'plo_m'], hMM, hmm,
                            epoch)
    fn = os.path.join(output,model,df.loc[0,'dataset'],'{}.gpickle'.format(filename))
    io.save_gpickle(g, fn)

def get_filename(dataset, model, N, fm, d, plo_M, plo_m, hMM=None, hmm=None, epoch=None):
    #aps-DBA-N2000-fm0.4-d0.00106-kminM5-kmaxM17-kminm6-kmaxm10-gM3.0-gm5.0-ID9.gpickle
    return "{}-{}-N{}-fm{}{}{}{}{}{}{}".format(dataset, model, N, 
                                  round(fm,5),
                                  '' if model in ['SBM','Null'] or d is None else '-d{}'.format(round(d,5)),
                                  '' if model in ['SBM','Random','Null'] or plo_M is None else '-ploM{}'.format(round(plo_M,1)),
                                  '' if model in ['SBM','Random','Null'] or plo_m is None else '-plom{}'.format(round(plo_m,1)),
                                  '' if model in ['DBA','Random','Null'] or hMM is None else '-hMM{}'.format(hMM),
                                  '' if model in ['DBA','Random','Null'] or hmm is None else '-hmm{}'.format(hmm),
                                  '' if epoch is None else '-ID{}'.format(epoch),
                                  )

def _create_graph(model, N, fm, d, plo_M, plo_m, hMM, hmm, verbose, seed):

    if model == 'DBAH':
        g = DBAH(N=N, fm=fm, d=d, 
            plo_M=plo_M, plo_m=plo_m, 
            h_MM=hMM, h_mm=hmm, 
            verbose=verbose, seed=seed)
    elif model == 'DBA':
        g = DBA(N=N, fm=fm, d=d, 
            plo_M=plo_M, plo_m=plo_m, 
            verbose=verbose, seed=seed)
    elif model == 'DH':
        g = DH(N=N, fm=fm, d=d, 
            plo_M=plo_M, plo_m=plo_m, 
            h_MM=hMM, h_mm=hmm,
            verbose=verbose, seed=seed)
    elif model == 'Random':
        g = Random(N=N, fm=fm, d=d, 
            verbose=verbose, seed=seed)
    else:
        raise Exception("model does not exist.")
        
    return g
        
################################################################
# Main
################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", help="path/folder where the graph metadata .csv files are.", type=str, required=True)
    parser.add_argument("--model", help="Network model (DBA, DH, DBAH).", type=str, required=True)
    parser.add_argument("--epochs", help="how many synthetic networks to generate.", type=int, default=1)
    parser.add_argument("--njobs", help="parallel jobs.", type=int, default=1)
    parser.add_argument("--output", help="path/folder where to store results.", type=str, default='.')
    args = parser.parse_args()
    
    start_time = time.time()
    run(args.metadata, args.model, args.epochs, args.njobs, args.output)
    print("--- %s seconds ---" % (time.time() - start_time))