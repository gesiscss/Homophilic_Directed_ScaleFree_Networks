################################################################
# System's dependencies
################################################################
import os
import sys
import time
import argparse

################################################################
# Local dependencies
################################################################
from org.gesis.lib import io
from org.gesis.lib import graph
from org.gesis.lib import homophily

################################################################
# Constants
################################################################
DATASETS = ['aps','blogs','hate','seventh','wikipedia']

################################################################
# Main
################################################################

def run(datapath, dataset, steps, njobs, output):
    
    if dataset not in DATASETS:
        raise Exception("dataset " + dataset +" does not exist.")
        
    print(dataset, steps, njobs)
    g = graph.get_graph(datapath, dataset)
    N, fm, d, plo_M, plo_m, pli_M, pli_m, EMM, EMm, EmM, Emm, hMM, hmm, _N, _d, _mindiff = homophily.get_metadata(g, steps, 
                                                                                           njobs=njobs, verbose=True, seed=None)

    print("N:{}".format(N))
    print("fm:{}".format(fm))
    print("d:{}".format(d))
    print("plo_M:{}".format(plo_M))
    print("plo_m:{}".format(plo_m))
    print("pli_M:{}".format(pli_M))
    print("pli_m:{}".format(pli_m))
    print("EMM:{}".format(EMM))
    print("EMm:{}".format(EMm))
    print("EmM:{}".format(EmM))
    print("Emm:{}".format(Emm))
    print("hMM:{}".format(hMM))
    print("hmm:{}".format(hmm))
    print("_N:{}".format(_N))
    print("_d:{}".format(_d))
    print("_mindiff:{}".format(_mindiff))
    
    ### Storing metadata info into .csv file
    t1 = "dataset,N,fm,d,plo_M,plo_m,pli_M,pli_m,EMM,EMm,EmM,Emm,hMM,hmm,_N,_d,_mindiff"
    t2 = ",".join([dataset, str(N), str(fm), str(d), str(plo_M), str(plo_m), str(pli_M), str(pli_m), 
                    str(EMM), str(EMm), str(EmM), str(Emm), str(hMM), str(hmm), str(_N), str(_d), str(_mindiff)])
    path = os.path.join(output,dataset,"network_metadata.csv")
    io.save_text("{}\n{}".format(t1,t2), path)
    
################################################################
# Main
################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help=",".join(DATASETS), type=str, required=True)
    parser.add_argument("--steps", help="decimals (eg. 0.01, 0.05) to compute homophily", type=float, required=True)
    parser.add_argument("--njobs", help="parallel jobs", type=int, default=1)
    parser.add_argument("--datapath", help="path/folder where the .gpickle files are.", type=str, required=True)
    parser.add_argument("--output", help="path/folder where to store csv file", type=str, default='.')
    args = parser.parse_args()
    
    start_time = time.time()
    run(args.datapath, args.dataset, args.steps, args.njobs, args.output)
    print("--- %s seconds ---" % (time.time() - start_time))