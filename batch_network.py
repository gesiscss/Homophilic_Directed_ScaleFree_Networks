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
from org.gesis.model.DH import DH
from org.gesis.model.DPA import DPA
from org.gesis.model.DPAH import DPAH
from org.gesis.model.Random import Random

################################################################
# Constants
################################################################
MODELS = ["DPA","DH","DPAH","Random"]

################################################################
# Main
################################################################

def run(model, N, fm, d, ploM, plom, hMM, hmm, epoch, output):
    
    if model not in MODELS:
        raise Exception("model " + model +" does not exist.")
        
    print(model, N, fm, d, ploM, plom, hMM, hmm, output)
    filename = get_filename(model, N, fm, d, ploM, plom, hMM, hmm, epoch)
    fn = os.path.join(output,model,'{}.gpickle'.format(filename))
    
    if os.path.exists(fn):
        print("{} already exist.".format(fn))
        return 
    
    ### Create network
    try:
        g = create_graph(model, N, fm, d, ploM, plom, hMM, hmm, verbose=False, seed=epoch) 
        io.save_gpickle(g, fn)
    except Exception as ex:
        print("1. error at creating the network")
        print(ex)
        
    ### Network metadata
    try:
        EMM, EMm, EmM, Emm = graph.get_edge_type_counts(g,True)
        pliM, plim = graph.get_indegree_powerlaw_exponents(g)
        t1 = "model,N,fm,d,plo_M,plo_m,pli_M,pli_m,EMM,EMm,EmM,Emm,hMM,hmm"
        t2 = ",".join([model, str(N), str(fm), str(d), str(ploM), str(plom), str(pliM.alpha), str(plim.alpha), 
                        str(EMM), str(EMm), str(EmM), str(Emm), str(hMM), str(hmm)])
        fn = os.path.join(output,model,'{}_netmeta.csv'.format(filename))
        io.save_text("{}\n{}".format(t1,t2), fn)
    except Exception as ex:
        print("2. error at creating the network metadata")
        print(ex)
        
    ### Node metadata
    try:
        df = graph.get_node_metadata_as_dataframe(g)
        fn = os.path.join(output,model,'{}.csv'.format(filename))
        io.save_csv(df, fn)
    except Exception as ex:
        print("3. error at creating the node metadata")
        print(ex)
        
def create_graph(model, N, fm, d, plo_M, plo_m, hMM, hmm, verbose, seed):

    if model == 'DPAH':
        g = DPAH(N=N, fm=fm, d=d, 
            plo_M=plo_M, plo_m=plo_m, 
            h_MM=hMM, h_mm=hmm, 
            verbose=verbose, seed=seed)
    elif model == 'DPA':
        g = DPA(N=N, fm=fm, d=d, 
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

def get_filename(model, N, fm, d, plo_M, plo_m, hMM=None, hmm=None, epoch=None):
    #DBA-N2000-fm0.4-d0.00106-kminM5-kmaxM17-kminm6-kmaxm10-gM3.0-gm5.0-ID9.gpickle
    return "{}-N{}-fm{}{}{}{}{}{}{}".format(model, N, 
                                             round(fm,1), 
                                             '' if model in ['SBM','Null'] else '-d{}'.format(round(d,5)), 
                                             '' if model in ['Random','SBM','Null'] else '-ploM{}'.format(round(plo_M,1)), 
                                             '' if model in ['Random','SBM','Null'] else '-plom{}'.format(round(plo_m,1)), 
                                             '' if model in ['DPA','DUniform','Null'] or hMM is None else '-hMM{}'.format(hMM),
                                             '' if model in ['DPA','DUniform','Null'] or hmm is None else '-hmm{}'.format(hmm),
                                             '' if epoch is None else '-ID{}'.format(epoch),
                                             )


################################################################
# Main
################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="network generation model (DPA, DH, DPAH)", type=str, required=True)
    parser.add_argument("--N", help="number of nodes", type=int, required=True)
    parser.add_argument("--fm", help="fraction of minorities", type=float, required=True)
    parser.add_argument("--d", help="density", type=float, default=None)
    parser.add_argument("--ploM", help="power-law out-degree distribution Majority", type=float, default=None)
    parser.add_argument("--plom", help="power-law out-degree distribution minority", type=float, default=None)
    parser.add_argument("--hMM", help="homophily between Majorities", type=float, default=0.5)
    parser.add_argument("--hmm", help="homophily between minorities", type=float, default=0.5)
    parser.add_argument("--epoch", help="ID (1,2,3,...)", type=int, default=0)
    parser.add_argument("--output", help="path/folder where to store csv file", type=str, default='.')
    args = parser.parse_args()
    
    start_time = time.time()
    run(args.model, args.N, args.fm, args.d, args.ploM, args.plom, args.hMM, args.hmm, args.epoch, args.output)
    print("--- %s seconds ---" % (time.time() - start_time))