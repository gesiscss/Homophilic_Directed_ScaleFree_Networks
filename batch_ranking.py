################################################################
# System's dependencies
################################################################
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
import time
import argparse

################################################################
# Local dependencies
################################################################
from org.gesis.lib import rank
from org.gesis.lib import paper

################################################################
# Constants
################################################################
DATASETS = ['aps','hate','blogs','wikipedia']
BETA = 0.05 #beta
MODELS = ["Random","DPA","DH","DPAH"]
ALLKIND = ['empirical'] + MODELS

################################################################
# Main
################################################################

def run(path, dataset):
    print(path, dataset)
    
    # create ranking measures for each network
    _ = rank.horizontal_inequalities_parallel(path, dataset)
    
    # create summary file
    root = path.split('/')[0] # resuts
    kind = path.split('/')[1] # synthetic, fit, empirical
    # @TODO: make it work for fit and empirical nets
    if kind == 'synthetic':
        # only works for synthetic 
        models = [path.split('/')[2]]
        print(root, kind, models)
        df_rank = paper.load_rank_synthetic_all_models(os.path.join(root,kind), models, BETA, True)
        print(df_rank.head())
        #df_rank = paper.load_rank_all_models(os.path.join(root,'fit'), models, SMOOTH, DATASETS)
        #df_rank = paper.load_rank(os.path.join(root,'empirical'), df_network_metadata_empirical, SMOOTH, DATASETS, ALLKIND)
        print(df_rank.shape)
        
    
################################################################
# Main
################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="directory where <dataset>/*.csv files are.", type=str, required=True)
    parser.add_argument("--dataset", help="datasets ({}).".format(",".join(DATASETS)), 
                        type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    run(args.path, args.dataset)
    print("--- %s seconds ---" % (time.time() - start_time))