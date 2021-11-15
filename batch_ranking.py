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
from org.gesis.lib import rank
from org.gesis.lib import paper

################################################################
# Constants
################################################################
DATASETS = ['aps','hate','blogs','wikipedia']

################################################################
# Main
################################################################

def run(path, dataset):
    print(path, dataset)
    
    # create ranking measures for each network
    _ = rank.horizontal_inequalities_parallel(path, dataset)
    
    # create summary file
    root = path.split("/synthetic")[0]
    models = [path.split("/synthetic/")[-1]]
    smooth = 0.05
    df_rank = paper.load_rank_synthetic_all_models(os.path.join(root,'synthetic'), models, smooth, True)
    print(df_rank.head())
    
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