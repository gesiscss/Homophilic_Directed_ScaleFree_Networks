################################################################
# System's dependencies
################################################################
import os
import sys
import argparse
import time

################################################################
# Local dependencies
################################################################
from org.gesis.lib import rank

################################################################
# Constants
################################################################
DATASETS = ['aps','apsgender3','apsgender8','blogs','github','hate','pokec','seventh','wikipedia']

################################################################
# Main
################################################################

def run(path, dataset):
    print(path, dataset)
    _ = rank.horizontal_inequalities_parallel(path, dataset)
    
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