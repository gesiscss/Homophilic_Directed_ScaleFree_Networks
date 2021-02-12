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
# Main
################################################################

def run(datafn, njobs):
    
    print(datafn, njobs)
    fn = datafn.replace(".gpickle", ".csv")
    if os.path.exists(fn):
        print("{} already exists.".format(fn))
        return
    
    ### 1. load network
    g = io.load_gpickle(datafn)
    
    ### 2. node metadata
    df = graph.get_node_metadata_as_dataframe(g, njobs=njobs)
    df.loc[:,'dataset'] = datafn.split('/')[-1].split('.gpickle')[0].split('-')[0]
    
    ### 3. Storing metadata info into .csv file
    ### ,node,minority,indegree,outdegree,pagerank,circle_of_trust,wtf,dataset
    io.save_csv(df, fn)
    
################################################################
# Main
################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafn", help=".gpickle files.", type=str, required=True)
    parser.add_argument("--njobs", help="parallel jobs", type=int, default=1)
    args = parser.parse_args()
    
    start_time = time.time()
    run(args.datafn, args.njobs)
    print("--- %s seconds ---" % (time.time() - start_time))