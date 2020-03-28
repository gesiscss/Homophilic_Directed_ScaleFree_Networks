import os
import networkx as nx

from org.gesis.libs.arguments import init_batch_node_attributes
from org.gesis.libs.network import get_nodes_metadata_big
from org.gesis.libs.utils import printf
from org.gesis.libs import io


def run(params):
    printf('=== {} ==='.format(params.dataset))
    path = os.path.join(params.root, params.dataset)

    fng = os.path.join(params.root, params.dataset, '{}_attributed_network_anon.gpickle'.format(params.dataset))
    G = nx.read_gpickle(fng)
    printf('computing...')

    get_nodes_metadata_big(G, path, original=True, num_cores=params.numcores)
    del (G)

if __name__ == "__main__":
    params = init_batch_node_attributes()
    run(params)