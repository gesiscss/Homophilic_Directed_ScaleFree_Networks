import os
import networkx as nx

from org.gesis.libs.arguments import init_batch_node_attributes
from org.gesis.libs.network import get_nodes_metadata_big
from org.gesis.libs.utils import printf
from org.gesis.libs import io


def run(params):
    printf('=== {} ==='.format(params.dataset))
    fn = os.path.join(params.root, params.dataset, 'nodes_metadata.csv')

    if os.path.exists(fn):
        printf('{} already done!'.format(params.dataset))
    else:
        fng = os.path.join(params.root, params.dataset, '{}_attributed_network_anon.gpickle'.format(params.dataset))
        G = nx.read_gpickle(fng)
        printf('computing...')

        df = get_nodes_metadata_big(G, fn, num_cores=params.numcores)
        df.loc[:, 'dataset'] = params.dataset
        del (G)

        io.save_csv(df, fn)


if __name__ == "__main__":
    params = init_batch_node_attributes()
    run(params)