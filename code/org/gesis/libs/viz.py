import matplotlib.pyplot as plt
import networkx as nx

def plot_simple_network(G):
    plt.close()
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G,
            pos=pos,
            node_size=[G.degree(n)*10 for n in G.nodes()],
            node_shape='s',
            alpha=1,
            node_color=[n[1][G.graph['class']] for n in G.nodes(data=True)],
            with_labels=False)
    plt.show()
    plt.close()