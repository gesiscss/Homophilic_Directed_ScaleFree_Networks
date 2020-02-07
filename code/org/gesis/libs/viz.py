import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import powerlaw

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

def plot_degree_distributions_groups(df_metadata_pivot, fn=None):
    g = sns.FacetGrid(df_metadata_pivot.query("metric!='pagerank'"), col='dataset', row='metric',
                      hue="minority",
                      legend_out=False,
                      margin_titles=True,
                      palette="deep",
                      sharex=False,
                      sharey=False,
                      hue_order=[0, 1]
                      )
    g = g.map_dataframe(_plot_empirical_and_powerlaw_fit, 'value')

    for ax in g.axes.ravel():
        ax.legend()

    if fn is not None:
        g.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

def _plot_empirical_and_powerlaw_fit(x, **kwargs):
    from org.gesis.libs.utils import fit_power_law
    '''
    kwargs: data, color, label
    '''
    ax = plt.gca()
    data = kwargs.pop("data")
    data = data[x]
    label = kwargs.pop("label")
    group = {1:'minority', 0:'Majority'}[label]
    color = sns.color_palette("deep", 2)[label]
    ###
    fit = fit_power_law(data)
    ####
    fit.plot_pdf(ax=ax,linewidth=3, color=color, label=group) # empirical
    label = 'Fit (' + '$k_{min}=$' + '{:.0f}; '.format(fit.power_law.xmin) + '$\gamma   =$' + '{:.2f})'.format(fit.power_law.alpha)
    fit.power_law.plot_pdf(ax=ax, linestyle='--', color=color, label=label) # model
    ####
    ax.set_ylabel(u"p(X)")
    ax.set_xlabel(x)