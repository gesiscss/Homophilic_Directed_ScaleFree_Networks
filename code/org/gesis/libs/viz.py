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

def plot_empirical_rankings(df_rank, df_summary, sharey=True, fn=None):
    tmp = df_rank.query("kind=='empirical'").copy()

    #### this code is temporal, until pokec and github (empirical) copmute cot and wtf
    ds = ['pokec', 'github']
    for metric in ['circle_of_trust', 'wtf']:
        tmp2 = df_rank.query("dataset.str.lower() in @ds & metric=='pagerank' & kind == 'empirical' ").copy()
        tmp2.loc[:, 'fmt'] = 0
        tmp2.loc[:, 'metric'] = metric
        tmp = tmp.append(tmp2, ignore_index=True)
        del (tmp2)
    #### end.

    tmp['rank'] = tmp['rank'].astype(int)
    tmp.sort_values('dataset', inplace=True)

    metric_order = ['pagerank', 'circle_of_trust', 'wtf']
    fg = sns.catplot(data=tmp,
                     col='dataset',
                     hue='metric', hue_order=metric_order,
                     x='rank', y='fmt',
                     kind='point',
                     sharey=sharey,
                     palette='Set2',
                     height=3, aspect=1, )

    for ax in fg.axes.flatten():
        dataset = ax.get_title().split(" = ")[-1].lower()
        tmp = df_summary.query("dataset.str.lower()==@dataset.lower()").copy()

        ### baseline, actual minority fraction
        minority_fraction = tmp.fm.mean()
        ax.axhline(minority_fraction, lw=1, c='black', ls='--')

        ### Values of homophily and outdegree distr. exponent
        x=2
        y= 0.05 if dataset == 'aps' else 0.4 if dataset in ['github','wikipedia'] else 0.2 if dataset == 'pokec' else 0

        s = r'$h_{MM}, h_{mm}=(' + str(round(tmp.hMM.mean(), 2)) + ',' + str(round(tmp.hmm.mean(), 2)) + ')' + \
            '$ \n $\gamma_{M},\gamma_{m}=(' + str(round(tmp.gammaM.mean(), 2)) + ',' + str(
            round(tmp.gammam.mean(), 2)) + ')' + \
            '$'
        ax.text(x=x, y=y, s=s)

        ### xticklabels
        ax.set_xticklabels(label if i in [1, 5, 9] else '' for i, label in enumerate(ax.get_xticklabels()))
        ax.set_xlabel('k%')
        ax.set_ylabel('Fraction of minorities in Top-k%')

    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

def plot_model_fit(df_rank, df_summary, metric="pagerank", sharey=True, fn=None):


    tmp = df_rank.query("metric==@metric").copy()

    #### this code is temporal, until pokec and github (empirical) copmute cot and wtf
    if metric != 'pagerank':
        ds = ['pokec','github']
        tmp2 = df_rank.query("dataset.str.lower() in @ds & metric=='pagerank' & kind == 'empirical' ").copy()
        tmp2.loc[:,'fmt'] = 0
        tmp = tmp.append(tmp2, ignore_index=True)
        del (tmp2)
    #### end.

    tmp['rank'] = tmp['rank'].astype(int)
    tmp.sort_values('dataset', inplace=True)

    kind_order = ['empirical','DH','DBA','DHBA']
    fg = sns.catplot(data=tmp,
                     col='dataset',
                     hue='kind', hue_order=kind_order,
                     x='rank', y='fmt',
                     kind='point',
                     sharey=sharey,
                     height=3, aspect=1, )

    for ax in fg.axes.flatten():
        dataset = ax.get_title().split(" = ")[-1]
        tmp = df_summary.query("dataset.str.lower()==@dataset.lower()").copy()

        ### baseline, actual minority fraction
        minority_fraction = tmp.fm.mean()
        ax.axhline(minority_fraction, lw=1, c='black', ls='--')

        ### Values of homophily and outdegree distr. exponent
        if metric in ['pagerank','circle_of_trust']:
            x = 2
            y = 0.1 if dataset.lower() in ['aps', 'pokec'] else 0.3
        else:
            x = 2
            y = 0.7 if dataset.lower() in ['aps', 'pokec'] else 0.4

        s = r'$h_{MM}, h_{mm}=(' + str(round(tmp.hMM.mean(), 2)) + ',' + str(round(tmp.hmm.mean(), 2)) + ')' + \
            '$ \n $\gamma_{M},\gamma_{m}=(' + str(round(tmp.gammaM.mean(), 2)) + ',' + str(
            round(tmp.gammam.mean(), 2)) + ')' + \
            '$'
        ax.text(x=x, y=y, s=s)

        ### xticklabels
        ax.set_xticklabels(label if i in [1,5,9] else '' for i,label in enumerate(ax.get_xticklabels()))
        ax.set_xlabel('k%')
        ax.set_ylabel('Fraction of minorities in Top-k%')

    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))