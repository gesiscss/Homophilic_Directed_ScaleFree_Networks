import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib import rc
import sympy
import seaborn as sns

import networkx as nx
import numpy as np

############################################################################################################
# Latex compatible
############################################################################################################

def plot_setup(latex=True):
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set_style("white")

    if latex:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        rc('text', usetex=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['text.usetex'] = True
        #mpl.rcParams['text.latex.unicode'] = True

        lw = 0.8
        sns.set_context("paper", rc={"lines.linewidth": lw})
    else:
        sns.set_context('paper', font_scale=1.2)

def latex_compatible_text(txt):
    return sympy.latex(sympy.sympify(txt)).replace("_", "\_")

def latex_compatible_dataframe(df, latex=True):
    tmp = df.copy()
    if latex:
        cols = {c:c if c=="N" or c.startswith("MSE") else sympy.latex(sympy.sympify(c)).replace("_","\_") for c in tmp.columns}
        if 'sampling' in tmp.columns:
            tmp.sampling = tmp.apply(lambda row: row.sampling.replace('_', '\_'), axis=1)
    else:
        cols = {c:c for c in tmp.columns}
    tmp.rename(columns=cols, inplace=True)
    return tmp, cols

def unlatexfyme(text):
    return text.replace("_", "").replace("\\", "").replace('{', '').replace('}', '').replace('$','').strip()

############################################################################################################
# Networks
############################################################################################################

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

############################################################################################################
# Distributions
############################################################################################################

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


############################################################################################################
# Ranking
############################################################################################################

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

############################################################################################################
# Gini and Lorenz curve
############################################################################################################

def _lorenzplot_empirical_fgmap(x, **kwargs):
    from org.gesis.libs.utils import gini
    from org.gesis.libs.utils import lorenz_curve

    ax = plt.gca()
    data = kwargs.pop("data")
    color = kwargs.pop("color")

    if data.size == 0:
        return

    X = np.sort(data[x].astype(np.float).values)
    X_lorenz = lorenz_curve(X)
    gc = gini(X)

    ax.plot(np.arange(X_lorenz.size) / (X_lorenz.size - 1), X_lorenz, color=color)
            # marker='.', linewidth=1, markersize=2, linestyle="-")

    if data.minority.nunique() > 1:
        ax.text(s="Gini: {:.2f}".format(gc), x=0.1, y=0.8)
    else:
        if data.minority.unique()[0]:
            ax.text(s="Gini min:" + "{:.2f}".format(gc), x=0.1, y=0.7, color=color)
        else:
            ax.text(s="Gini maj:" + "{:.2f}".format(gc), x=0.1, y=0.8, color=color)

    ## line plot of equality
    ax.plot([0, 1], [0, 1], color='k')

def _lorenzplot_fit_fgmap(x, **kwargs):
    from org.gesis.libs.utils import gini
    from org.gesis.libs.utils import lorenz_curve
    import pandas as pd

    ax = plt.gca()
    data = kwargs.pop("data")
    color = kwargs.pop("color")

    if data.size == 0:
        return

    gc = []
    df = None
    for epoch in data['epoch'].unique():
        X = np.sort(data.query("epoch==@epoch")[x].astype(np.float).values)
        X_lorenz = lorenz_curve(X)

        if df is None:
            df = pd.DataFrame(index=np.arange(X_lorenz.size) / (X_lorenz.size - 1))

        df.loc[:,str(epoch)] = X_lorenz
        gc.append(gini(X))

    ax.plot(df.index, df.mean(axis=1).values, color=color)
    gc = np.mean(gc)

    if data.minority.nunique() > 1:
        ax.text(s="Gini: {:.2f}".format(gc), x=0.1, y=0.8)
    else:
        if data.minority.unique()[0]:
            ax.text(s="Gini min:" + "{:.2f}".format(gc), x=0.1, y=0.7, color=color)
        else:
            ax.text(s="Gini maj:" + "{:.2f}".format(gc), x=0.1, y=0.8, color=color)

    ## line plot of equality
    ax.plot([0, 1], [0, 1], color='k')

def plot_lorenz_curve(df, metric, row=None, hue=None, fn=None):
    plt.close()

    ### plot
    fg = sns.FacetGrid(data=df, col='dataset', row=row, hue=hue, hue_order=[0, 1], dropna=True, margin_titles=True)

    if row is None:
        fg = fg.map_dataframe(_lorenzplot_empirical_fgmap, metric)
    else:
        fg = fg.map_dataframe(_lorenzplot_fit_fgmap, metric)

    ### ylabel
    if row is None:
        fg.axes[0,0].set_ylabel('% of total {}'.format(metric))
    else:
        nr = df[row].nunique()
        if nr % 2 != 0:
            fg.axes[int(nr/2), 0].set_ylabel('mean % of total {}'.format(metric))
        else:
            for ax in fg.axes.flatten():
                ax.set_ylabel('mean % of total {}'.format(metric))
            #fg.axes[int(nr / 2)-1, 0].set_ylabel('% of total {}'.format(metric))
            #fg.axes[int(nr / 2), 0].set_ylabel('% of total {}'.format(metric))

    ### xlabel
    for c in np.arange(fg.axes.shape[1]):
        fg.axes[-1,c].set_xlabel('% of nodes')

    ### savefig
    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()