import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib import rc
import sympy
import seaborn as sns
import networkx as nx
import numpy as np


from org.gesis.libs.ranking import VALID_METRICS
from org.gesis.libs.utils import gini
from org.gesis.libs.utils import fit_power_law
from palettable.colorbrewer.diverging import BrBG_11
from palettable.colorbrewer.diverging import BrBG_5

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

    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    if fn is not None:
        g.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

def _plot_empirical_and_powerlaw_fit(x, **kwargs):
    '''
    kwargs: data, color, label
    '''
    ax = plt.gca()
    data = kwargs.pop("data")
    data = data[x].values.astype(np.float)
    label = kwargs.pop("label")
    group = {1:'minority', 0:'Majority'}[label]
    color = sns.color_palette("deep", 2)[label]
    ###
    fit = fit_power_law(data)
    g = round(gini(data), 2)
    ####
    fit.plot_pdf(ax=ax,linewidth=3, color=color, label='{} (Gini: {})'.format(group,g)) # empirical
    legend = 'Fit (' + '$k_{min}=$' + '{:.0f}; '.format(fit.power_law.xmin) + \
             '$\gamma   =$' + '{:.2f})'.format(fit.power_law.alpha)
    fit.power_law.plot_pdf(ax=ax, linestyle='--', color=color, label=legend) # model
    ####
    ax.set_ylabel(u"p(X)")
    ax.set_xlabel(x)


############################################################################################################
# Ranking
############################################################################################################

def plot_vh_inequalities_empirical(df_rank, fn=None):

    ### only man data points
    tmp = df_rank.query("rank==5").copy()
    tmp.drop(columns=['rank', 'fmt'], inplace=True)

    ### main plot
    fg = sns.catplot(data=tmp,
                     x='gini', y='mae',
                     height=2.0,aspect=0.9,
                     hue='metric', col='dataset')
    [plt.setp(ax.texts, text="") for ax in fg.axes.flat]
    fg.set_titles(row_template='{row_name}', col_template='{col_name}')

    ### labels and xticks
    for ax in fg.axes.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")

        # xticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_minor_locator(plt.MaxNLocator(10))

        # xticklabels
        xtls = ax.get_xticklabels()
        ax.set_xticklabels([round(float(xtl.get_text()), 2) for i, xtl in enumerate(xtls)], rotation=0)

    ### ylabel
    ylabel = 'MAE of \nfraction of minorities in top-k%'
    if fg.axes.shape[0] % 2 != 0:
        fg.axes[int(fg.axes.shape[0] / 2), 0].set_ylabel(ylabel)
    else:
        fg.axes[int(fg.axes.shape[0] / 2), 0].text(-50, 0.28, ylabel, {'ha': 'center', 'va': 'center'}, rotation=90)

    ### xlabel
    xlabel = 'Gini coefficient'
    if fg.axes.shape[1] % 2 != 0:
        fg.axes[-1, int(fg.axes.shape[1] / 2)].set_xlabel(xlabel)
    else:
        fg.axes[-1, int(fg.axes.shape[1] / 2)].text(0, -0.05, xlabel, {'ha': 'center', 'va': 'center'}, rotation=0)

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))


def plot_vh_inequalities_fit(df_rank, group=False, fn=None):

    ### only main data points
    metrics = ['pagerank', 'wtf']
    tmp = df_rank.query("rank==5 & kind in ['empirical','DHBA'] & metric in @metrics").copy()
    tmp.drop(columns=['rank', 'fmt'], inplace=True)

    if group:
        tmp = tmp.groupby(['dataset','kind','metric']).mean().reset_index()

    tmp.kind = tmp.kind.astype("category")
    tmp.kind.cat.set_categories(['empirical','DHBA'], inplace=True)
    tmp.sort_values(['kind','dataset','metric'], inplace=True)

    ### main plot
    nrows = tmp.metric.nunique()
    ncols = tmp.dataset.nunique()
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, 5), sharey=True, sharex=True)

    ### subplots
    colors = sns.color_palette("tab20")
    for col, dataset in enumerate(tmp.dataset.unique()):

        axes[0, col].set_title(dataset)

        for row, metric in enumerate(tmp.metric.unique()):

            if col == ncols-1:
                axes[row, col].text(s=metric,
                                    x=1.08 if not group else 1.06,
                                    y=(tmp.mae.max()/2.)+(len(metric)*0.005/2.), rotation=-90)

            for hue, kind in enumerate(tmp.kind.unique()):
                data = tmp.query("dataset==@dataset & metric==@metric & kind==@kind").copy()
                axes[row,col].scatter(x=data.gini.values, y=data.mae.values, label=kind, color=colors[hue], marker='x' if kind!='empirical' else 'o')

    ### legend
    plt.legend()

    ### ylabel
    ylabel = 'MAE of \nfraction of minorities in top-k%'
    if nrows % 2 != 0:
        axes[int(axes.shape[0]/2), 0].set_ylabel(ylabel)
    else:
        axes[int(axes.shape[0] / 2), 0].text(0.16 if not group else 0.17,
                                             0.26 if not group else 0.16,
                                             ylabel, {'ha': 'center', 'va': 'center'}, rotation=90)

    ### xlabel
    xlabel = 'Gini coefficient'
    if ncols % 2 != 0:
        axes[-1, int(axes.shape[1]/2)].set_xlabel(xlabel)
    else:
        axes[-1, int(axes.shape[1] / 2)].text(0.35,
                                              -0.08 if not group else -0.04,
                                              xlabel, {'ha': 'center', 'va': 'center'}, rotation=0)

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))


def plot_vh_inequalities_synthetic(df_rank, metric='pagerank', sym=True, fn=None):

    ### validation
    if metric not in VALID_METRICS:
        raise ValueError('metric {} is not valid.'.format(metric))

    ### only main data points
    tmp = df_rank.query("rank==5 & metric==@metric").copy()
    tmp.drop(columns=['rank', 'fmt'], inplace=True)

    if sym:
        tmp = tmp.query("hmm == hMM").copy()
        colors = BrBG_11.mpl_colors
    else:
        hm = [0.2,0.8]
        hM = [0.0, 0.2, 0.5, 0.8, 1.0]
        tmp = tmp.query("hmm in @hm and hMM in @hM").copy()
        colors = BrBG_5.mpl_colors
    colors[int(len(colors)/2)] = 'lightgrey'

    tmp.sort_values(['hmm','hMM','fm'], inplace=True)

    ### main plot
    nrows = 1 if sym else tmp.hmm.nunique()
    ncols = tmp.fm.nunique()
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, 5 if not sym else 2.5), sharey=True, sharex=True)

    ### subplots
    for col, fm in enumerate(tmp.fm.unique()):

        if sym:
            axes[col].set_title("fm={}".format(fm))
            for hue, hMM in enumerate(tmp.hMM.unique()):
                data = tmp.query("fm==@fm & hmm==hMM & hMM==@hMM").copy()
                axes[col].scatter(x=data.gini.values, y=data.mae.values, label=hMM, color=colors[hue], marker='x')
            axes[0].legend(loc='lower left', title='homophily',
                           bbox_to_anchor=(-0.04, 1.12, ncols*1.075, 0.2), mode='expand',
                           ncol=tmp.hMM.nunique(), handletextpad=0.05, frameon=False)
            # plt.legend(loc='center left', title='h', bbox_to_anchor=(1, 0.5))
        else:
            axes[0, col].set_title("fm={}".format(fm))
            for row, hmm in enumerate(tmp.hmm.unique()):
                if col == ncols - 1:
                    s = 'hmm={}'.format(hmm)
                    axes[row, col].text(s=s,
                                        x=tmp.gini.max() + 0.035,
                                        y=(tmp.mae.max() / 2.) + (len(s) * 0.0065 / 2.), rotation=-90)

                for hue, hMM in enumerate(tmp.hMM.unique()):
                    data = tmp.query("fm==@fm & hmm==@hmm & hMM==@hMM").copy()
                    axes[row, col].scatter(x=data.gini.values, y=data.mae.values, label=hMM, color=colors[hue], marker='x')

            axes[0,1].legend(loc='lower left', title='hMM',
                             bbox_to_anchor=(-0.26, 1.12, 1.5, 0.2), mode='expand',
                             ncol=tmp.hMM.nunique(), handletextpad=0.05, frameon=False)
            #axes[0, 2].legend(loc='lower left', title='hMM'

    ### ylabel
    ylabel = 'MAE of \nfraction of minorities in top-k%'
    row = int(nrows / 2)
    ax = axes[row,0] if not sym else axes[0]
    if nrows % 2 != 0:
        ax.set_ylabel(ylabel)
    else:
        ax.text(0.64 if not sym else 100,
                0.65 if not sym else 0.65,
                ylabel, {'ha': 'center', 'va': 'center'}, rotation=90)

    ### xlabel
    xlabel = 'Gini coefficient'
    col = int(ncols / 2)
    ax = axes[-1,col] if not sym else axes[col]
    if ncols % 2 != 0:
        ax.set_xlabel(xlabel)
    else:
        ax.text(0.35,
               -0.08 if not sym else -0.05,
                xlabel, {'ha': 'center', 'va': 'center'}, rotation=0)

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))





def plot_synthetic_rankings(df_rank, sym=True, fn=None):
    col = 'fm'
    row = 'hmm'

    ### Type of homophily: symmetric or not
    if sym:
        tmp = df_rank.query("hmm == hMM").copy()
        fg = sns.catplot(data=tmp,
                         col=col,
                         hue='hMM',
                         x='rank', y='fmt',
                         kind='point',
                         sharey=True,
                         height=3, aspect=1,
                         legend=True,
                         legend_out=True,
                         palette=BrBG_11.mpl_colors
                         )
    else:
        hs = [0.0, 0.2, 0.5, 0.8, 1.0]
        tmp = df_rank.query("hmm in @hs and hMM in @hs").copy()
        fg = sns.catplot(data=tmp,
                         col=col,
                         row=row,
                         hue='hMM',
                         margin_titles=True,
                         x='rank', y='fmt',
                         kind='point',
                         sharey=True,
                         height=2.5, aspect=1,
                         legend=True,
                         legend_out=True,
                         palette=BrBG_5.mpl_colors)

    ### baseline: fm
    ncol = df_rank[col].nunique()
    nrow = 1 if sym else df_rank[row].nunique()
    for i, ax in enumerate(fg.axes.flatten()):
        # labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # xticks
        ax.set_xticklabels([int(float(xtl.get_text())) if int(float(xtl.get_text())) in [10, 50, 90] else '' for xtl in ax.get_xticklabels()], rotation=0)

        # baseline
        try:
            r = int(i / ncol)
            c = i - (r * ncol)
            fm = float(fg.axes[0,c].get_title().replace("fm = ","").replace(" ",""))
            ax.axhline(fm, c='grey', ls='--', lw=2.0, zorder=1000)
        except:
            pass

    ### labels
    fg.axes[-1,1].set_xlabel('Rank k%')
    fg.axes[int(fg.axes.shape[0]/2),0].set_ylabel('Fraction of minorities in Top-k%')

    ### legend
    if sym:
        fg._legend.set_title("h")

    ### Save fig
    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))


def plot_empirical_rankings(df_rank, df_summary, hue='metric', sharey=True, df_metadata=None, fn=None):
    tmp = df_rank.query("kind=='empirical'").copy()

    #### @todo: this code is temporal, until pokec  (empirical) copmute cot and wtf
    if df_rank.metric.nunique() > 1:
        ds = ['pokec']
        for metric in ['wtf']:
            tmp2 = df_rank.query("dataset.str.lower() in @ds & metric=='pagerank' & kind == 'empirical' ").copy()
            tmp2.loc[:, 'fmt'] = 0
            tmp2.loc[:, 'metric'] = metric
            tmp = tmp.append(tmp2, ignore_index=True)
            del (tmp2)
    #### end.

    tmp['rank'] = tmp['rank'].astype(int)
    tmp.sort_values('dataset', inplace=True)

    if hue is None:
        metric_order = None
    else:
        metric_order = ['pagerank', 'wtf']

    fg = sns.catplot(data=tmp,
                     col='dataset',
                     hue=hue, hue_order=metric_order,
                     x='rank', y='fmt',
                     kind='point',
                     sharey=sharey,
                     palette=None if hue is None else 'Set2',
                     height=3, aspect=1,
                     legend=df_rank.metric.nunique() > 1,
                     legend_out=True,
                     )

    for col, ax in enumerate(fg.axes.flatten()):
        dataset = ax.get_title().split(" = ")[-1].lower()
        tmp = df_summary.query("dataset.str.lower()==@dataset.lower()").copy()

        if hue is None:
            ax.text(s="metric: {}".format(df_rank.query("dataset.str.lower()==@dataset").metric.unique()[0]), x=2, y=0.6)

        ### baseline, actual minority fraction
        minority_fraction = tmp.fm.mean()
        ax.axhline(minority_fraction, lw=1, c='black', ls='--')

        ### Values of homophily and outdegree distr. exponent
        x=2
        y= 0 if col == 0 else 0.4 if col in [1,2,3,5] else 0.1 if col == 4 else 0

        if df_rank.metric.nunique() > 1:
            s = r' $h_{MM}, h_{mm}=(' + str(round(tmp.hMM.mean(), 2)) + ',' + str(round(tmp.hmm.mean(), 2)) + ')' + \
                '$ \n $\gamma_{M},\gamma_{m}=(' + str(round(tmp.gammaM.mean(), 2)) + ',' + str(round(tmp.gammam.mean(), 2)) + ')' + \
                '$ \n $fm=' + str(minority_fraction) + \
                '$'
        elif df_metadata is not None:
            from org.gesis.libs.utils import gini
            X = np.sort(df_metadata.query("dataset.str.lower() == @dataset.lower()")[df_rank.metric.unique()[0]].astype(np.float).values)
            gc = gini(X)
            s = r' $gini=' + str(round(gc,2)) + \
                '$ \n $h_{MM}, h_{mm}=(' + str(round(tmp.hMM.mean(), 2)) + ',' + str(round(tmp.hmm.mean(), 2)) + ')' + \
                '$ \n $\gamma_{M},\gamma_{m}=(' + str(round(tmp.gammaM.mean(), 2)) + ',' + str(round(tmp.gammam.mean(), 2)) + ')' + \
                '$ \n $fm=' + str(minority_fraction) + \
                '$'

        ax.text(x=x, y=y, s=s)

        ### xticklabels
        ax.set_xticklabels(label if i in [1, 5, 9] else '' for i, label in enumerate(ax.get_xticklabels()))
        ax.set_xlabel('Rank k%')

    fg.axes[0,0].set_ylabel('Fraction of minorities in Top-k%')

    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

def plot_model_fit(df_rank, df_summary, model=None, metric="pagerank", sharey=True, fn=None):

    tmp = df_rank.query("metric==@metric").copy()

    if model is not None:
        tmp = tmp.query("kind in ['empirical',@model]")

    #### @todo: this code is temporal, until pokec (empirical) copmute cot and wtf
    if metric != 'pagerank':
        ds = ['pokec']
        tmp2 = df_rank.query("dataset.str.lower() in @ds & metric=='pagerank' & kind == 'empirical' ").copy()
        tmp2.loc[:,'fmt'] = 0
        tmp = tmp.append(tmp2, ignore_index=True)
        del (tmp2)
    #### end.

    tmp['rank'] = tmp['rank'].astype(int)
    tmp.sort_values('dataset', inplace=True)

    kind_order = []
    for ko in ['empirical','DH','DBA','DHBA']:
        if ko in tmp.kind.unique():
            kind_order.append(ko)

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
        ax.set_xlabel('Rank k%')
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

def plot_lorenz_curve(df, metric, col='dataset', row=None, groups=False, fn=None):
    plt.close()

    ### plot
    fg = sns.FacetGrid(data=df, col=col, row=row, hue='minority' if groups else None, hue_order=[0,1], dropna=True, margin_titles=True)

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


def _ranking_cdf(x, y, hue, **kwargs):
    import pandas as pd

    ax = plt.gca()
    data = kwargs.pop("data")
    color = kwargs.pop("color")
    df_summary = kwargs.pop("df_summary")

    if data.size == 0:
        return

    dataset = data.dataset.unique()[0]
    tmp = data.sort_values("rank")
    cdf = tmp.fmt.value_counts().sort_index()
    cdf = cdf.cumsum()
    cdf = cdf / cdf.max()
    fm = df_summary.query("dataset==@dataset").fm.iloc[0]

    if cdf.index[-1] < fm:
        cdf.drop(labels=[cdf.index[-1]], inplace=True)
        cdf = cdf.append( pd.Series([1], index=[fm]) )
    ax = cdf.plot(ax=ax, color=color, legend=True, label=data.metric.unique()[0])

    if data.metric.unique()[0] == 'pagerank':
        sd = np.std(tmp.fmt) / 10

        ax.text(s='fm={}'.format(fm),
                # @todo: remove the pokec condition once there is node_metada.csv
                x=0.5 if dataset.lower()=='aps' else 0.05 if dataset.lower()=='apsgender3' else 0.07 if dataset.lower()=='apsgender8' else 0.03 if dataset.lower()=='github' else 0.01 if dataset.lower()=='wikipedia' else 0.43,
                y=0.5 if dataset.lower()=='aps' else 0.8,)
        ax.axvline(fm, c='grey', lw=1, ls='--')

        _,_ = ax.set_xlim(cdf.index[0]-sd,cdf.index[-1]+sd)

def plot_cdf_ranking(df, df_summary, col='dataset', hue='metric', hue_order=['pagerank','circle_of_trust','wtf'], fn=None):
    plt.close()

    ### plot
    fg = sns.FacetGrid(data=df, col=col, hue=hue, hue_order=hue_order,
                       dropna=True, margin_titles=True,
                       sharex=False, sharey=True,
                       height=2.1, aspect=1.2,
                       )

    fg = fg.map_dataframe(_ranking_cdf, None, None, hue, df_summary=df_summary)
    fg.add_legend()

    fg.axes[0,0].set_ylabel("CDF")
    for ax in fg.axes.flatten():
        ax.set_xlabel('Fraction of minorities in Top-k%')
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### savefig
    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()

def plot_vh_inequalities_per_dataset_and_metric(df_metadata, df_rank, df_summary, fn=None):
    from org.gesis.libs.utils import gini
    from org.gesis.libs.utils import lorenz_curve

    plt.close()
    metrics = ['pagerank','wtf']
    colors = sns.xkcd_palette(["medium green", "medium purple"])

    ### Main figure
    nrows = 2
    ncols = df_summary.dataset.nunique()
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, 5), sharey=True)

    ### curves (per dataset and metric)
    for col, dataset in enumerate(df_summary.dataset.unique()):

        ### title (dataset)
        axes[0,col].set_title(dataset)

        ### vertical inequality (gini)
        row = 0
        tmp = df_metadata.query("dataset.str.lower()==@dataset.lower()").copy()
        for i,metric in enumerate(metrics):
            X = tmp[metric].astype(np.float).values
            X_lorenz = lorenz_curve(X)
            gc = round(gini(X), 2)
            axes[row, col].plot(np.arange(X_lorenz.size) / (X_lorenz.size - 1), X_lorenz, label=metric, color=colors[i])
            axes[row, col].text(s=r'$Gini='+str(gc)+'$', x=0, y=0.9 if metric == metrics[0] else 0.8, color=colors[i])
            axes[row, 0].set_ylabel('Fraction of total wealth\nin Bottom-k%')
            axes[row, col].set_xlabel('Bottom-k% of nodes')
            axes[row, col].plot([0,1],[0,1],linestyle='--',color='grey')
            axes[row, col].xaxis.set_major_locator(plt.MaxNLocator(3))
            axes[row, col].xaxis.set_minor_locator(plt.MaxNLocator(10))
            axes[row, col].set_xlim(0-0.05, 1+0.05)

        ### horizontal inequality (groups)
        row = 1
        tmp = df_rank.query("dataset.str.lower()==@dataset.lower()").copy()
        for i, metric in enumerate(metrics):
            tmp_m = tmp.query("metric==@metric").copy()

            if tmp_m.shape[0] == 0:
                continue

            tmp_m.loc[:, 'rank'] = tmp['rank'].apply(lambda x: x / 100)
            tmp_m.sort_values("rank", inplace=True)
            tmp_m.plot(x='rank', y='fmt', ax=axes[row, col], label=metric, legend=col==ncols-1, color=colors[i])

            if col==ncols-1:
                axes[row, col].legend(loc='center right')

            fm = df_summary.query("dataset.str.lower()==@dataset.lower()").fm.unique()[0]
            _ = axes[row, col].axhline(fm, c='grey', ls='--')

            d = tmp_m.mae.unique()[0]
            _ = axes[row, col].text(s='$MAE='+str(round(d, 3))+'$',
                                     x=0,  # tmp['rank'].values[1],
                                     y=0.9 if metric == metrics[0] else 0.8, color=colors[i])

            axes[row, 0].set_ylabel('Fraction of minorities\nin Top-k%')
            axes[row, col].set_xlabel('Rank k%')
            axes[row, col].xaxis.set_major_locator(plt.MaxNLocator(3))
            axes[row, col].xaxis.set_minor_locator(plt.MaxNLocator(10))
            axes[row, col].set_xlim(0 - 0.05, 1 + 0.05)

    ### space between subplots
    plt.subplots_adjust(hspace=0.35, wspace=0.1)

    ### savefig
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()


def plot_vertical_and_horizontal_inequalities_per_dataset(df_metadata, df_rank, df_summary, dataset, fn=None):
    from org.gesis.libs.utils import gini
    from org.gesis.libs.utils import lorenz_curve

    fig, axes = plt.subplots(2, 2, figsize=(2 * 3, 5), sharey=True)

    # vertical inequality (gini)
    tmp = df_metadata.query("dataset.str.lower()==@dataset").copy()
    col = -1
    for metric in ['pagerank', 'wtf']:

        if metric not in tmp.columns:
            continue

        if tmp[metric].nunique() <= 1:
            continue

        col += 1

        # ALL
        X = tmp[metric].astype(np.float).values
        X_lorenz = lorenz_curve(X)
        gc = round(gini(X), 2)
        axes[0, col].plot(np.arange(X_lorenz.size) / (X_lorenz.size - 1), X_lorenz, label='ALL')
        axes[0, col].text(s='Gini all={}'.format(gc), x=0, y=0.9)

        # MAJ
        X = tmp.query("minority==0")[metric].astype(np.float).values
        X_lorenz = lorenz_curve(X)
        gc = round(gini(X), 2)
        axes[0, col].plot(np.arange(X_lorenz.size) / (X_lorenz.size - 1), X_lorenz, label='majority')
        axes[0, col].text(s='Gini maj={}'.format(gc), x=0, y=0.8)

        # MIN
        X = tmp.query("minority==1")[metric].astype(np.float).values
        X_lorenz = lorenz_curve(X)
        gc = round(gini(X), 2)
        axes[0, col].plot(np.arange(X_lorenz.size) / (X_lorenz.size - 1), X_lorenz, label='minority')
        axes[0, col].text(s='Gini min={}'.format(gc), x=0, y=0.7)

        axes[0, col].plot([0, 1], [0, 1], linestyle='--', color='grey')
        axes[0, col].set_title(metric)
        axes[0, 0].set_ylabel('% total wealth\nin Bottom-k%')
        axes[0, col].set_xlabel('Bottom-k% of nodes') # if col == 2 else '')

    # horizontal inequality (groups)
    tmp = df_rank.query("dataset.str.lower()==@dataset")
    col = -1
    for metric in ['pagerank', 'wtf']:

        tmp_m = tmp.query("metric==@metric").copy()
        if tmp_m.shape[0] <= 1:
            continue

        col += 1
        tmp_m.loc[:, 'rank'] = tmp['rank'].apply(lambda x: x / 100)

        tmp_m.sort_values("rank", inplace=True)
        tmp_m.plot(x='rank', y='fmt', ax=axes[1, col], legend=False)
        fm = df_summary.query("dataset.str.lower()==@dataset").fm.unique()[0]
        ax = axes[1, col].axhline(fm, c='grey', ls='--')

        d = tmp_m.mae.unique()[0]
        ax = axes[1, col].text(s='MAE={}'.format(round(d, 2)),
                               x=0.2,  # tmp['rank'].values[1],
                               y=0.75)  # max(fm+tmp.fmt.std(), tmp.fmt.max()-tmp.fmt.std()))

        axes[1, 0].set_ylabel('Fraction of minorities\nin Top-k%')
        axes[1, col].set_xlabel('Rank k%') # if col == 2 else '')

    plt.subplots_adjust(hspace=0.35, wspace=0.20)
    plt.suptitle("dataset = {}".format(dataset.upper()))

    ### savefig
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()