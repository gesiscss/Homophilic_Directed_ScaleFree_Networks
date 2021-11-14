import matplotlib as mpl
import warnings
warnings.simplefilter(action='ignore', category=mpl.MatplotlibDeprecationWarning)

################################################################################
# System dependencies
################################################################################
import powerlaw
import numpy as np
import pandas as pd
import collections
import networkx as nx
import seaborn as sns
from matplotlib import rc
from functools import reduce
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from powerlaw import plot_pdf, Fit, pdf
from palettable.colorbrewer.diverging import BrBG_11
from palettable.colorbrewer.diverging import BrBG_5
from palettable.colorbrewer.sequential import Blues_3, Reds_3, Greens_3
from palettable.colorbrewer.qualitative import Paired_11, Set2_8, Set1_9, Set3_8, Accent_8, Dark2_8, Set1_6
#https://jiffyclub.github.io/palettable/colorbrewer/sequential/
#https://jiffyclub.github.io/palettable/tableau/#tableaumedium_10
#https://jiffyclub.github.io/palettable/cartocolors/qualitative/

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
    
################################################################################
# Local dependencies
################################################################################
from org.gesis.lib import utils
from org.gesis.lib import graph

################################################################################
# Constants
################################################################################
MAIN_COLORS = {'min':'#ec8b67', 'maj':'#6aa8cb'}

################################################################################
# Plot setup
################################################################################
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
        
################################################################################
# Distributions
################################################################################

def plot_degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg, rotation=90)

def plot_degree_powerlaw(G):
    fig,axes = plt.subplots(1,2,figsize=(10,3))
    colors = ['blue','orange']
    labels = ['Majority','minority']
    titles = ['Outdegree', 'Indegree']
    
    for d in [0,1]: #in/ out degree (columns)
        for k in [0,1]: #min vs maj
            if d:
                data = [i for n,i in G.in_degree() if G.node[n][G.graph['label']]==k]
            else:
                data = [o for n,o in G.out_degree() if G.node[n][G.graph['label']]==k]
                
            fit = powerlaw.Fit(data, discrete=True)
            fig = fit.plot_pdf(linewidth=3, color=colors[k], label=labels[k], ax=axes[d])
            fit.power_law.plot_pdf(ax=axes[d], color=colors[k], linestyle='--', 
                                    label='Power law fit ({})'.format(round(fit.power_law.alpha,1)))
            fig.set_ylabel(u"p(X≥x)")
            fig.set_xlabel(titles[d])
            
        handles, lbs = fig.get_legend_handles_labels()
        fig.legend(handles, lbs, loc=3)


def plot_degree_powerlaw_fit(df_metadata):
    metrics = ['outdegree','indegree']#,'pagerank']
    fig,axes = plt.subplots(1,len(metrics),figsize=(len(metrics)*4,3))
    colors = ['blue','orange']
    labels = ['Majority','minority']
    title = 'Power-law degree distributions'
    
    for c,metric in enumerate(metrics):
        discrete = metric!='pagerank'
        
        for m in df_metadata.minority.unique():
            data_emp = df_metadata.query("kind=='empirical' & minority==@m")[metric].values
            data_fit = df_metadata.query("kind!='empirical' & minority==@m")[metric].values

            ax = axes[c]
            emp = powerlaw.Fit(data_emp, discrete=discrete)
            emp.power_law.plot_ccdf(ax=ax, color=colors[m], linestyle='solid', label='Empirical {}'.format(round(emp.power_law.alpha,1)))

            fit = powerlaw.Fit(data_fit, discrete=discrete)
            fit.power_law.plot_ccdf(ax=ax, color=colors[m], linestyle='dotted', label='Model {}'.format(round(fit.power_law.alpha,1)))
            ax.set_xlabel(metric)
            leg1 = ax.legend(loc=3) 
    
    hs = []
    for i,color in enumerate(colors):
        hs.append(mpatches.Patch(color=color, label=labels[i]))
    leg2 = plt.legend(handles=hs, loc=1)
    axes[-1].add_artist(leg1)

    axes[0].set_ylabel(u"p(X≥x)")  
    plt.suptitle(title)
    
def plot_lorenz_curve(df_metadata):
    
    metrics = ['pagerank']
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    title = 'Vertical Inequality\n(Individual level)'
    colors = ['green','lightgreen']
    labels = ['Empirical','Model']
    markers = ['-','--']
    
    for m, metric in enumerate(metrics):
        data_emp = df_metadata.query("kind=='empirical'")[metric].values
        lc_emp = utils.lorenz_curve(data_emp)
        ax.plot(np.arange(lc_emp.size)/(lc_emp.size-1), lc_emp, linestyle=markers[m], color=colors[0])
        
        data_fit = df_metadata.query("kind!='empirical'")[metric].values
        lc_fit = utils.lorenz_curve(data_fit)
        ax.plot(np.arange(lc_fit.size)/(lc_fit.size-1), lc_fit, linestyle=markers[m], color=colors[1])
        
    # baseline: equality
    ax.plot([0,1], [0,1], linestyle='--', color='grey')
    
    # legend 1: empirical vs model (colors)
    hs = []
    for i,color in enumerate(colors):
        hs.append(mpatches.Patch(color=color, label=labels[i]))
    leg1 = plt.legend(handles=hs, loc=2)
    
    # legend 2: metric (marker)
    lines = [Line2D([0], [0], color='black', linewidth=1, linestyle=markers[m]) for m,metric in enumerate(metrics)]
    leg2 = plt.legend(lines, metrics, loc=3)
    ax.add_artist(leg1)
    
    return

def plot_fraction_minorities(df_rank):
    fg = sns.catplot(data=df_rank, x='rank', y='fmt', hue='kind', 
                     kind='point', palette='BrBG', height=3, aspect=1, legend_out=False,
                     estimator=np.mean, ci='sd')
    fg.ax.axhline(df_rank.query("kind=='empirical' & rank==100").fmt.values[0], ls='--', c='grey', lw=1.0)
    
    
def plot_distribution_min_maj(datae, datam, dataset, model, metric):
    colors = ['red','blue'] #sns.color_palette("colorblind")
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    
    for minority in [0,1]:
        ##############
        # empirical
        ##############

        # scatter
        tmpe = datae.query("dataset==@dataset & minority==@minority").copy().loc[:,metric].astype(int).values
        x, y = pdf(tmpe[tmpe>0], linear_bins=False)
        ind = y>0
        y = y[ind]
        x = x[:-1]
        x = x[ind]
        axes[minority].scatter(x, y, color=colors[0], label='Empirical')

        # pdf
        plot_pdf(tmpe[tmpe>0], ax=axes[minority], color=colors[0], linewidth=2, label='pdf')

        # pdf powelaw fit
        if dataset not in ['seventh']:
            fit = Fit(tmpe, discrete=True)
            fit.power_law.plot_pdf(ax=axes[minority], linestyle='--', color=colors[0], label='power-law')
            posy = min(y)
            axes[minority].text(color=colors[0], x=1, y=posy, s='Empirical = {} ({},{})'.format(round(fit.power_law.alpha,2), int(fit.power_law.xmin), fit.power_law.xmax ))


        ##############
        # model
        ##############

        # scatter
        tmpm = datam.query("dataset==@dataset & kind==@model & minority==@minority").copy().loc[:,metric].astype(int).values
        x, y = pdf(tmpm[tmpm>0], linear_bins=False)
        ind = y>0
        y = y[ind]
        x = x[:-1]
        x = x[ind]
        axes[minority].scatter(x, y, color=colors[1], label=model)

        # pdf
        plot_pdf(tmpm[tmpm>0], ax=axes[minority], color=colors[1], linewidth=2, label='pdf')

        # pdf powelaw fit
        if dataset not in ['seventh']:
            fit = Fit(tmpm, discrete=True)
            fit.power_law.plot_pdf(ax=axes[minority], linestyle='--', color=colors[1], label='power-law')
            posy *= 10
            axes[minority].text(color=colors[1], x=1, y=posy, s='{} = {} ({},{})'.format(model,round(fit.power_law.alpha,2), int(fit.power_law.xmin), fit.power_law.xmax ))

    axes[0].set_title("Majorities")
    axes[1].set_title("minorities")
    plt.suptitle(dataset.upper() + '-' + metric.lower())
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.show()
    plt.close()
    
    
def plot_degree_distributions_groups_fit(df_summary_empirical, df_metadata_empirical, df_metadata_fit, model='DPAH', forcepl=False, fn=None):

    plt.close()

    ### main data
    metrics = ['indegree', 'outdegree']
    discrete = True
    labels = {0:'Majority', 1:'minority'}
    
    
    #datasets = reduce(np.intersect1d, (df_summary_empirical.dataset.unique(),
    #                                  df_metadata_empirical.dataset.unique(),
    #                                  df_metadata_fit.dataset.dropna().unique()))
    datasets = df_summary_empirical.dataset.unique()
    
    ### main plot
    nrows = len(metrics)
    ncols = len(datasets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, 5), sharey=False, sharex=False) #3, 4.5

    ### subplots
    colors = sns.color_palette("colorblind")
    for col, dataset in enumerate(datasets):

        axes[0, col].set_title(dataset)
        xye = {}
        xym = {}
        for row, metric in enumerate(metrics):

            ### Power-law fit
            txt_emp = "Empirical:" + "\n" + r"$\gamma_{M}=$" + "<maj>" + "\n" + r"$\gamma_{m}=$" + "<min>"
            txt_fit = model + "\n" + r"$\gamma_{M}=$" + "<maj>" + "\n" + r"$\gamma_{m}=$" + "<min>"

            for minority in sorted(df_metadata_fit.minority.unique()):
                sum_emp = df_summary_empirical.query("dataset.str.lower()==@dataset.lower()").iloc[0]

                data_emp = df_metadata_empirical.query("dataset.str.lower()==@dataset.lower() & minority==@minority")[metric].values.astype(np.float)
                data_fit = df_metadata_fit.query("dataset.str.lower()==@dataset.lower() & minority==@minority & kind==@model")[metric].values.astype(np.float)

                
#DPAH
#            outdegree   |  indegree
# APS         4  30      |  9   180
# BLOGS      16 180      |  35  600
# HATE       14 400      |   5  120
# SEVENTH   
# WIKIPEDIA   5  30      |  10  200 
                
#EMPIRICAL
#            outdegree   |  indegree
# APS           4  14    |    1   20
# BLOGS         6  70    |    3   120
# HATE          4  150   |    1    50 
# SEVENTH   
# WIKIPEDIA    10   50   |    7   200

                minmax = {'empirical':
                              {'indegree':
                                 {'aps':{'xmin':1, 'xmax':20},
                                  'blogs':{'xmin':3, 'xmax':120},
                                  'hate':{'xmin':1, 'xmax':50},
                                  'wikipedia':{'xmin':7, 'xmax':200}},
                              'outdegree':
                                 {'aps':{'xmin':4, 'xmax':14},
                                  'blogs':{'xmin':6, 'xmax':70},
                                  'hate':{'xmin':4, 'xmax':150},
                                  'wikipedia':{'xmin':10, 'xmax':50}}},
                          'DPAH':
                             {'indegree':
                                 {'aps':{'xmin':9, 'xmax':180},
                                  'blogs':{'xmin':35, 'xmax':600},
                                  'hate':{'xmin':5, 'xmax':120},
                                  'wikipedia':{'xmin':10, 'xmax':200}},
                              'outdegree':
                                 {'aps':{'xmin':4, 'xmax':30},
                                  'blogs':{'xmin':16, 'xmax':180},
                                  'hate':{'xmin':14, 'xmax':400},
                                  'wikipedia':{'xmin':5, 'xmax':30}}}}
    
                ### Empirical:
                try:
                    label = '{} empirical'.format(labels[minority])
                    
                    if forcepl and 'empirical' in minmax and dataset.lower() in minmax['empirical'][metric]:
                        xmin = minmax[model][metric][dataset.lower()]['xmin']
                        xmax = minmax[model][metric][dataset.lower()]['xmax']
                        fit_emp = graph.fit_power_law_force(data_emp, discrete=discrete, xmin=xmin, xmax=xmax)
                        fit_emp.power_law.plot_pdf(ax=axes[row, col], linestyle='-', color=colors[minority], label=label)
                    else:
                        fit_emp = graph.fit_power_law(data_emp, discrete=discrete)
                        fit_emp.power_law.plot_pdf(ax=axes[row, col], linestyle='-', color=colors[minority], label=label)
                    
                    txt_emp = txt_emp.replace("<min>" if minority else "<maj>", str(round(fit_emp.power_law.alpha,1)))
                except Exception as ex:
                    print(ex)
                    print('?')
                    pass
                
                


                ### Model:
                try:
                    if data_fit.shape[0] > 0:  
                        label = '{} {}'.format(labels[minority], model)
                        
                        if forcepl and model in minmax and dataset.lower() in minmax[model][metric]:
                            xmin = minmax[model][metric][dataset.lower()]['xmin']
                            xmax = minmax[model][metric][dataset.lower()]['xmax']
                            fit_mod = graph.fit_power_law_force(data_fit, discrete=discrete, xmin=xmin, xmax=xmax)
                            fit_mod.power_law.plot_pdf(ax=axes[row, col], linestyle='--', color=colors[minority], label=label)
                        else:
                            fit_mod = graph.fit_power_law(data_fit, discrete=discrete)
                            fit_mod.power_law.plot_pdf(ax=axes[row, col], linestyle='--', color=colors[minority], label=label)
                            
                        txt_fit = txt_fit.replace("<min>" if minority else "<maj>", str(round(fit_mod.power_law.alpha)))
                except:
                    pass
                
            ### Exponents
            if row == 0:
                # indegree
                xye[metric] = {'aps': (40, 0.5), 
                               'hate': (30, 0.5), 
                               'blogs': (150, 0.05), 
                               'wikipedia': (40, 0.4)}
                
                xym[metric] = {'aps': (2, 0.0002), 
                               'hate': (3, 0.001), 
                               'blogs': (32, 0.00025), 
                               'wikipedia': (2, 0.0002)}

            else:
                # outdegree
                xye[metric] = {'aps': (25, 0.7), 
                               'hate': (50, 0.4), 
                               'blogs': (80, 0.25),
                               'wikipedia': (25, 0.6)}
                
                xym[metric] = {'aps': (4, 0.0001), 
                               'hate': (2, 0.000015), 
                               'blogs': (18, 0.0001),
                               'wikipedia': (7, 0.0002)}

            ### Column name (dataset)
            axes[row, col].text(s=txt_emp, x=xye[metric][dataset.lower()][0], y=xye[metric][dataset.lower()][1], horizontalalignment='left', va='top')
            axes[row, col].text(s=txt_fit, x=xym[metric][dataset.lower()][0], y=xym[metric][dataset.lower()][1], horizontalalignment='left', va='top')

            ### y-label right
            if col == ncols - 1:
                xt = axes[row, col].get_xticks()
                yt = axes[row, col].get_yticks()
                axes[row, col].text(s=metric,
                                    x=700 if row == 0 else 78,
                                    y=0.003 , rotation=-90, va='center')

    ### legend
    width = 4*1.1
    row = 0
    col = int(axes.shape[1] / 2)
    axes[row,col].legend(loc='lower left',
                     bbox_to_anchor=(width/-1.8, 1.12, width, 0.2), mode='expand',
                     ncol=4, handletextpad=0.3, frameon=False)

    ### ylabel left
    ylabel = 'P(x)'
    row = int(axes.shape[0] / 2)
    col = 0
    if nrows % 2 != 0:
        axes[row, col].set_ylabel(ylabel)
    else:
        xt = axes[row, col].get_xticks()
        yt = axes[row, col].get_yticks()
        axes[row, col].text(min(xt) * 10,
                            max(yt) / 15,
                            ylabel, {'ha': 'center', 'va': 'center'}, rotation=90)

    ### xlabel
    xlabel = 'Degree'
    row = -1
    col = int(axes.shape[1] / 2)
    if ncols % 2 != 0:
        axes[row, col].set_xlabel(xlabel)
    else:
        xt = axes[row, col].get_xticks()
        yt = axes[row, col].get_yticks()
        axes[row, col].text(min(xt) * 10,
                            min(yt) * 4.7,
                            xlabel, {'ha': 'center', 'va': 'center'}, rotation=0)

    ### space between subplots
    plt.subplots_adjust(hspace=0.2, wspace=0.32)

    ### Save fig
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    ###
    plt.show()
    plt.close()
   
   
    
def plot_gini_density_distribution(df, title, fn=None):
    x = 'd'
    y = 'gini'
    metrics = ['PageRank','WTF']
    colors = ['red', 'blue']
    bps = []

    fig,ax = plt.subplots(1,1,figsize=(6,4))
    for metric,color in zip(*(metrics,colors)):
        tmp = df.query("metric==@metric.lower() & rank==100", engine='python').copy()
        labels, data = zip(*[(name, tmp[y]) for name, tmp in tmp.groupby(x)])
        tmp = ax.boxplot(data)
        for box in tmp['boxes']:
            box.set(color=color, linewidth=3)
        bps.append(tmp)

    ### details
    ax.set_title(title)
    ax.set_xlabel('Edge density')
    ax.set_ylabel('Inequality\n(Gini coef. of entire rank distribution)')
    ax.set_xticklabels(labels)
    ax.set_ylim((0-0.03,1+0.03))
    ax.legend([bp['boxes'][0] for bp in bps], metrics, loc='upper right')

    ### vertical baselines
    ax.axhline(y=0.3, ls='--', color='darkgrey', lw=0.5)
    ax.axhline(y=0.6, ls='--', color='darkgrey', lw=0.5)

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    ###
    plt.show()
    plt.close()
    
    
################################################################################
# Ranking Empirical
################################################################################

def plot_vh_inequalities_empirical(df_rank, graph_fnc=None, datapath=None, vtype='mae', metric=None, fn=None):

    tmp = df_rank.query("kind=='empirical' & metric in ['pagerank','wtf']").copy()
    if metric != 'all':
        tmp = df_rank.query("metric==@metric").copy()

    ### plot setup
    r = 3 # rows
    c = tmp.dataset.nunique() # columns (datasets)
    w = 2.2 # width cell
    h = 2 # height cell
    lw = 1 # line width plot
    blw = 0.8 # line witdth baselines
    
    ### graph setup
    nsize = 1           # node size
    ecolor = '#c8cacc'  # edge color
    ewidth = 0.1        # thickness of edges
    asize = 1           # size of edge arrow (viz)
    
    ### https://xkcd.com/color/rgb/
    colors = sns.xkcd_palette(["medium green", "medium purple"])
    lightcolors = sns.xkcd_palette(["light green", "light purple"])
    
    ### plot
    fig, axes = plt.subplots(r,c,figsize=(c*w, r*h),sharex=False, sharey=False)
    
    counter = 0
    for mc, met in enumerate(tmp.metric.unique()):
        counter -= 0.15
        
        ### graph, individual and vertical inequalities (row 0, row 1, row 2)
        for i, (dataset, df_group) in enumerate(tmp.query("metric==@met").groupby("dataset")):
            # graph
            if mc == 0:
                try:
                    g = graph_fnc(datapath,dataset)
                    g = g.subgraph(max(nx.connected_components(g.to_undirected()), key=len))
                    ncolor = [MAIN_COLORS['min'] if obj[g.graph['label']] else MAIN_COLORS['maj'] 
                              for n,obj in g.nodes(data=True)]
                    
                    nx.draw(g, 
                            pos = nx.nx_pydot.graphviz_layout(g, prog='neato'), 
                            edge_color=ecolor,
                            node_size=nsize, 
                            node_color=ncolor, 
                            width=ewidth, 
                            arrows=True, 
                            arrowsize=asize,
                            with_labels=False, 
                            ax=axes[0,i])
                    
                except Exception as ex:
                    print(ex)
                axes[0,i].axis('off')
                axes[0,i].set_title(dataset.title() if dataset!='aps' else 'APS')

            # inequalities
            df_group = df_group.sort_values("rank")
            axes[1,i].plot(df_group['rank'], df_group['gt'], color=colors[mc], label=met, linewidth=lw)  # individual: gini
            axes[2,i].plot(df_group['rank'], df_group['fmt'], color=colors[mc], label=met, linewidth=lw) # group: % of min

            # baseline (gini_all)
            #axes[1,i].axhline(df_group['gini'].unique(), c=lightcolors[mc], ls='--', lw=blw)
            axes[1,i].text(x=10, y=1.0+counter, s="Gini$_{}={}$".format('{all}',round(df_group['gini'].unique()[0],2)), color=colors[mc], zorder=100)

            # baseline (fraction of min in network)
            axes[2,i].axhline(df_group['fm'].unique(), c='grey', ls='--', lw=blw)
            axes[2,i].text(x=50, y=1.0+counter, s="{}: {}".format(vtype.upper(),round(df_group[vtype].unique()[0],2)), color=colors[mc], zorder=100)

    axes[1,0].set_ylabel("Gini\nin top-k\%")
    axes[2,0].set_ylabel("\% of minorities\nin top-k\%")
    axes[1,c-1].legend(loc='lower right')
    
    # Labels
    for i in np.arange(0,c):
        
        axes[1,i].set_xlabel('')
        axes[1,i].set_xticklabels("")
        
        axes[1,i].set_ylim((-0.1,1.1))
        axes[2,i].set_ylim((-0.1,1.1))
        
        axes[1,i].set_xlim((5,100))
        axes[2,i].set_xlim((5,100))

        axes[2,i].set_xticks([20,50,80])

        if i>0:
            axes[1,i].set_yticklabels("")
            axes[2,i].set_yticklabels("")

        if c%2 != 0:
            axes[2,i].set_xlabel('' if (i!=int(c/2)) else "Top-k\% PageRank")


    ### border color
    for ax in axes.flatten():
        ax.tick_params(color='grey', labelcolor='grey')
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')

    ### Save fig        
    plt.subplots_adjust(hspace=0.05, wspace=0.1)
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print("{} saved!".format(fn))

    plt.show()
    plt.close()

    return
    
def plot_vh_inequalities_empirical_summary(df_rank, x='mae', fn=None):

    if x not in ['mae','me']:
        raise Exception('invalid x-axis (horizontal ineq.)')
        
    ### only man data points
    tmp = df_rank.groupby(['dataset','kind','metric']).mean().reset_index()
    tmp.drop(columns=['rank', 'fmt'], inplace=True)

    ### main plot
    tmp.sort_values(by=["dataset","metric"], inplace=True)
    fg = sns.catplot(data=tmp,
                     x=x, y='gini',
                     height=2.0,aspect=0.9,
                     hue='metric', col='dataset')
    [plt.setp(ax.texts, text="") for ax in fg.axes.flat]
    fg.set_titles(row_template='{row_name}', col_template='{col_name}')

    ### labels and xticks
    for ax in fg.axes.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")

        # xticks
        #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        #ax.xaxis.set_minor_locator(plt.MaxNLocator(10))

        # xticklabels
        xtls = ax.get_xticklabels()
        ax.set_xticklabels([round(float(xtl.get_text()), 2) for i, xtl in enumerate(xtls)], rotation=0)
        
        #ax.axvline(x=0 if x=='me' else 0.5, ls='--', c='black', lw=0.5)
        #ax.axhline(y=0.5, ls='--', c='black', lw=0.5)
        
    ### ylabel
    ylabel = 'Gini'
    if fg.axes.shape[0] % 2 != 0:
        fg.axes[int(fg.axes.shape[0] / 2), 0].set_ylabel(ylabel)
    else:
        fg.axes[int(fg.axes.shape[0] / 2), 0].text(-50, 0.28, ylabel, {'ha': 'center', 'va': 'center'}, rotation=90)

    ### xlabel
    xlabel = x.upper() #'MAE of fraction of minorities in top-k%'
    if fg.axes.shape[1] % 2 != 0:
        fg.axes[-1, int(fg.axes.shape[1] / 2)].set_xlabel(xlabel)
    else:
        fg.axes[-1, int(fg.axes.shape[1] / 2)].text(0, 0.15, xlabel, {'ha': 'center', 'va': 'center'}, rotation=0)

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))


def plot_vh_inequalities_per_dataset_and_metric(df_metadata, df_rank, df_summary, fn=None):
    plt.close()
    metrics = ['pagerank','wtf']
    colors = sns.xkcd_palette(["medium green", "medium purple"])

    ### Main figure
    nrows = 2
    ncols = df_summary.dataset.nunique()
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, 5), sharey=True)
    
    ### curves (per dataset and metric)
    df_summary.sort_values("dataset", inplace=True)
    for col, dataset in enumerate(df_summary.dataset.unique()):

        ### title (dataset)
        axes[0,col].set_title(dataset)

        ### vertical inequality (gini)
        row = 0
        tmp = df_metadata.query("dataset.str.lower()==@dataset.lower()").copy()
        for i,metric in enumerate(metrics):
            X = tmp[metric].astype(np.float).values
            X_lorenz = utils.lorenz_curve(X)
            gc = round(utils.gini(X), 2)
            axes[row, col].plot(np.arange(X_lorenz.size) / (X_lorenz.size - 1), X_lorenz, label=metric, color=colors[i])
            axes[row, col].text(s=r'$Gini='+str(gc)+'$', x=0, y=0.9 if metric == metrics[0] else 0.8, color=colors[i])
            axes[row, 0].set_ylabel('Fraction of total wealth\nin Bottom-k%')
            axes[row, col].set_xlabel('')
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
    
            d = tmp_m.me.unique()[0]
            _ = axes[row, col].text(s='$ME='+str(round(d, 3))+'$',
                                     x=0,  # tmp['rank'].values[1],
                                     y=0.9 if metric == metrics[0] else 0.8, color=colors[i])
        
            axes[row, 0].set_ylabel('Fraction of minorities\nin Top-k rank %')
            axes[row, col].set_xlabel('')
            axes[row, col].xaxis.set_major_locator(plt.MaxNLocator(3))
            axes[row, col].xaxis.set_minor_locator(plt.MaxNLocator(10))
            axes[row, col].set_xlim(0 - 0.05, 1 + 0.05)

    ### xlabel
    xlabels = ['Bottom-k% of nodes', 'Top-k rank %']
    col = int(axes.shape[1] / 2)
    for row, xlabel in enumerate(xlabels):
        if ncols % 2 != 0:
            axes[row, col].set_xlabel(xlabel)
        else:
            xt = axes[row, col].get_xticks()
            yt = axes[row, col].get_yticks()
            axes[row, col].text(min(xt) + 0.3,
                                min(yt) - 0.1,
                                xlabel, {'ha': 'center', 'va': 'center'}, rotation=0)

    ### space between subplots
    plt.subplots_adjust(hspace=0.35, wspace=0.1)

    ### savefig
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()
    
    

    
################################################################################
# Ranking Fit
################################################################################

def plot_inequalities_fit_improved(df_best_fit, df_empirical, models, markers, valid_metrics=None, vtype='mae', fn=None):
    
    ### attribute for horizontal inequality
    _, mini, mid, _ = setup_plot_HI_simplified(vtype)
    label = vtype.upper() 
    
    ### datasets (hue1) and metrics (columns)
    datasets = df_empirical.dataset.unique().categories #sorted(df_best_fit.dataset.str.lower().unique())
    metrics = sorted(df_best_fit.metric.unique()) 
    metrics = metrics if valid_metrics is None else [m for m in metrics if m in valid_metrics]
    
    ### init plot
    ncol = len(metrics)
    nrow = 1
    colors = cycle(Set1_9.mpl_colors)
    x, y = vtype, 'gini'
    xmin, xmax = -1,1 #df_best_fit[x].min(), df_best_fit[x].max()
    ymin, ymax = 0,1 #df_best_fit[y].min(), df_best_fit[y].max()
    fig,axes = plt.subplots(nrow, ncol, figsize=(2.2*ncol, 2.2), sharey=True, sharex=True)
    
    ### scatter plot
    tmp_emp = df_empirical.groupby(['dataset','metric','kind']).mean().reset_index()
    for h, dataset in enumerate(datasets):
        color = next(colors)
        
        for c, metric in enumerate(metrics):

            ### empirical
            tmp = tmp_emp.query("metric == @metric & dataset.str.lower() == @dataset")
            axes[c].scatter(x=tmp[x], y=tmp[y], color=color, marker='s', label=dataset)
            
            ### synthetic
            tmp = df_best_fit.query("metric == @metric & dataset.str.lower() == @dataset")
            axes[c].scatter(x=tmp[x], y=tmp[y], color=color, marker=markers[models.index(tmp.kind.iloc[0])], label=None)
            
            ### visuals
            axes[c].set_xlim((xmin-0.03, xmax+0.03))
            axes[c].set_ylim((ymin-0.03, ymax+0.03))
            axes[c].set_ylabel(y.title() if c==0 else '')
            axes[c].set_xlabel(label)
            axes[c].set_title(metric.upper())
            
            for i in np.arange(mini,1.0+0.1,0.1):
               i = round(i,1)
               axes[c].axhline(y=i, lw=0.5, ls='-', c='#FAF8F7', zorder=0)
               axes[c].axvline(x=i, lw=0.5, ls='-', c='#FAF8F7', zorder=0)

            axes[c].axhline(y=0.3, lw=0.5, ls='--', c='grey', zorder=0)
            axes[c].axhline(y=0.6, lw=0.5, ls='--', c='grey', zorder=0)

            if vtype == 'mae':
                axes[c].axvline(x=0.5, lw=0.5, ls='--', c='grey', zorder=0)
            else:
                smooth = 0.05
                axes[c].axvline(x=0.0+smooth, lw=0.5, ls='--', c='grey', zorder=0)
                axes[c].axvline(x=0.0-smooth, lw=0.5, ls='--', c='grey', zorder=0)

    ### legend 1 (datasets) 
    legend1 = axes[-1].legend(bbox_to_anchor=(1.04,1), borderaxespad=0, title='Dataset')
    
    ### legend 2 (empirical, model)
    labels = [m for m in models if m in df_best_fit.kind.unique()]
    h = [plt.plot([],[], color="black", marker=markers[models.index(m)], ls="")[0] for m in labels]
    axes[-1].legend(handles=h, labels=labels, bbox_to_anchor=(2.14,1), borderaxespad=0, title="Model", frameon=True)
    plt.gca().add_artist(legend1)
    
    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()

    
def plot_inequalities_fit(df_fit, df_empirical, models, markers, vtype='mae', fn=None):
    vat, mini, mid, color = setup_plot_HI_simplified(vtype)
    
    ### data & same datasets (uppercase)
    metrics = ['pagerank','wtf'] #, 'indegree','outdegree']
    datasets = sorted([d.lower() for d in df_empirical.dataset.unique()])
    
    ### best model
    
    # calculating gm (sum of absolute differences between model and empirical gini and mae)
    data = df_fit.query("metric in @metrics & not gt.isnull()", engine='python').copy()
    data.loc[:,'dataset'] = data.apply(lambda row: [d for d in datasets if d.lower()==row.dataset][0] ,axis=1)
    data.loc[:,'gm'] = data.apply(lambda row: 
                        abs(row['gt'] - _get_mean_val_from_df(df_empirical, 'gt', row)) + 
                        abs(row[vat] - _get_mean_val_from_df(df_empirical, vat, row)) , axis=1)
    data = data.groupby(['dataset','kind','metric']).mean().reset_index()
    
    # searching for smalles gm
    idx = data.groupby(['metric','dataset']).apply(lambda data:data.gm.abs().argmin())
    data = data.loc[idx, :]
    
    # getting empirical mean values
    tmp = df_empirical.groupby(['metric','kind','dataset']).mean().reset_index()
    tmp.loc[:, 'gm'] = None
    cols = tmp.columns
    
    # combining empirical and best model
    data = data[cols].append(tmp[cols], ignore_index=True)
    
    ### colors
    colors = cycle(Set1_9.mpl_colors)
    me = 's'
    mm = set()
    
    ### plot per metric / model / dataset
    fig,axes = plt.subplots(1,len(metrics),figsize=(len(metrics)*3.7,3),sharex=True,sharey=True)
    for dataset in datasets:
        color = next(colors)
        
        for c, metric in enumerate(metrics): 
            axes[c].set_title(metric.upper())

            # empirical
            tmp = data.query("dataset==@dataset & metric==@metric & kind=='empirical'")
            axes[c].scatter(x=tmp[vat], y=tmp.gini, color=color, label=dataset, marker=me)

            # model
            for m,model in zip(*(markers,models)):
                tmp = data.query("dataset==@dataset & metric==@metric & kind==@model")
                if tmp.shape[0] > 0:
                    axes[c].scatter(x=tmp[vat], y=tmp.gini, color=color, marker=m, s=200, label=None)
                    mm.add(model)
               
            # visuals
            axes[c].set_xlabel("Horizontal Inequality\n({} fraction of minorities in top-k rank)".format(vtype.upper()))
            axes[c].set_ylabel("")
            axes[c].set_ylim((0-0.03,1+0.03))
            axes[c].set_xlim((mini-0.03,1+0.03))
            for i in np.arange(mini,1.0+0.1,0.1):
                i = round(i,1)
                axes[c].axhline(y=i, lw=0.5 if i!=0.5 else 1, ls='--', c='lightgrey' if i!=0.5 else 'black', zorder=0)
                axes[c].axvline(x=i, lw=0.5 if i!=mid else 1, ls='--', c='lightgrey' if i!=mid else 'black', zorder=0)
            
    ### general visuals
    axes[0].set_ylabel("Vertical Inequality\n(Gini of rank distribution)")
        
    ### legend 1 (datasets) 
    legend1 = axes[-1].legend(bbox_to_anchor=(1.04,1), borderaxespad=0, title='Dataset')
    
    ### legend 2 (empirical, model)
    h = [plt.plot([],[], color="black", marker=markers[models.index(m)], ls="")[0] for m in mm]
    axes[-1].legend(handles=h, labels=mm, bbox_to_anchor=(1.04,0 if len(metrics)==2 else -0.2), 
                    loc='lower left',
                    title="Model", frameon=True, borderaxespad=0)
    plt.gca().add_artist(legend1)
        
    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
        
def plot_vh_inequalities_fit(df_rank, x='mae', group=False, kind=['empirical','DPAH'], metrics='all', fn=None):

    if x not in ['mae','me']:
        raise Exception('Invalid x-axis (horizontal ineq.)')

    datasets = df_rank.dataset.unique()
    models = df_rank.kind.unique()
    
    ### only main data points
    if metrics == 'all':
        metrics = ['pagerank', 'wtf']
    tmp = df_rank.query("kind in @kind & metric in @metrics").copy()
    tmp = tmp.groupby(['dataset','kind','metric','epoch']).mean().reset_index()
    tmp.drop(columns=['rank', 'fmt'], inplace=True)
    tmp.loc[:,'dataset'] = tmp.loc[:,'dataset'].str.lower()
    
    if group:
        tmp = tmp.groupby(['dataset','kind','metric']).mean().reset_index()

    ### main plot
    nrows = tmp.metric.nunique()
    ncols = tmp.dataset.nunique()
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2., nrows * 2.), sharey=True, sharex=True)

    ### subplots
    #colors = sns.color_palette() #"tab20")
    colors = ['black'] + Set1_6.mpl_colors
    for col, dataset in enumerate(datasets):

        ax = axes[0, col] if nrows > 1 else axes[col]
        ax.set_title(dataset)

        for row, metric in enumerate(tmp.metric.unique()):

            ### y-label right
            if nrows > 1:
                if col == ncols-1:
                    ax = axes[row, col] if nrows > 1 else axes[col]
                    ax.text(s=metric,
                            x=0.9,# if nrows >1 else 0.34,
                            y=0.5+(len(metric)*0.018),# if nrows >1 else 0.46,
                            rotation=-90)
                                    
            for hue, kind in enumerate(models):
                data = tmp.query("dataset==@dataset & metric==@metric & kind==@kind").copy()
                ax = axes[row, col] if nrows > 1 else axes[col]
                ax.scatter(y=data.gini.values, x=data[x].values, label=kind, color=colors[hue], 
                          marker='x' if kind!='empirical' else 'o',
                          zorder=1000 if kind == 'empirical' else 1)

    ### legend
    ax = axes[0, -1] if nrows > 1 else axes[-1]
    ax.legend(bbox_to_anchor=(1.18 if nrows>1 else 1.05,1), borderaxespad=0)

    
    ### baseline
    for ax in axes.flatten():
        ax.axhline(y=0.3, ls='--', color='darkgrey', lw=0.5)
        ax.axhline(y=0.6, ls='--', color='darkgrey', lw=0.5)
        ax.set_title(ax.get_title().title() if ax.get_title() != 'aps' else 'APS')
        
        if x=='mae':
            ax.axvline(x=0.5, ls='--', color='lightgrey', lw=0.5)
        elif x=='me':
            beta = 0.05
            ax.axvline(x=0.0-beta, ls='--', color='lightgrey', lw=0.5)
            ax.axvline(x=0.0+beta, ls='--', color='lightgrey', lw=0.5)
    
    ### ylabel left
    ylabel = 'Inequality\n(Gini coef. of entire rank distribution)'
    #ylabel = 'Individual Inequality\n(Gini coef. of entire rank distribution)'
    if nrows % 2 != 0:
        ax = axes[int(axes.shape[0]/2), 0] if nrows > 1 else axes[0]
        ax.set_ylabel(ylabel)
    else:
        ax = axes[int(axes.shape[0] / 2), 0] if nrows > 1 else axes[0]
        ax.text(-0.85 if not group else -0.85,
                1.1,
                ylabel, {'ha': 'center', 'va': 'center'}, rotation=90)

    ### xlabel
    xlabel = 'Inequity\n({} error of fraction of minorities across all top-k\% rank)'.format('Mean' if x=='me' else 'Mean absolute')
    #xlabel = 'Group Inequality\n({} error of fraction of minorities across all top-k\%'.format('Mean' if x=='me' else 'Mean absolute')
    if ncols % 2 != 0:
        ax = axes[-1, int(axes.shape[1]/2)] if nrows > 1 else axes[int(axes.shape[0]/2)]
        ax.set_xlabel(xlabel)
    else:
        ax = axes[-1, int(axes.shape[1] / 2)] if nrows > 1 else axes[int(axes.shape[0] / 2)]
        ax.text(-0.20,
                -0.1 if not group else 0.05,
                xlabel, {'ha': 'center', 'va': 'center'}, rotation=0)

    ### limits
    #for ax in axes.flatten():
    #    smooth=0.03
    #    mi=0 if x=='mae' else -1
    #    ax.set_ylim(0.0-smooth,1.0+smooth)
    #    ax.set_xlim(mi-smooth,1.0+smooth)
        
    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()

        
################################################################################
# Ranking Synthetic
################################################################################
        
def setup_plot_HI_simplified(vtype='mae'):
    x01 = ['mae','aefmt']
    x11 = ['me','efmt']
    pos = vtype in x01
    neg = vtype in x11
    
    vat = vtype #'aefmt' if vtype == 'mae' else 'efmt' if vtype == 'me' else None
    mini = 0 if pos else -1 if neg else None
    mid = 0.5 if pos else 0.0 if neg else None
    #color = "YlGnBu" if pos else "YlOrRd" if neg else None
    color = "YlGnBu" if pos else "RdBu" if neg else None
    return vat, mini, mid, color


def plot_inequalities_simplified_by_model(df, model, metric, vtype='mae', title=False, fn=None):
    vat, mini, mid, color = setup_plot_HI_simplified(vtype)
    
    ### data
    data = df.query("metric == @metric & kind == @model").copy()
        
    ### x,y axis
    values = ['gini', vat]
    nrows = len(values)
    ncols = data.fm.nunique()
    ind = 'hMM' 
    col = 'hmm' 
    
    ### plot
    fig,axes = plt.subplots(nrows,ncols,figsize=(ncols*2,nrows*2),sharey=True,sharex=True)

    for c,fm in enumerate(sorted(data.fm.unique())):
        tmp = data.query("fm==@fm")
        r = 0
        for ax, value in zip(*(axes,values)):
            vmin = mini if value == vat else 0
            label = vtype.upper() if value == vat else value.upper()
            cmap = color if value == vat else "YlGnBu"
            
            ax = sns.heatmap(tmp.pivot_table(index=ind,columns=col,values=value,aggfunc=np.mean), 
                             cmap=cmap, 
                             vmin=vmin, 
                             vmax=1,
                             ax=axes[r,c], 
                             cbar=c==ncols-1,
                             cbar_kws={'label': label, 'use_gridspec':False, 'anchor':(2.2,2.2)})

            ax.set_title("fm = {}".format(fm) if r==0 else '')
            ax.set_xlabel(col if r==nrows-1 and c==int(ncols/2.) else '')
            ax.set_xticklabels(ax.get_xticklabels() if r==nrows-1 else [], rotation=0)
            ax.set_ylabel(ind if c==0 else '')
            ax.set_aspect('auto')

            ytl = ax.get_yticklabels()
            if c == 0 and len(ytl)>0:
                ax.set_yticklabels(ytl, rotation=0)

            if c==0:
                y = ax.get_yticklabels()
            try:
                cbar = ax.collections[0].colorbar
                cbar.set_label(label, ha='right', rotation=-90, va='center')
            except:
                pass
            r += 1

    if title:
        _ = fig.suptitle(model, y=1.)
    
    ### space between subplots
    plt.gca().invert_yaxis()
    plt.subplots_adjust(hspace=0.1, wspace=0.12)
    
    ### Save fig
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
        

def plot_inequalities_simplified(df, models, metric, vtype='mae', fm=None, sym=True, title=False, fn=None):
    vat, mini, mid, color = setup_plot_HI_simplified(vtype)
    
    ### data
    s = "hmm == hMM" if sym else "fm == @fm"
    s = "metric == @metric & {}".format(s)
    data = df.query(s).copy()
    if sym:
        data.rename(columns={'hmm':'h'}, inplace=True)
        
    ### x,y axis
    values = ['gini', vat]
    nrows = len(values)
    ncols = len(models)
    ind = 'fm' if sym else 'hMM' 
    col = 'h' if sym else 'hmm' 
    
    ### plot
    fig,axes = plt.subplots(nrows,ncols,figsize=(ncols*2,nrows*2),sharey=True,sharex=not sym)

    for c,model in enumerate(models):

        tmp = data.query("kind==@model")
        r = 0
        for ax, value in zip(*(axes,values)):
            vmin = mini if value == vat else 0
            label = vtype.upper() if value == vat else value.upper()
            cmap = color if value == vat else "YlGnBu"
            
            ax = sns.heatmap(tmp.pivot_table(index=ind,columns=col,values=value,aggfunc=np.mean), 
                             cmap=cmap, vmin=vmin, vmax=1,
                             ax=axes[r,c], 
                             cbar=c==ncols-1,
                             cbar_kws={'label': label, 'use_gridspec':False, 'anchor':(2.2,2.2)})

            ax.set_title(model if r==0 else '')
            ax.set_xlabel(col if r==nrows-1 and c==int(ncols/2.) else '')
            ax.set_xticklabels(ax.get_xticklabels() if r==nrows-1 else [], rotation=0)
            ax.set_ylabel(ind if c==0 else '')
            ax.set_aspect('auto')

            ytl = ax.get_yticklabels()
            if c == 0 and len(ytl)>0:
                ax.set_yticklabels(ytl, rotation=0)

            if c==0:
                y = ax.get_yticklabels()
            try:
                cbar = ax.collections[0].colorbar
                cbar.set_label(label, ha='right', rotation=-90, va='center')
            except:
                pass
            r += 1

    if title:
        _ = fig.suptitle("Symmetric Homophily" if sym else "hMM vs. hmm [fm={}]".format(fm), y=1.)
    
    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.gca().invert_yaxis()

    ### Save fig
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
    
    
def plot_inequalities(df, models, markers, vtype='mae', mean=False, metric="pagerank", empirical=None, title=None, fn=None):
    vat, mini, mid, color = setup_plot_HI_simplified(vtype)
    title = 'Model' if title is None else title
    
    ### data
    data = df.query("metric == @metric").copy()
    if mean:
        data = data.groupby(['kind','N','fm','d','hmm','hMM','ploM','plom'])[['gini','me','mae']].agg(['mean','std']).reset_index()
    
    ### color
    colors = Set1_6.mpl_colors
    colors = cycle(colors)
    
    ### plot
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    zorder = len(models)
    
    handles = []
    zorder = len(models)
    y = 'gini'
    for model,marker in zip(*(models,markers)):
        tmp = data[data[('kind','')]==model]
        #ax.scatter(x=tmp[vtype], y=tmp[y], color=next(colors), label=model, marker=marker, zorder=zorder)
        #zorder-=1
        x = tmp[('me','mean')]
        xe = tmp[('me','std')]
        y = tmp[('gini','mean')]
        ye = tmp[('gini','std')]

        h, = ax.plot(x, y, 'o', color=next(colors), label=model, markersize=1, zorder=zorder)
        handles.append(h)
        ax.errorbar(x=x, y=y, xerr=xe, yerr=ye, fmt='none', alpha=0.5, ecolor='grey', zorder=zorder)
        zorder -= 1
        
    if empirical is not None:
        legend1 = ax.legend(title=title, bbox_to_anchor=(1.04,1), borderaxespad=0, frameon=False)
    else:
        #ax.legend(title='Model',bbox_to_anchor=(0.5,1.0), loc="upper right", ncol=2)
        ax.legend(title=title,
                 handles=handles,
                 ncol=len(models),
                 bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, frameon=False, markerscale=6)
            
            
    ### empirical
    zorder = 1000
    h = []
    m = 's'
    if empirical is not None:
        datasets = sorted(empirical.dataset.unique().tolist())
        for dataset in datasets:
            tmp = empirical.query("dataset==@dataset & metric == @metric")
            color = next(colors)
            ax.scatter(x=tmp[vtype], y=tmp[y], color=color, label=dataset, marker=m, zorder=zorder)
            h.append(plt.plot([],[], color=color, marker=m, ls="")[0])
            zorder-=1
            
        ### second legend
        ax.legend(handles=h, labels=datasets, title="Empirical", frameon=True,
                  bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        plt.gca().add_artist(legend1)

        
    ### visuals
    #for i in np.arange(mini,1.0+0.1,0.1):
    #    i = round(i,1)
    #    ax.axhline(y=i, lw=0.1, ls='--', c='lightgrey')
    #    ax.axvline(x=i, lw=0.1, ls='--', c='lightgrey')
    
    ax.axhline(y=0.3, lw=0.5, ls='--', c='darkgrey')
    ax.axhline(y=0.6, lw=0.5, ls='--', c='darkgrey')
    if vtype in ['mae']:
        ax.axvline(y=0.5, lw=0.5, ls='--', c='darkgrey')
    else:
        ax.axvline(x=0.05, lw=0.5, ls='--', c='darkgrey')
        ax.axvline(x=-0.05, lw=0.5, ls='--', c='darkgrey')
        
        
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylabel("Inequality\n(Gini coef. of entire rank distribution)")
    ax.set_xlabel("Inequity\n(Mean error of fraction of minorities across all top-k\% rank)")
    
    #ax.set_title(metric.upper())
    ax.set_ylim((0-0.03,1+0.03))
    ax.set_xlim((mini-0.03,1+0.03))
        
    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
        
    plt.show()
    plt.close()

        
def plot_inequalities_symmetric(df, models, markers, mean=True, metric='pagerank', fn=None):

    ### data
    h = [0.2,0.5,0.8]
    fm = [0.1, 0.3, 0.5]
    df_sym = df.query("rank == 5 & hmm == hMM & metric == @metric & hmm in @h & fm in @fm").copy()
    df_sym.rename(columns={"hmm":"h"}, inplace=True)

    ### callback
    def inequality_plot_sym(x, y, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        if mean:
            data = data.groupby(['fm','h','kind']).mean().reset_index()
        data.plot.scatter(x=x, y=y, ax=ax, grid=False, s=150, **kwargs)

    ### plot
    fg = sns.FacetGrid(df_sym, col="fm", row="h", 
                       hue="kind", hue_order=models, hue_kws=dict(marker=markers),
                       height=1 if df_sym.h.nunique()==11 else 2, aspect=1, margin_titles=True, dropna=False)
    fg = fg.map_dataframe(inequality_plot_sym, "mae", "gini")
    fg.add_legend(title='Model')

    ### visuals
    for ax in fg.axes.flatten():
        for i in np.arange(0.0,1.0+0.1,0.1):
            i = round(i,1)
            ax.axhline(y=i, lw=0.5 if i!=0.5 else 1, ls='--', c='lightgrey' if i!=0.5 else 'black')
            ax.axvline(x=i, lw=0.5 if i!=0.5 else 1, ls='--', c='lightgrey' if i!=0.5 else 'black')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim((0-0.03,1+0.03))
    fg.axes[int(df_sym.fm.nunique()/2),0].set_ylabel("Vertical Inequality\n(Gini of rank distribution)")
    fg.axes[-1,int(df_sym.h.nunique()/2)].set_xlabel("Horizontal Inequality\n(MAE fraction of minorities in top-k rank)")

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

        
def plot_inequalities_asymmetric(df, models, markers, mean=True, metric='pagerank', fn=None):

    ### data
    hmm = [0.0, 0.5, 1.0]
    hMM = [0.0, 0.2, 0.5, 0.8, 1.0]
    fm = [0.1, 0.3, 0.5]
    data = df.query("rank == 5 & hmm in @hmm & hMM in @hMM & fm in @fm & metric == @metric").copy()

    ### callback
    def inequality_plot_asymm(x, y, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        c = kwargs.pop("color")
        if data.hMM.unique() == 0.5:
            c = "grey"

        for i,model in enumerate(models):
            tmp = data.query("kind==@model")
            if mean:
                tmp = tmp.groupby(['fm','hmm','hMM']).mean().reset_index()

            if tmp.shape[0]>0:
                tmp.plot.scatter(x=x, y=y, ax=ax, grid=False, c=c, marker=markers[i], s=150, **kwargs)

    ### plot
    palette = BrBG_11 if data.hMM.nunique() == 11 else BrBG_5
    fg = sns.FacetGrid(data, col="fm", row="hmm", hue="hMM",dropna=False,
                       height=2, aspect=1, margin_titles=True, palette=palette.mpl_colors)
    fg = fg.map_dataframe(inequality_plot_asymm, "mae", "gini")
    fg.add_legend()

    ### visuals
    for ax in fg.axes.flatten():
        for i in np.arange(0.0,1.0+0.1,0.1):
            i = round(i,1)
            ax.axhline(y=i, lw=0.5 if i!=0.5 else 1, ls='--', c='lightgrey' if i!=0.5 else 'black')
            ax.axvline(x=i, lw=0.5 if i!=0.5 else 1, ls='--', c='lightgrey' if i!=0.5 else 'black')
        ax.set_ylim((0-0.03,1+0.03))
        ax.set_xlabel("")
        ax.set_ylabel("")
    fg.axes[1,0].set_ylabel("Vertical Inequality\n(Gini of rank distribution)")
    fg.axes[-1,int(data.fm.nunique()/2.)].set_xlabel("Horizontal Inequality\n(MAE fraction of minorities in top-k rank)")

    ### second legend
    h = [plt.plot([],[], color="black", marker=m, ls="")[0] for m,l in zip(*[markers,models])]
    fg.axes[-1,int(data.fm.nunique()/2.)].legend(handles=h, labels=models, bbox_to_anchor=(-1.0,3.4,3,1), 
                                          loc="lower left", title="Model", frameon=False, 
                                          mode='expand',ncol=len(models), borderaxespad=0)

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
        
def plot_synthetic_quadrant_homophily(df_rank, qtype='qae', model=None, metric='pagerank', fn=None):
    '''
    var  qtype: quadrant type, qae (absolute error [0,1]) qe (error [-1,1])
    '''
    def facet_heatmap_quadrant_homo(data, color, **kwargs):
        n = kwargs['vmax'] # quadrants
        ROM = ['I','II','III','IV','V','VI','VII','VIII','IX']
        ROM = ROM[:n]
        
        ax = plt.gca()
        tmp = data.pivot_table(index='hMM', columns='hmm', values=qtype, aggfunc=lambda x: x.mode().iat[0])
        #print(tmp)
        #cmap = sns.color_palette("Paired", 6 if qtype == 'qe' else 4)
        #if qtype == 'qe':
        #    colors = [cmap[1],cmap[3],cmap[5]]
        
        # individual ineq: high, low
        #cmap = sns.color_palette("Paired", 6 if qtype == 'qe' else 4) 
        #if qtype == 'qe':
        #    colors = [cmap[1],cmap[3],cmap[5],cmap[4],cmap[2],cmap[0]]
        
        # individual ineq: high, medium, low
        if qtype == 'qe':
            colors = Blues_3.mpl_colors + Reds_3.mpl_colors + Greens_3.mpl_colors
            colors = [colors[5],colors[8],colors[2],
                      colors[4],colors[7],colors[1],
                      colors[3],colors[6],colors[0]]
            colors = colors[:n]
            
        ax = sns.heatmap(tmp, cmap=colors, fmt = '', **kwargs)
        # modify colorbar:
        colorbar = ax.collections[0].colorbar 
        colorbar.ax.set_ylabel("Disparity", rotation=270, labelpad=10)  #Region
        r = colorbar.vmax - colorbar.vmin 
        colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
        colorbar.set_ticklabels(ROM[:n])  
        # change order
        colorbar.ax.invert_yaxis()
        
    from org.gesis.lib import paper
    
    data = df_rank.query("kind==@model & metric==@metric & rank==100").copy()
    important_cols = ['kind', 'metric', 'fm', 'hMM', 'hmm', 'gini', 'mae', 'me'] # global, using the whole ranking
    data.drop(columns=[c for c in data.columns if c not in important_cols], inplace=True)
    data.loc[:,'qe'] = data.apply(lambda row: paper.get_quadrant_error(row,herror='me',verror='gini'), axis=1)
    data.loc[:,'qae'] = data.apply(lambda row: paper.get_quadrant_absolute_error(row,herror='mae',verror='gini'), axis=1)
    
    htype = 'mae' if qtype == 'qae' else 'me'
    vmin, vmax = data[qtype].min(), data[qtype].max()
    
    col = 'fm'
    fg = sns.FacetGrid(data=data, col=col, margin_titles=True, height=2.5, aspect=0.8)
    cbar_ax = fg.fig.add_axes([.99, .3, .02, .4])
    fg.map_dataframe(facet_heatmap_quadrant_homo, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)

    nc = data[col].nunique()
    nr = 1
    hmms = sorted(data['hmm'].unique())
    for k,ax in enumerate(fg.axes.flatten()):
        r, c = int(k/nc), k%nc
        ax.set_ylabel(r"$h_{MM}$" if c==0 and r==int(nr/2.) else '')
        ax.set_xlabel(r"$h_{mm}$" if r==nr-1 and c==int(nc/2) else '')
        
        if ax.get_title() != '':
            ax.set_title(ax.get_title().replace("fm",r"$f_{m}$"))
        
        ax.set_xticklabels([xtl.get_text() if i%2==0 else '' for i,xtl in enumerate(ax.get_xticklabels())], rotation=0)
        
        if k==0:
            ax.set_yticklabels([xtl.get_text() if i%2==0 else '' for i,xtl in enumerate(ax.get_yticklabels())], rotation=0)

    ### invert y axis
    plt.gca().invert_yaxis()
    
    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
        
    plt.show()
    plt.close()
        
def plot_synthetic_quadrants(df_rank, qtype='qae', model=None, metric='pagerank', all=False, fn=None):
    '''
    var  qtype: quadrant type, qae (absolute error [0,1]) qe (error [-1,1])
    '''
    
    def facet_heatmap_quadrant(data, color, **kwargs):
        n = kwargs['vmax'] # quadrants
        ROM = ['I','II','III','IV','V','VI','VII','VIII','IX']
        
        ax = plt.gca()
        tmp = data.pivot_table(index='hMM', columns='rank', values=qtype, aggfunc=lambda x: x.mode().iat[0])
        tmp_dir = data.pivot_table(index='hMM', columns='rank', values='dir', aggfunc=lambda x: x.mode().iat[0])
        
        if qtype == 'qe':
            ### when error is signed (+ or -)
            tmp_dir.replace(['+','-','='], '', inplace=True)
            
        # individual ineq: high, low
        #cmap = sns.color_palette("Paired", 6 if qtype == 'qe' else 4) 
        #if qtype == 'qe':
        #    colors = [cmap[1],cmap[3],cmap[5],cmap[4],cmap[2],cmap[0]]
        
        # individual ineq: high, medium, low
        if qtype == 'qe':
            colors = Blues_3.mpl_colors + Reds_3.mpl_colors + Greens_3.mpl_colors
            colors = [colors[5],colors[8],colors[2],
                      colors[4],colors[7],colors[1],
                      colors[3],colors[6],colors[0]]
        
        ax = sns.heatmap(tmp, cmap=colors, fmt = '', **kwargs)
        # modify colorbar:
        colorbar = ax.collections[0].colorbar
        colorbar.ax.set_ylabel("Disparity", rotation=270, labelpad=10)  #Region
        r = colorbar.vmax - colorbar.vmin 
        colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
        colorbar.set_ticklabels(ROM[:n])  
        # change order
        colorbar.ax.invert_yaxis()

    hmm = [0.1, 0.5, 0.9]
    hMM = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fm = [0.1, 0.3, 0.5]
    if all:
        hmm = hMM
        fm = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    data = df_rank.query("kind==@model & metric==@metric & hmm in @hmm & hMM in @hMM & fm in @fm").copy()
    data.drop(columns=['dataset'], inplace=True)
    vmin, vmax = data[qtype].min(), data[qtype].max()
    
    col = 'fm'
    row = 'hmm'
    fg = sns.FacetGrid(data=data, col=col, row=row, margin_titles=True, height=2, aspect=1)
    cbar_ax = fg.fig.add_axes([.99, .3, .02, .4])
    fg.map_dataframe(facet_heatmap_quadrant, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)

    nc = data[col].nunique()
    nr = data[row].nunique() 
    ranks = sorted(data['rank'].unique())
    
    for k,ax in enumerate(fg.axes.flatten()):
        r, c = int(k/nc), k%nc
        ax.set_ylabel(r"$h_{MM}$" if c==0 and r==int(nr/2.) else '')
        ax.set_xlabel("Top-k\% rank" if r==nr-1 and c==int(nc/2) else '')
        if r == nr-1:
            ax.set_xticks([r+0.5 for r in np.arange(len(ranks))])
            ax.set_xticklabels([r if r in [10,50,90] else '' for r in ranks], rotation=0)

    ### invert y axis
    plt.gca().invert_yaxis()
    
    ### right-ylabel
    #[plt.setp(ax.texts, text="") for ax in fg.axes.flat] 
    #fg.set_titles(row_template = row + ' = {row_name}', bbox=dict(boxstyle='square,pad=-0.3', fc="white", ec="none")) 
    for ax in fg.axes.flatten():
        if ax.get_title() != '':
            ax.set_title(ax.get_title().replace("fm",r"$f_{m}$"))
            
        if ax.texts:
            txt = ax.texts[0]
            
            if txt.get_text() != '':    
                ax.text(txt.get_unitless_position()[0]+0.01, 
                        txt.get_unitless_position()[1],
                        txt.get_text().replace("hmm",r"$h_{mm}$"),
                        transform=ax.transAxes,
                        rotation=270,
                        va='center')
                ax.texts[0].remove()
                
    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    
    ### Save fig
    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()

def plot_synthetic_rankings(df_rank, model=None, metric='pagerank', y='fmt', sym=True, fn=None):
    
    if y not in ['fmt','gt']:
        raise Exception('Invalid x-axis (horizontal ineq.)')
        
    col = 'fm'
    row = 'hmm'
    
    fm = [0.1, 0.3, 0.5]
    data = df_rank.query("metric == @metric & fm in @fm").copy()
    if model is not None:
        data = data.query("kind == @model")
    
    ### Type of homophily: symmetric or not
    if sym:
        data = data.query("hmm == hMM").copy()
        colors = BrBG_11.mpl_colors
        colors[int(len(colors)/2)] = 'lightgrey'
        fg = sns.catplot(data=data,
                         col=col,
                         hue='hMM',
                         x='rank', y=y,
                         kind='point',
                         sharey=True,
                         height=2.5, aspect=1,
                         legend=True,
                         legend_out=True,
                         palette=colors
                         )
    else:
        hm = [0.2, 0.5, 0.8]
        hM = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        palette = BrBG_11 if len(hM)==11 else BrBG_5
        colors = palette.mpl_colors
        colors[int(len(colors)/2)] = 'lightgrey'
        data = data.query("hmm in @hm and hMM in @hM").copy()
        fg = sns.catplot(data=data,
                         col=col,
                         row=row,
                         hue='hMM',
                         margin_titles=True,
                         x='rank', y=y,
                         kind='point',
                         sharey=True,
                         height=2, aspect=1,
                         legend=True,
                         legend_out=True,
                         palette=colors)

    ### baseline: fm
    ncol = data[col].nunique()
    nrow = 1 if sym else data[row].nunique()
    for i, ax in enumerate(fg.axes.flatten()):
        # labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # xticks
        ax.set_xticklabels([int(float(xtl.get_text())) 
                            if int(float(xtl.get_text())) in [10, 50, 90] else '' for xtl in ax.get_xticklabels()], 
                           rotation=0)

        # baseline
        try:
            r = int(i / ncol)
            c = i - (r * ncol)
            
            if y == 'fmt':
                fm = float(fg.axes[0,c].get_title().replace("fm = ","").replace(" ",""))
            else:
                fm = 0
            ax.axhline(fm, c='black', ls='--', lw=2.0, zorder=1000)
        except:
            pass

    ### labels
    fg.axes[-1,int(ncol/2)].set_xlabel('Top-k rank %')
    ylabel = 'Fraction of minorities' if y == 'fmt' else 'Gini'
    fg.axes[int(fg.axes.shape[0]/2),0].set_ylabel('{} in Top-k rank %'.format(ylabel))

    ### legend
    if sym:
        fg._legend.set_title("h")

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
        
    plt.show()
    plt.close()

        
def plot_vh_inequalities_synthetic(df_rank, metric='pagerank', sym=True, fn=None):

    ### validation
    VALID_METRICS = ['pagerank','wtf']
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
                axes[col].scatter(y=data.gini.values, x=data.mae.values, label=hMM, color=colors[hue], marker='x')
            axes[0].legend(loc='lower left', title='homophily',
                           bbox_to_anchor=(-0.04, 1.12, ncols*1.075, 0.2), mode='expand',
                           ncol=tmp.hMM.nunique(), handletextpad=0.05, frameon=False)
            # plt.legend(loc='center left', title='h', bbox_to_anchor=(1, 0.5))
        else:
            axes[0, col].set_title("fm={}".format(fm))

            ### ylabel (right)
            for row, hmm in enumerate(tmp.hmm.unique()):
                if col == ncols - 1:
                    s = 'hmm={}'.format(hmm)
                    axes[row, col].text(s=s,
                                        y=0.67,
                                        x=(df_rank.mae.max()-0.04) , rotation=-90)

                ### scatter plot
                for hue, hMM in enumerate(tmp.hMM.unique()):
                    data = tmp.query("fm==@fm & hmm==@hmm & hMM==@hMM").copy()
                    axes[row, col].scatter(y=data.gini.values, x=data.mae.values, label=hMM, color=colors[hue], marker='x')

            axes[0,1].legend(loc='lower left', title='hMM',
                             bbox_to_anchor=(-0.26, 1.12, 1.5, 0.2), mode='expand',
                             ncol=tmp.hMM.nunique(), handletextpad=0.05, frameon=False)
            #axes[0, 2].legend(loc='lower left', title='hMM'

    ### ylabel (left)
    ylabel = 'Gini coefficient'
    row = int(nrows / 2)
    ax = axes[0] if sym else axes[row,0]
    if nrows % 2 != 0:
        ax.set_ylabel(ylabel)
    else:
        ax.text(0.64 if sym else -0.25 , #100,
                0.65 if sym else 0.8, #0.65,
                ylabel, {'ha': 'center', 'va': 'center'}, rotation=90)

    ### xlabel
    xlabel = 'MAE of fraction of minorities in top-k%'
    col = int(ncols / 2)
    ax = axes[-1,col] if not sym else axes[col]
    if ncols % 2 != 0:
        ax.set_xlabel(xlabel)
    else:
        ax.text(0.35,
               -0.08 if not sym else -0.1,
                xlabel, {'ha': 'center', 'va': 'center'}, rotation=0)

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
        
        
################################################################################
# Special handlers
################################################################################

def _get_mean_val_from_df(df_emp, att, row):
    s = "dataset=='{}' & metric=='{}' & rank=={}".format(row.dataset, row.metric, row['rank'])
    return df_emp.query(s)[att].mean()
     
def feature_importance(data, model, metric, kfold, fn=None):
    fig,axes = plt.subplots(2,2,figsize=(5,5))
    df_summary = pd.DataFrame(columns=['kind','output','r2mean','r2std','feature','importance'])

    for r,local in enumerate([False,True]):
        for c,output in enumerate(['gini','error']):
            
            if local:
                df = data.query("kind==@model & metric==@metric").copy() #local
                y = 'efmt' if output == 'error' else 'gt'
                features = ['fm','hMM','hmm','rank','random', y]
            else:
                df = data.query("rank==5 & kind==@model & metric==@metric").copy() #global
                y = 'me' if output == 'error' else 'gini'
                features = ['fm','hMM','hmm','random', y]

            df.loc[:,'random'] = np.random.random(size=df.shape[0])
            df = df[features]
            scaler = MinMaxScaler(feature_range=(0, 1))
            Z = scaler.fit_transform(df)

            X = Z[:,:-1]
            y = Z[:,-1]

            ### model performance
            rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
            r2s = cross_val_score(rf, X, y, cv=kfold)
            preds = cross_val_predict(rf, X, y, cv=kfold)
            
            axes[r,c].scatter(y, preds, edgecolors=(0, 0, 0))
            axes[r,c].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
            axes[r,c].text(s='R2={:.2f}\n({:.4f})'.format(np.mean(r2s), np.std(r2s)),x=0,y=0.8,zorder=100)
            axes[r,c].set_xlabel('Measured' if r==1 else '')
            axes[r,c].set_ylabel('Predicted' if c==0 else '')
            axes[r,c].set_title("{} {}".format('Local' if local else 'Global', output.title()))
            
            ### feature importance
            cv = cross_validate(rf, X, y, cv=kfold, scoring = 'r2', return_estimator =True)
            tmp = pd.DataFrame(columns = features[:-1])
            for idx,estimator in enumerate(cv['estimator']):
                tmp = tmp.append(pd.Series(estimator.feature_importances_,index=features[:-1]), ignore_index=True)

            tmp = tmp.mean().sort_values(ascending=False)            
            df_summary = df_summary.append({'kind':'local' if local else 'global',
                                            'output':output,
                                            'r2mean':round(np.mean(r2s),2),
                                            'r2std':round(np.std(r2s),4),
                                            'feature':tmp.index.values,
                                            'importance':tmp.values.round(2)
                                           }, ignore_index=True)
            
    ### space between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    ### Save fig
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
        
    plt.show()
    plt.close()
    return df_summary


################################################################################ 
# OLS (deprecated)
################################################################################

#deprecated
def OLS_observed_predicted(data, result, smooth=0, x='mae', fn=None):
    fig, ax = plt.subplots()
    
    # Plot observed values
    ax.scatter(data[x], data['gini'], alpha=0.5, label='observed')

    # Plot predicted values
    ax.scatter(data[x], result.predict(), alpha=0.5, label='predicted')

    ax.legend()
    ax.set_title('OLS predicted values')
    ax.set_xlabel(x.title())
    ax.set_ylabel('Gini')
    
    ax.set_xlim(((-1 if x in ['efmt','me'] else 0)-0.03,1+0.03))
    ax.set_ylim((0-0.03,1+0.03))
    
    ax.axhline(y=0.5, lw=1, ls='--', c='black')
    mid = 0 if x in ['efmt','me'] else 0.5
    smooth = 0 if mid==0.5 else smooth
    if smooth == 0:
        ax.axvline(x=mid, lw=1, ls='--', c='black')
    else:
        ax.axvline(x=mid-smooth, lw=1, ls='--', c='black')
        ax.axvline(x=mid+smooth, lw=1, ls='--', c='black')

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
       
