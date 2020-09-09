import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import powerlaw
import numpy as np
import collections
import seaborn as sns
from itertools import cycle
from matplotlib.lines import Line2D
from palettable.colorbrewer.diverging import BrBG_11
from palettable.colorbrewer.diverging import BrBG_5
from palettable.colorbrewer.qualitative import Paired_11, Set2_8, Set1_9, Set3_8, Accent_8

from org.gesis.lib import utils
from org.gesis.lib import graph

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
    #fg.ax.set_ylim(0,0.5)
    
    

############################################################################################################
# All datasets
############################################################################################################


###########################################################
# Degree dist
###########################################################
def plot_degree_distributions_groups_fit(df_summary_empirical, df_metadata_empirical, df_metadata_fit, model='DBAH', fn=None):

    plt.close()

    ### main data
    metrics = ['indegree', 'outdegree']
    discrete = True
    labels = {0:'Majority', 1:'minority'}
    datasets = sorted(df_metadata_fit.dataset.unique())
    
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
                data_fit = df_metadata_fit.query("dataset.str.lower()==@dataset.lower() & minority==@minority")[metric].values.astype(np.float)

                ### Empirical:
                try:
                    label = '{} empirical'.format(labels[minority])
                    fit_emp = graph.fit_power_law(data_emp, discrete=discrete)
                    fit_emp.power_law.plot_pdf(ax=axes[row, col], linestyle='-', color=colors[minority], label=label)
                    txt_emp = txt_emp.replace("<min>" if minority else "<maj>", str(round(fit_emp.power_law.alpha,1)))
                except:
                    pass
                
                ### Model:
                try:
                    if data_fit.shape[0] > 0:                       
                        label = '{} {}'.format(labels[minority], model)
                        fit_mod = graph.fit_power_law(data_fit, discrete=discrete)
                        fit_mod.power_law.plot_pdf(ax=axes[row, col], linestyle='--', color=colors[minority], label=label)
                        txt_fit = txt_fit.replace("<min>" if minority else "<maj>", str(round(fit_mod.power_law.alpha)))
                except:
                    pass
                
            ### Exponents
            if row == 0:
                # indegree
                xye[metric] = {'aps': (13, 0.04), 'apsgender3': (120, 0.007), 'apsgender8': (100, 0.01), 
                               #'github': (500, 0.00055), 'pokec': (1600, 0.00001), 
                               'blogs': (100, 0.03), 'seventh': (40, 0.05), 'hate': (10, 0.05), 
                               'wikipedia': (35, 0.013)}
                xym[metric] = {'aps': (2, 0.0005), 'apsgender3': (20, 0.00002), 'apsgender8': (13, 0.00002), 
                               #'github': (12, 0.00000008), 'pokec': (140, 0.000000000008), 
                               'blogs': (25, 0.0004), 'seventh': (7, 0.005), 'hate': (3, 0.001), 
                               'wikipedia': (2, 0.00002)}

            else:
                # outdegree
                xye[metric] = {'aps': (25, 0.01), 'apsgender3': (60, 0.025), 'apsgender8': (180, 0.0025), 
                               #'github': (500, 0.0004), 'pokec': (1300, 0.00009), 
                               'blogs': (80, 0.004), 'seventh': (100, 0.01), 'hate': (100, 0.005), 
                               'wikipedia': (30, 0.05)}
                xym[metric] = {'aps': (4, 0.000008), 'apsgender3': (10, 0.00009), 'apsgender8': (7, 0.0000005), 
                               #'github': (30, 0.0000000095), 'pokec': (110, 0.000000001), 
                               'blogs': (20, 0.00005), 'seventh': (12, 0.0005), 'hate': (2, 0.000006), 
                               'wikipedia': (10, 0.0005)}

            ### Column name (dataset)
            axes[row, col].text(s=txt_emp, x=xye[metric][dataset.lower()][0], y=xye[metric][dataset.lower()][1])
            axes[row, col].text(s=txt_fit, x=xym[metric][dataset.lower()][0], y=xym[metric][dataset.lower()][1])

            ### y-label right
            if col == ncols - 1:
                xt = axes[row, col].get_xticks()
                yt = axes[row, col].get_yticks()
                axes[row, col].text(s=metric,
                                    x=580 if row == 0 else 78,
                                    y=0.01 if row == 0 else 0.01 , rotation=-90)

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
        axes[row, col].text(min(xt) * 110,
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
    
    
    
    
###########################################################
# Ranking 
###########################################################

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
    
    
def plot_inequalities(df, models, markers, vtype='mae', mean=False, metric="pagerank", empirical=None, fn=None):
    vat, mini, mid, color = setup_plot_HI_simplified(vtype)
    
    ### data
    data = df.query("metric == @metric").copy()
    if mean:
        data = data.groupby(['kind','fm','hmm','hMM']).mean().reset_index()
    data.sort_values([vtype,'gini'], inplace=True)

    ### color
    colors = Accent_8.mpl_colors
    colors = cycle(colors)
    
    ### plot
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    zorder = len(models)
    y = 'gini'
    for model,marker in zip(*(models,markers)):
        tmp = data.query("kind==@model")
        ax.scatter(x=tmp[vtype], y=tmp[y], color=next(colors), label=model, marker=marker, zorder=zorder)
        zorder-=1
    if empirical is not None:
        legend1 = ax.legend(title='Model', bbox_to_anchor=(1.04,1), borderaxespad=0)
    else:
        ax.legend(title='Model',bbox_to_anchor=(0.5,1.0), loc="upper right", ncol=2)
        
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
    for i in np.arange(mini,1.0+0.1,0.1):
        i = round(i,1)
        ax.axhline(y=i, lw=0.5 if i!=0.5 else 1, ls='--', c='lightgrey' if i!=0.5 else 'black')
        ax.axvline(x=i, lw=0.5 if i!=mid else 1, ls='--', c='lightgrey' if i!=mid else 'black')
        
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylabel("Vertical Inequality\n(Gini of rank distribution)")
    ax.set_xlabel("Horizontal Inequality\n(ME fraction of minorities in top-k rank)")
    
    #ax.set_title(metric.upper())
    ax.set_ylim((0-0.03,1+0.03))
    ax.set_xlim((mini-0.03,1+0.03))
        
    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
        

def plot_inequalities_fit_improved(df_best_fit, df_empirical, models, markers, valid_metrics=None, vtype='mae', fn=None):
    
    ### attribute for horizontal inequality
    _, mini, mid, _ = setup_plot_HI_simplified(vtype)
    label = vtype.upper() 
    
    ### datasets (hue1) and metrics (columns)
    datasets = sorted(df_best_fit.dataset.str.lower().unique())
    metrics = sorted(df_best_fit.metric.unique()) 
    metrics = metrics if valid_metrics is None else [m for m in metrics if m in valid_metrics]
    
    ### init plot
    ncol = len(metrics)
    nrow = 1
    colors = cycle(Set1_9.mpl_colors)
    x, y = vtype, 'gini'
    xmin, xmax = -1,1 #df_best_fit[x].min(), df_best_fit[x].max()
    ymin, ymax = 0,1 #df_best_fit[y].min(), df_best_fit[y].max()
    fig,axes = plt.subplots(nrow, ncol, figsize=(3*ncol, 3), sharey=True, sharex=True)
    
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
                axes[c].axhline(y=i, lw=0.5 if i!=0.5 else 1, ls='--', c='#FAF8F7' if i!=0.5 else 'black', zorder=0)
                axes[c].axvline(x=i, lw=0.5 if i!=mid else 1, ls='--', c='#FAF8F7' if i!=mid else 'black', zorder=0)
                
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
        
def _get_mean_val_from_df(df_emp, att, row):
    s = "dataset=='{}' & metric=='{}' & rank=={}".format(row.dataset, row.metric, row['rank'])
    return df_emp.query(s)[att].mean()
    #df_empirical.query("dataset=='{}' & rank=={}".format(row.dataset,row['rank'])).gt.mean()
     
    
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


###########################################################
# OLS
###########################################################

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
        
        
###########################################################
# Ranking Empirical
###########################################################

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

#             d = tmp_m.mae.unique()[0]
#             _ = axes[row, col].text(s='$MAE='+str(round(d, 3))+'$',
#                                      x=0,  # tmp['rank'].values[1],
#                                      y=0.9 if metric == metrics[0] else 0.8, color=colors[i])

            d = tmp_m.me.unique()[0]
            _ = axes[row, col].text(s='$ME='+str(round(d, 3))+'$',
                                     x=0,  # tmp['rank'].values[1],
                                     y=0.9 if metric == metrics[0] else 0.8, color=colors[i])
        
            axes[row, 0].set_ylabel('Fraction of minorities\nin Top-k%')
            axes[row, col].set_xlabel('')
            axes[row, col].xaxis.set_major_locator(plt.MaxNLocator(3))
            axes[row, col].xaxis.set_minor_locator(plt.MaxNLocator(10))
            axes[row, col].set_xlim(0 - 0.05, 1 + 0.05)

    ### xlabel
    xlabels = ['Bottom-k% of nodes', 'Rank k%']
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
    
    

    
###########################################################
# Ranking Fit
###########################################################

def plot_vh_inequalities_fit(df_rank, x='mae', group=False, all=True, fn=None):

    if x not in ['mae','me']:
        raise Exception('Invalid x-axis (horizontal ineq.)')
        
    if all:
        kind = ['empirical','DH','DBA','DBAH','Random','SBM']
    else:
        kind = ['empirical', 'DBAH']

    ### only main data points
    metrics = ['pagerank', 'wtf']
    tmp = df_rank.query("kind in @kind & metric in @metrics").copy()
    tmp = tmp.groupby(['dataset','kind','metric','epoch']).mean().reset_index()
    tmp.drop(columns=['rank', 'fmt'], inplace=True)
    tmp.loc[:,'dataset'] = tmp.loc[:,'dataset'].str.lower()
    
    if group:
        tmp = tmp.groupby(['dataset','kind','metric']).mean().reset_index()

    tmp.kind = tmp.kind.astype("category")
    tmp.kind.cat.set_categories(kind, inplace=True)
    tmp.sort_values(['kind','dataset','metric'], inplace=True)

    ### main plot
    nrows = tmp.metric.nunique()
    ncols = tmp.dataset.nunique()
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, 5), sharey=True, sharex=True)

    ### subplots
    colors = sns.color_palette() #"tab20")
    for col, dataset in enumerate(tmp.dataset.unique()):

        axes[0, col].set_title(dataset)

        for row, metric in enumerate(tmp.metric.unique()):

            ### y-label right
            if col == ncols-1:
                axes[row, col].text(s=metric,
                                    x=0.9, #0.28 if not group else 0.235,
                                    y=0.5+(len(metric)*0.018), 
                                    rotation=-90)
                                    
            for hue, kind in enumerate(tmp.kind.unique()):
                data = tmp.query("dataset==@dataset & metric==@metric & kind==@kind").copy()
                axes[row,col].scatter(y=data.gini.values, x=data[x].values, label=kind, color=colors[hue], 
                                      marker='x' if kind!='empirical' else 'o',
                                      zorder=1000 if kind == 'empirical' else 1)

    ### legend
    axes[0,-1].legend(bbox_to_anchor=(1.18,1), borderaxespad=0)

    
    ### baseline
    for ax in axes.flatten():
        ax.axhline(y=0.5, ls='--', color='lightgrey', lw=0.5)
        ax.axvline(x=0.5 if x=='mae' else 0, ls='--', color='lightgrey', lw=0.5)
    
    ### ylabel left
    ylabel = 'Gini coefficient'
    if nrows % 2 != 0:
        axes[int(axes.shape[0]/2), 0].set_ylabel(ylabel)
    else:
        axes[int(axes.shape[0] / 2), 0].text(-0.85 if not group else -0.85,
                                             1.1,
                                             ylabel, {'ha': 'center', 'va': 'center'}, rotation=90)

    ### xlabel
    xlabel = '{} of fraction of minorities in top-k%'.format(x.upper())
    if ncols % 2 != 0:
        axes[-1, int(axes.shape[1]/2)].set_xlabel(xlabel)
    else:
        axes[-1, int(axes.shape[1] / 2)].text(-0.05,
                                              0.05 if not group else 0.05,
                                              xlabel, {'ha': 'center', 'va': 'center'}, rotation=0)

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))


        
###########################################################
# Ranking Synthetic
###########################################################
        
def plot_synthetic_quadrante(df_rank, qtype='qae', model=None, metric='pagerank', all=False, fn=None):
    '''
    var  qtype: quadrant type, qae (absolute error [0,1]) qe (error [-1,1])
    '''
    
    def facet_heatmap_quadrant(data, color, **kwargs):
        n = kwargs['vmax'] # quadrants
        ROM = ['I','II','III','IV','V','VI']
        
        ax = plt.gca()
        tmp = data.pivot_table(index='hMM', columns='rank', values=qtype, aggfunc=lambda x: x.mode().iat[0])
        tmp_dir = data.pivot_table(index='hMM', columns='rank', values='dir', aggfunc=lambda x: x.mode().iat[0])

        if qtype == 'qe':
            ### when error is signed (+ or -)
            tmp_dir.replace(['+','-','='], '', inplace=True)
            
        cmap = cmap=sns.color_palette("deep", n)
        ax = sns.heatmap(tmp, cmap=cmap, annot=tmp_dir, fmt = '', **kwargs)
        # modify colorbar:
        colorbar = ax.collections[0].colorbar 
        colorbar.ax.set_ylabel("Quadrant", rotation=270, labelpad=10) 
        r = colorbar.vmax - colorbar.vmin 
        colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
        colorbar.set_ticklabels(ROM[:n])  
        # change order
        colorbar.ax.invert_yaxis()

    hmm = [0.0, 0.5, 1.0]
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
        ax.set_ylabel("hMM" if c==0 and r==int(nr/2.) else '')
        ax.set_xlabel("Top-k rank %" if r==nr-1 and c==int(nc/2) else '')
        if r == nr-1:
            ax.set_xticks([r+0.5 for r in np.arange(len(ranks))])
            ax.set_xticklabels([r if r in [10,50,90] else '' for r in ranks])

    ### invert y axis
    plt.gca().invert_yaxis()
    
    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    ### Save fig
    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
        

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
        hm = [0.0, 0.5, 1.0]
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
        
        

        
        

    
# import pylab
# pylab.rcParams['xtick.major.pad']='8'
# pylab.rcParams['ytick.major.pad']='8'
# #pylab.rcParams['font.sans-serif']='Arial'

# from matplotlib import rc
# rc('font', family='sans-serif')
# rc('font', size=10.0)
# rc('text', usetex=False)

# from matplotlib.font_manager import FontProperties

# panel_label_font = FontProperties().copy()
# panel_label_font.set_weight("bold")
# panel_label_font.set_size(12.0)
# panel_label_font.set_family("sans-serif")


# def plot_degree_powerlaw_fit(G_emp, G_fit, outdegree=True):
#     fig,axes = plt.subplots(1,2,figsize=(15,3))
#     color = ['blue','orange']
#     labels = ['Majority','minority']
#     title = '{}degree'.format('Out' if outdegree else 'In')
    
#     for k in [0,1]: #min vs maj
#         if outdegree:
#             data_emp = [o for n,o in G_emp.out_degree() if G_emp.node[n]['minority']==k]
#             data_fit = [o for n,o in G_fit.out_degree() if G_fit.node[n]['m']==k]
#         else:
#             data_emp = [o for n,o in G_emp.in_degree() if G_emp.node[n]['minority']==k]
#             data_fit = [o for n,o in G_fit.in_degree() if G_fit.node[n]['m']==k]

#         ax = axes[k]
#         emp = powerlaw.Fit(data_emp, discrete=True)
#         fig = emp.plot_ccdf(linewidth=3, color=color[k], linestyle='solid', label='Empirical ({})'.format(labels[k]), ax=ax)
#         emp.power_law.plot_ccdf(ax=ax, color=color[k], linestyle='dotted', label='Power-Law E.')
        
#         fit = powerlaw.Fit(data_fit, discrete=True)
#         fig = fit.plot_ccdf(color=color[k], linestyle='solid', label='Model', ax=ax)
#         fit.power_law.plot_ccdf(ax=ax, color=color[k], linestyle='dashdot', label='Power-Law M.')
#         ax.set_xlabel(title)
#         ax.legend(loc=3)
        
#     axes[0].set_ylabel(u"p(X≥x)")   
    

    
    #     data = df_fit.copy()
#     
#     data.loc[:,'dataset'] = data.apply(lambda row: [d for d in datasets if d.lower()==row.dataset][0] ,axis=1)
#     data = data.append(df_empirical, ignore_index=True)
#     if mean:
#         data = data.groupby(['dataset','kind','metric']).mean().reset_index()
#     else:
#         data = data.query("metric in @metrics & rank==5").copy()
#     data.sort_values(['mae','gini'], inplace=True)