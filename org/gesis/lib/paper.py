################################################################################
# System dependencies
################################################################################            
import os
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

################################################################################
# Local dependencies
################################################################################
from org.gesis.lib import io


################################################################################
# Datasets summary
################################################################################

def all_datasets_summary_as_latex(df_summary, output=None):
    df_latex_summary = df_summary.pivot_table(columns='dataset', aggfunc=lambda x: ' '.join(str(v) for v in x))
    columns = ['N', 'class', 'min', 'maj', 'fm', 'd', 'plo_M','plo_m', 'EMM', 'EMm', 'EmM', 'Emm', 'hMM', 'hmm']
    df_latex_summary = df_latex_summary.reindex(columns)
    
    if output is not None:
        fn = os.path.join(output, 'summary_datasets.tex')
        df_latex_summary.to_latex(fn, float_format=lambda x: '%.5f' % x)
        print('{} saved!'.format(fn))
        
    return df_latex_summary

################################################################################
# Loading Data
################################################################################

def load_graphs(data, datasets):
    graphs = {}
    
    for dataset in datasets:
        fn = [fn for fn in os.listdir(os.path.join(data,dataset)) if fn.endswith(".gpickle")]
        graphs[dataset] = io.load_gpickle(os.path.join(data,dataset,fn))
        
    return graphs
     
def load_graph_fnc(data, dataset):
    fn = [fn for fn in os.listdir(os.path.join(data,dataset)) if fn.endswith(".gpickle")][0]
    return io.load_gpickle(os.path.join(data,dataset,fn))
    
def _load_model_metadata(files, path, model, dataset=None):
    df = None
    cols = None
    
    for fn in files:
        fn = os.path.join(path, model, fn) if dataset is None else os.path.join(path, model, dataset, fn)
        tmp = io.read_csv(fn)

        if dataset is not None:
            if 'dataset' not in tmp.columns:
                tmp.loc[:,'dataset'] = dataset

        if 'kind' not in tmp.columns:
            tmp.loc[:,'kind'] = model

        try:
            tmp.drop(['circle_of_trust'], axis=1, inplace=True)
        except: pass
        
        if df is None:
            df = tmp.copy()
            cols = df.columns
        else:
            df = df.append(tmp[cols], ignore_index=True)
            
    return df
                        
def _load_metadata(path, kind):
    
    if kind not in ['network','nodes']:
        raise Exception("kind does not exist")
    
    fn_final = os.path.join(path,'all_datasets_{}_metadata.csv'.format(kind))
    if os.path.exists(fn_final):
        return io.read_csv(fn_final)
    
    #if '/fit' in path or '/synthetic':
    if path.endswith("/fit") or path.endswith("/synthetic"):
        models = [d for d in os.listdir(path)]
    else:
        models = ['']
    
    cols = None
    df = None
    
    for model in models:
        
        # synthetic (all models)
        if '/synthetic' in path:
            files = [fn for fn in io.get_files(os.path.join(path, model), ext='.csv')
                    if not fn.endswith("_rank.csv") and not fn.endswith("_netmeta.csv")]
            print(len(files))
            tmp = _load_model_metadata(files, path, model)
        
        else:
            # empirical and fit (all models and datasets)
            datasets = [d for d in os.listdir(os.path.join(path,model)) if not d.endswith(".csv")]

            for dataset in datasets:
                files = io.get_files(os.path.join(path, model, dataset), prefix=kind, ext='_metadata.csv')

                if len(files) == 0 and '/fit' in path:
                    files = [fn for fn in io.get_files(os.path.join(path, model, dataset), ext='.csv')
                            if not fn.endswith("_rank.csv") and not fn.endswith("_netmeta.csv")]

                tmp = _load_model_metatada(files, path, model, dataset)
            
        if df is None:
            df = tmp.copy()
            cols = df.columns
        else:
            df = df.append(tmp[cols], ignore_index=True)
            
    #io.save_csv(df, fn_final)
    print(fn_final)
    return df
    
def load_network_metadata(path):
    return _load_metadata(path, 'network')

def load_node_metadata(path):
    return _load_metadata(path, 'nodes')

def _get_netmeta_from_fn(attribute, fn):
    #DBAH-N2000-fm0.1-d0.0015-ploM3.0-plom3.0-hMM0.0-hmm0.0-ID9_rank.csv
    #DBA-N2000-fm0.5-d0.0015-ploM3.0-plom3.0-ID9_rank.csv
    #DH-N2000-fm0.5-d0.0015-ploM3.0-plom3.0-hMM1.0-hmm1.0-ID9_rank.csv
    #Random-N2000-fm0.1-d0.0015-hMM0.5-hmm0.5-ID6_rank.csv
    #SBM-N2000-fm0.5-hMM0.5-hmm0.5-ID3_rank.csv

    is_synthetic = len(fn.split("/")[-1].split("-fm")[0].split("-"))==2
    model = fn.split("/")[-1].split("-fm")[0].split('-')[int(not is_synthetic)]
    if attribute in ['model','kind']:
        return model
    
    # special cases
    if attribute == 'd' and model in ["SBM","Null"]:
        return None
        
    if attribute in ['ploM','plom'] and model in ["Random","SBM","Null"]:
        return None

    if attribute in ['hMM','hmm'] and model in ["DPA","DUniform","Random","Null"]:
        return 0.5
    
    if attribute == 'epoch':
        return float(fn.split("-ID")[-1].split("_")[0])
    
    # cast to int
    if attribute in ['N','']:
        return int(fn.split("-{}".format(attribute))[-1].split("-")[0])
    
    # cast to float
    if attribute in ['fm','d','ploM','plom','hMM','hmm']:
        return float(fn.split("-{}".format(attribute))[-1].split("-")[0])

    return None
    
    
def load_rank(path, metadata=None, smooth=0.05):

    fn_final = os.path.join(path,'all_datasets_rank.csv')
    if os.path.exists(fn_final):
        #return io.read_csv(fn_final)
        df = io.read_csv(fn_final)
    else:
        datasets = [d for d in os.listdir(path) if not d.endswith(".csv")]

        df = None
        cols = None
        cols_netmeta = ['N','fm','hmm', 'hMM', 'plo_M', 'plo_m', 'd']

        for dataset in datasets:

            if dataset.lower() in ['pokec','github']:
                continue

            files = [fn for fn in os.listdir(os.path.join(path, dataset)) if fn.endswith(".csv") 
                     and (fn.startswith("rank_") or fn.endswith("_rank.csv"))]

            for fn in files:
                fn = os.path.join(path, dataset, fn)
                tmp = io.read_csv(fn)
                if 'dataset' not in tmp.columns:
                    tmp.loc[:,'dataset'] = dataset
                tmp = tmp.query("metric!='circle_of_trust'")

                for cnm in cols_netmeta:
                    if cnm not in tmp and metadata is not None:
                        tmp.loc[:,cnm.replace("_","")] = metadata.query("dataset.str.lower()==@dataset")[cnm].iloc[0]
                    elif cnm not in tmp and metadata is None:
                        tmp.loc[:,cnm.replace("_","")] = _get_netmeta_from_fn(cnm.replace("_",""), fn.split("/")[-1])

                if df is None:
                    df = tmp.copy()
                    cols = df.columns
                else:
                    df = df.append(tmp[cols], ignore_index=True)

    ### quadrants
    df = update_quadrants(df, smooth)
    
    ### sorting columns and casting
    df = _sort_cast_ranking_df(df)
    
    if not os.path.exists(fn_final):
        io.save_csv(df, fn_final)
    return df


def load_rank_all_models(path, models, smooth=0.05):
    df_rank_fit = None
    for model in models:
        tmp = load_rank(os.path.join(path,model), smooth=smooth)
        if df_rank_fit is None:
            df_rank_fit = tmp.copy()
        else:
            df_rank_fit = df_rank_fit.append(tmp, ignore_index=True)
    
    ### quadrants
    df_rank_fit = update_quadrants(df_rank_fit)
    
    ### sorting columns
    df_rank_fit = _sort_cast_ranking_df(df_rank_fit)
    
    return df_rank_fit
    
def load_rank_synthetic_all_models(path, models, smooth=0.05, update=False):
    df = None
    
    for model in models:
        tmp = _load_rank_synthetic(os.path.join(path,model), smooth, update)
        if df is None:
            df = tmp.copy()
        else:
            df = df.append(tmp, ignore_index=True)
    
    return df
            
def _load_rank_synthetic(path, smooth=0.05, update=False):
    fn_final = os.path.join(path,'all_networks_rank.csv')
    
    if os.path.exists(fn_final):
        df = io.read_csv(fn_final)
    else:
        df = None
        cols = None

        files = [fn for fn in os.listdir(os.path.join(path)) if fn.endswith(".csv") 
                 and (fn.startswith("rank_") or fn.endswith("_rank.csv"))]

        for fn in files:
            fn = os.path.join(path, fn)
            tmp = io.read_csv(fn)
            tmp = tmp.query("metric!='circle_of_trust'")
            tmp.drop(columns=["dataset"], inplace=True)

            for c in ['N', 'fm', 'd', 'ploM', 'plom', 'hMM', 'hmm', 'epoch']:
                tmp.loc[:,c] = _get_netmeta_from_fn(c, fn)

            if df is None:
                df = tmp.copy()
                cols = df.columns
            else:
                df = df.append(tmp[cols], ignore_index=True)

    # update (or not) quadrants
    if update:
        ### quadrants
        df = update_quadrants(df, smooth)

        ### sorting columns and casting
        df = _sort_cast_ranking_df(df)

        ### save
        io.save_csv(df, fn_final)
            
    return df

################################################################################
# Special Handlers
################################################################################

def best_fit(df_fit, df_empirical, datasets=None, models=None, vtype='mae'):
    ### best model 
    def _get_difference_fit(row,tmp_emp,vtype):
        dataset = row.dataset
        metric = row.metric
        tmp = tmp_emp.query("dataset==@dataset & metric==@metric")
        return abs(row['gini']-tmp.iloc[0]['gini']) + abs(row[vtype]-tmp.iloc[0][vtype])

    tmp_fit = df_fit.copy()
    
    if datasets is not None:
        tmp_fit = tmp_fit.query("dataset in @datasets")    
        
    if models is not None:
        tmp_fit = tmp_fit.query("kind in @models")
    
    tmp_fit = tmp_fit.groupby(['dataset','metric','kind']).mean().reset_index()
    tmp_emp = df_empirical.groupby(['dataset','metric','kind']).mean().reset_index()
    tmp_fit.loc[:,'dif'] = tmp_fit.apply(lambda row:_get_difference_fit(row,tmp_emp,vtype), axis=1)
    idx = tmp_fit.groupby(['metric','dataset']).apply(lambda row:row['dif'].argmin())
    tmp_fit = tmp_fit.loc[idx, :]
    return tmp_fit

def only_new_cases(df):
    tmp = df.query("kind in ['Random','DPAH']").copy()
    tmp.loc[:,'kind'] = tmp.apply(lambda row: 'DPAH2' if (row.ploM==2 and row.kind=='DPAH') else 'DPAH3' if (row.ploM==3 and row.kind=='DPAH') else row.kind, axis=1)
    tmp.loc[:,'kind'] = tmp.apply(lambda row: 'Random8' if (row.d==0.0008 and row.kind=='Random') else 'Random15' if (row.d==0.0015 and row.kind=='Random') else row.kind, axis=1)
    models = tmp.kind.unique().tolist()
    markers = ["*",".","x","v"]
    return tmp, models, markers

def remove_extra_synthetic(df):
    return df.query("(ploM==3 & kind!='Random') | (kind=='Random'&d==0.0015)")
    
def get_quadrant_error(row, beta=0.05, herror='efmt', verror='gt'):
    mid = 0.0
    
    ### 9 regions
    # abs(fmt) over-estimated
    if row[herror] > (mid+beta):
        return 3 if row[verror] >= 0.6 else 9 if row[verror] < 0.3 else 6
    
    # abs(fmt) under-estimated
    if row[herror] < (mid-beta):
        return 1 if row[verror] >= 0.6 else 7 if row[verror] < 0.3 else 4
    
    # abs(fmt) fair
    if row[herror] >= (mid-beta) and row[herror] <= (mid+beta):
        return 2 if row[verror] >= 0.6 else 8 if row[verror] < 0.3 else 5

    return 0

def get_quadrant_absolute_error(row, herror='aefmt', verror='gt'):
    mid = 0.5
    
    ### 6 regions (dont confuse with 6 regions from get_quadrant_error)###
    # abs(fmt) over-estimated
    if row[herror] >= (mid):
        return 2 if row[verror] >= 0.6 else 6 if row[verror] < 0.3 else 4
    
    # abs(fmt) under-estimated
    if row[herror] < (mid):
        return 1 if row[verror] >= 0.6 else 5 if row[verror] < 0.3 else 3
    
    return 0

def update_quadrants(df, beta=0.05):
    
    # error (deviation) 
    df.loc[:,'efmt'] = df.apply(lambda row: row['fmt']-row['fm'], axis=1) #pred-true: (+) over-estimated (-) under-estimated

    # absolute error (absolute deviation) 
    df.loc[:,'aefmt'] = df.apply(lambda row: abs(row['efmt']), axis=1)

    # direction (over/under representation of minorities)
    df.loc[:,'dir'] = df.apply(lambda row: '+' if row['efmt'] > (0+beta) 
                                           else '-' if row['efmt'] < (0-beta)
                                           else '=', axis=1)

    # quadrant (using absoulte error)
    df.loc[:,'qae'] = df.apply(lambda row: get_quadrant_absolute_error(row), axis=1)

    # quadrant (using error)
    df.loc[:,'qe'] = df.apply(lambda row: get_quadrant_error(row, beta), axis=1)
    
    return df
    
def _sort_cast_ranking_df(df):
    #,kind,metric,N,dir,
    #gini,mae,fmt,gt,fm,d,ploM,plom,hMM,hmm,efmr,aefmr,
    #epoch,rank,qae,qe
    
    ### casting to float
    cflo = ['fm','d','hMM','hmm','ploM','plom','fmt','gt','efmt','aefmt','gini','mae','me']
    df[cflo] = df[cflo].astype(np.float)
    
    ### casting to int
    if df['epoch'].isnull().any():
        df.loc[:,'epoch'] = -1
        
    cint = ['N','rank','epoch','qae','qe']    
    df[cint] = df[cint].astype(int)

    ### sorting columns
    cols = ['dataset','kind','metric','N','fm','d','ploM','plom','hMM','hmm','epoch',
            'rank','fmt','gt','efmt','qe','aefmt','qae','dir',
            'gini','mae','me']
    
    return df.loc[:,cols]

    
################################################################################
# Correlation
################################################################################

def spearmancor_inequality_inequity(df, model, metric, local=True):
    if local:
        tmp = df.query("kind==@model & metric==@metric").copy()
        x='gt'
        y='efmt'
    else:
        tmp = df.query("kind==@model & metric==@metric & rank==5").copy()
        x='gini'
        y='me'
    return stats.spearmanr(tmp[x].values, tmp[y].values)
    
    
    
################################################################################
# OLS: linear regression (deprecated)
################################################################################
#deprecated
def OLS_prepare_data(df, model, metric='pagerank', grouped=False):
    data = df.query("kind == @model & metric == @metric").copy()
    if grouped:
        data = data.groupby(['fm','hmm','hMM','kind','metric']).mean().reset_index()
    #data.loc[:,'pw'] = data.apply(lambda row: int(row.metric=='pagerank'), axis=1)
    #data.loc[:,'hMM-hmm'] = data.apply(lambda row: row.hMM-row.hmm, axis=1)
    #data.loc[:,'abs(hMM-hmm)'] = data.apply(lambda row: abs(row.hMM-row.hmm), axis=1)
    data.loc[:,'Intercept'] = 1
    return data

#deprecated
def OLS_best_model(data):
    # Create lists of variables to be used in each regression
    X = []
    X.append(['Intercept'])
    X.append(['Intercept','gini', 'fm', 'hmm', 'hMM'])
    X.append(['Intercept', 'fm'])
    X.append(['Intercept', 'fm', 'hmm'])
    X.append(['Intercept', 'fm', 'hmm', 'hMM'])
    
    # Estimate an OLS regression for each set of variables
    regs = []
    for xi in X:
        regs.append(sm.OLS(data['me'], data[xi], missing='drop').fit())
    
    # Summary
    info_dict={'Adj. R-squared' : lambda x: "{:.3f}".format(x.rsquared_adj),
               'R-squared' : lambda x: "{:.3f}".format(x.rsquared),
               'No. observations' : lambda x: "{:d}".format(int(x.nobs))}

    results_table = summary_col(results=regs,
                                float_format='%0.3f',
                                stars = True,
                                model_names=['Model {}'.format(i+1) for i,r in enumerate(regs)],
                                info_dict=info_dict,
                                regressor_order=['Intercept',
                                                 'pw',
                                                 'me',
                                                 'mae',
                                                 'gini',
                                                 'fm',
                                                 'hmm',
                                                 'hMM',
                                                ])
    results_table.add_title('OLS Regressions')
    
    # Best model
    rs = np.array([reg.rsquared for reg in regs])
    best = np.argmax(rs)
    best = regs[best]
    return results_table, best
