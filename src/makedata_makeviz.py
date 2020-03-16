
# BTB impact measuremennt, stats from quarterly poll perception research
# Rik Linssen - March
# github repository here: www.github.com/riklinssen/ei_btb


#############IMPORTS########################
import numpy as np
import pandas as pd
import pathlib
import datetime
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.cm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
##########################BOILERPLATE##########
# Oxfam colors
hex_values = ['#E70052',  # rood
              '#F16E22',  # oranje
              '#E43989',  # roze
              '#630235',  # Bordeax
              '#53297D',  # paars
              '#0B9CDA',  # blauw
              '#61A534',  # oxgroen
              '#0C884A'  # donkergroen
              ]


colormap = {
    'Moderne burgerij': '#E70052',
    'Opwaarts mobielen': '#F16E22',
    'Postmaterialisten': '#E43989',
    'Nieuwe conservatieven': '#630235',
    'Traditionele burgerij': '#53297D',
    'Kosmopolieten': '#0B9CDA',
    'Postmoderne hedonisten': '#FBC43A',
    'Gemaksgeoriënteerden': '#BECE45',
}

mentalitytranslator = {'Moderne burgerij': 'Modern mainstream',
                       'Opwaarts mobielen': 'Social climbers',
                       'Postmaterialisten': 'Post-materialists',
                       'Nieuwe conservatieven': 'New conservatives',
                       'Traditionele burgerij': 'Traditionals',
                       'Kosmopolieten': 'Cosmopolitans',
                       'Postmoderne hedonisten': 'Post-modern hedonists',
                       'Gemaksgeoriënteerden': 'Convenience-oriented'}


segmentcolormap_en = dict(zip(
    [v for k, v in mentalitytranslator.items()],
    [v for k, v in colormap.items()]
))

#add total
segmentcolormap_en['Total']='000000'

# generate grouping into warm cold and peripheral audience
mentalityaudiencemap = {'Modern mainstream': 'Peripheral A',
                        'Social climbers': 'Cold',
                        'Post-materialists': 'Warm',
                        'New conservatives': 'Peripheral B',
                        'Traditionals': 'Peripheral A',
                        'Cosmopolitans': 'Warm',
                        'Post-modern hedonists': 'Peripheral B',
                        'Convenience-oriented': 'Cold'
                        }


mentalityaudiencemap_nl={'Moderne burgerij': 'Peripheral A',
                        'Opwaarts mobielen': 'Cold',
                        'Postmaterialisten': 'Warm',
                        'Nieuwe conservatieven': 'Peripheral B',
                        'Traditionele burgerij': 'Peripheral A',
                        'Kosmopolieten': 'Warm',
                        'Postmoderne hedonisten': 'Peripheral B',
                        'Gemaksgeoriënteerden': 'Cold'
                        }


# audiencecolormap
# colors hot/cold

# hot        #dd6e6e
# lauw       #ed8a38
# cold       #0f2491


# audience colors
audiencecolormap = dict(zip(
    ['Warm', 'Peripheral A',
        'Peripheral B', 'Cold'],
    ['#dd6e6e', '#ed8a38', '#eecd75', '#0f2491']
))

# weight funcs


def wavg_func(datacol, weightscol):
    def wavg(group):
        dd = group[datacol]
        ww = group[weightscol] * 1.0
        return (dd * ww).sum() / ww.sum()
    return wavg




def df_wavg(df, groupbycol, weightscol):
    grouped = df.groupby(groupbycol)
    df_ret = grouped.agg({weightscol: sum})
    datacols = [cc for cc in df.columns if cc not in [groupbycol, weightscol]]
    for dcol in datacols:
        try:
            wavg_f = wavg_func(dcol, weightscol)
            df_ret[dcol] = grouped.apply(wavg_f)
        except TypeError:  # handle non-numeric columns
            df_ret[dcol] = grouped.agg({dcol: min})
    return df_ret




def grouped_weights_statscol (df, statscol, groupbycol, weightscol):
    df.dropna(subset=[statscol], inplace=True)
    nrobs=len(df)
    grouped=df.groupby(groupbycol)
    stats={}
    means=[]
    lower=[]
    upper=[]
    groups=list(grouped.groups.keys())
    for gr in groups:
        stats=DescrStatsW(grouped.get_group(gr)[statscol], weights=grouped.get_group(gr)[weightscol], ddof=0)
        means.append(stats.mean)
        lower.append(stats.tconfint_mean()[0])
        upper.append(stats.tconfint_mean()[1])
    weightedstats=pd.DataFrame([means, lower, upper], columns=groups, index=['weighted mean', 'lower bound', 'upper bound']).T
    weightedstats['numberofobs']=nrobs
    return weightedstats

    #weightedstats=pd.DataFrame([means, lower, upper], index=groups)
    #return weightedstats





##try to generalize func towards more cols
def grouped_weights_statsdf(df, statscols, groupbycol, weightscol):
    """generates df with weighted means and 95% CI by groupbycol for cols in statscols
    

    Parameters
    ----------
    df : df
        df to be weigthed
    statscols : list
        cols/outcomes for weigthed stats
    groupbycol : str
        column name in df that defines groups 
    weightscol : str
        column name in df with weigths 
              
    
    Returns
    -------
    df
        multi-indexed df with outcome and groups as index
        stats generated: weighted mean, upper bound (95 CI), lower bound (95% CI), weighted n by group, total n unweighted

    """    
    alldata=pd.DataFrame()
    for c in statscols: 
        cdf=df.dropna(subset=[c])
        nrobs=len(cdf)
        grouped=cdf.groupby(groupbycol)
        stats={}
        means=[]
        lower=[]
        upper=[]
        nrobs_gr=[]
        groups=list(grouped.groups.keys())
        for gr in groups:
            stats=DescrStatsW(grouped.get_group(gr)[c], weights=grouped.get_group(gr)[weightscol], ddof=0)
            means.append(stats.mean)
            lower.append(stats.tconfint_mean()[0])
            upper.append(stats.tconfint_mean()[1])
            nrobs_gr.append(stats.nobs)          
        weightedstats=pd.DataFrame([means, lower, upper, nrobs_gr], columns=groups, index=['weighted mean', 'lower bound', 'upper bound', 'wei_n__group']).T
        weightedstats['tot_n_unweigthed']=nrobs
        weightedstats['outcome']=c
        weightedstats.index.name='groups'
        colstats=weightedstats.reset_index()
        colstats=colstats.set_index(['outcome', 'groups'])
        alldata=pd.concat([alldata, colstats])
               
    return alldata

def autolabelpercentmid(ax, xpos='center'):
    """
    Attach a text label above each bar (a percentage) in *ax*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
    bars=ax.patches
    for bar in bars:
        height = bar.get_height()
        heightval=str(int(bar.get_height()*100))+ '%'
        ax.text(bar.get_x() + bar.get_width()*offset[xpos], (0.5*height),
        heightval, fontsize=8, ha=ha[xpos], va='bottom', color='white', alpha=1)


def autolabelpercenttop(ax, xpos='center'):
    """
    Attach a text label above each bar (a percentage) in *ax*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
    bars=ax.patches
    for bar in bars:
        height = bar.get_height()
        heightval=str(int(bar.get_height()*100))+ '%'
        ax.text(bar.get_x() + bar.get_width()*offset[xpos], (1.01*height),
        heightval, fontsize=12, ha=ha[xpos], va='bottom', color=bar.get_facecolor(), alpha=1)






##FILEPATHS

base_path = pathlib.Path.cwd()
data = base_path / "data"
graphs = base_path / "graphs"



#MAKE DATASETS

