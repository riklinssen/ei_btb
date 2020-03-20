
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
        heightval, fontsize='small', ha=ha[xpos], va='bottom', color='white', alpha=1)


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
        heightval, fontsize='small', ha=ha[xpos], va='bottom', color=bar.get_facecolor(), alpha=1)






##FILEPATHS

base_path = pathlib.Path.cwd()
data = base_path / "data"
graphs = base_path / "graphs"



#MAKE DATASETS
q12020= pd.read_pickle(data/"q1_2020_clean.pkl" )
q32019= pd.read_pickle(data/"q3_2019_clean.pkl" )
q12019= pd.read_pickle(data/"q1_2019_clean.pkl" )
longi= pd.read_pickle(data/"longitudinal.pkl" )



btbitems=['btb_fairtreatment', 'btb_fairtreatment_d', 'btb_livingwage', 'btb_livingwage_d', 'btb_wouldswitch', 'btb_wouldswitch_d']

for c in btbitems: 
    print(longi.groupby('surveysource')[c].value_counts(dropna=False))

#btb_fairtreatment: 
#In hoeverre ben je het eens of oneens met deze stellingen? Ik vind het belangrijk dat de supermarkt waar ik boodschappen doe er alles aan doet zodat werknemers in de hele productieketen eerlijk behandeld worden
#all polls (q4 2018, q12019, q32019, q12020)

#btb_livingwage
#Q8_3 In hoeverre ben je het eens of oneens met deze stellingen? Ik vind het belangrijk dat de supermarkt waar ik boodschappen doe er alles aan doet zodat werknemers loon verdienen waar zij met hun gezin van kunnen leven
#q12019, q32019, q12020

#btb_wouldswitch
#all polls (q4 2018, q12019, q32019, q12020)

longi['total']='total'

#surveydate
longi['surveydate']=longi['surveydate'].fillna('2020-02-27 00:00:00')
longi['surveydate']=pd.to_datetime(longi['surveydate'])

surveyitems=['btb_fairtreatment_d', 'btb_livingwage_d', 'btb_wouldswitch_d']
data_t=grouped_weights_statsdf(longi, surveyitems, ['surveydate','total'], 'wgprop')
data_t['date']=[c[0] for c in data_t.index.get_level_values('groups')]
#data_t['date']=data_t['date'].dt.to_period("Q")
data_t['group']=[c[1] for c in data_t.index.get_level_values('groups')]
data_t['err']=data_t['weighted mean']-data_t['lower bound']


data_mt=grouped_weights_statsdf(longi, surveyitems, ['surveydate','mentality_en'], 'wgprop')
data_mt['date']=[c[0] for c in data_mt.index.get_level_values('groups')]
data_mt['group']=[c[1] for c in data_mt.index.get_level_values('groups')]

mentalitynames=[m for m in segmentcolormap_en.keys() if  'Total' not in m]


idx = pd.IndexSlice


titledict = {'btb_fairtreatment_d': "It is important that my supermarket \nis comitted to a fair treatment of workers,\nby poll and segment",
             'btb_livingwage_d': "It is important that my supermarket \nis comitted to ensuring fair wages\nthroughout the value chain,\nby poll and segment",
             'btb_wouldswitch_d': "Likelihood of switching to supermarket \nmore comitted to fair treatment of workers"}



rows=len(mentalitynames)+1

#######btb items
for surveyitem in surveyitems: 
    filename=graphs/"{}_trend_by_segment.png".format(surveyitem)


    fig, axes = plt.subplots(nrows=rows, ncols=2, sharex='col', figsize=(4, 10))
    sns.set_style('white')

    axs = fig.axes


    #total:
    sel = data_t.loc[idx[surveyitem, :], :].set_index(['date'])


    axs[0].plot(sel.index, sel['weighted mean'], ls='-', marker='.', color='black')
    #annotations
    last_q = sel.index.max()
    #set annotate value
    percstr = str(int(round(sel.at[last_q, 'weighted mean']*100, 0))) + '%'
    percloc = (last_q, sel.at[last_q, 'weighted mean'])
    axs[0].annotate(s=percstr, xy=percloc, xytext=[5, -2],
                    textcoords='offset points', color='black', size='small')
    axs[0].set_title('Total', size='small')

    #by mentality
    for i, ment in zip(range(1, rows), mentalitynames):
        outc = data_mt.loc[idx[surveyitem, :], :]
        sel = outc.loc[outc['group'] == ment].set_index('date')
        kleur = segmentcolormap_en[ment]
        axs[i].plot(sel.index, sel['weighted mean'],
                    ls='-', marker='.', color=kleur)
        #set annotate value
        percstr = str(int(round(sel.at[last_q, 'weighted mean']*100, 0))) + '%'
        percloc = (last_q, sel.at[last_q, 'weighted mean'])
        axs[i].annotate(s=percstr, xy=percloc, xytext=[5, -2],
                        textcoords='offset points', color=kleur, size='small')
        axs[i].set_title(ment, size='small', color=kleur)


    #spines
    for i in range(0, len(axs)):
        axs[i].spines['left'].set_visible(True)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines["left"].set_position(("outward", +5))
        axs[i].spines["bottom"].set_position(("outward", +5))
        axs[i].tick_params(axis='x', bottom=True)
        axs[i].tick_params(axis='y', left=True)

    #all axes
    for ax in axs:
        ax.set_ylim((0, 1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    plt.subplots_adjust(top=0.4)


    fig.autofmt_xdate()
    fig.tight_layout()

    fig.suptitle(titledict[surveyitem], size=14, weight='bold', y=1.07, color='black')
    nrobs=str(sel.tot_n_unweigthed[0])
    #footnotes
    if surveyitem=='btb_wouldswitch_d':
        plt.figtext(0,-0.05,'% of respondents (very) likely to switch supermarket \n by poll & segment\nSource: Quarterly polls Q4-2018-Q1-2020\nntotal='+ nrobs,fontsize='small', ha='left')
    else: 
        plt.figtext(0,-0.05,'% of respondents agree or completely agree \nby poll & segment\nSource: Quarterly polls Q4-2018-Q1-2020\nntotal='+ nrobs,fontsize='small', ha='left')
    
    fig.savefig(filename, dpi=600, facecolor='w', bbox_inches='tight')
    fig.show()


##




# by segment allpolls

########################################

data_mt=grouped_weights_statsdf(longi, surveyitems, 'mentality_en', 'wgprop')
data_mt['err']=data_mt['weighted mean']-data_mt['lower bound']
data_mt['color']=data_mt.index.get_level_values('groups').map(segmentcolormap_en)

data_t=grouped_weights_statsdf(longi, surveyitems, 'total', 'wgprop')



##by segment

titledict_s={k: v.replace('by poll and segment', 'by segment') for (k,v) in titledict.items()}

for surveyitem in surveyitems: 
    filename=graphs/"{}_by_segment.png".format(surveyitem)

    fig, axes = plt.subplots(nrows=1, ncols=2, sharey='col', gridspec_kw={'width_ratios':[1,8], 'wspace':0.3},  figsize=(4,3))
    axs = fig.axes

    sel=data_t.loc[idx[surveyitem, :], :].droplevel('outcome')


    axs[0].bar(x=sel.index, height=sel['weighted mean'], color='black')
    axs[0].set_title('Total', size='small')


    
    sel_mt=data_mt.loc[idx[surveyitem,:], :].droplevel('outcome').sort_values(by='weighted mean', ascending=False)
    axs[1].bar(x=sel_mt.index, height=sel_mt['weighted mean'], color=sel_mt['color'])
    axs[1].set_title('by segment', size='small')
    axs[1].get_yaxis().set_visible(False)

    #annotate
    for ax in axs: 
        autolabelpercenttop(ax)
        ax.set_ylim((0, 1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

        # spines #ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines["left"].set_position(("outward", +5))
        ax.spines["bottom"].set_position(("outward", +5))
        ax.tick_params(axis='y', left=True)

    for tick, color in zip(axs[1].get_xticklabels(),sel_mt['color']) : 
        tick.set_color(color)
        tick.set_rotation(90)

    fig.suptitle(titledict_s[surveyitem], size='large', weight='bold', y=1.15, x=0, ha='left',  color='black')
    nrobs=str(sel.tot_n_unweigthed[0])
    #footnotes
    if surveyitem=='btb_wouldswitch_d':
        plt.figtext(0,-0.6,'% of respondents (very) likely to switch supermarket \n by poll & segment\nSource: Quarterly polls Q4-2018-Q1-2020 combined\nn total='+ nrobs,fontsize='small', ha='left')
    else: 
        plt.figtext(0,-0.6,'% of respondents agree or completely agree \nby poll & segment\nSource: Quarterly polls Q4-2018-Q1-2020\nn total='+ nrobs,fontsize='small', ha='left')
    
    fig.savefig(filename, dpi=600, facecolor='w', bbox_inches='tight')
    fig.show()

