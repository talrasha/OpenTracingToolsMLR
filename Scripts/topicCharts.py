import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import date
import seaborn as sns

import scipy.stats as st
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy import stats
import statsmodels.api as sm
from scipy.stats import norm
import pylab
from scipy import stats
from scipy.stats import kstest, norm
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy import stats
from scipy.stats import norm
import pylab
from scipy.interpolate import interp1d
import math
from numpy import mean
from numpy import std

np.set_printoptions(linewidth=320)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns',25)

df = pd.read_csv('OpenTracingTools/df_info_new_with_topics.csv', lineterminator='\n')
topics = ['Setup','Instrumentation', 'Performance', 'Deployment/Scalability', 'Management',
          'Architecture', 'Adjusting', 'Usefulness', 'Inspection/Logging']
toolList = ['appdynamics', 'elasticapm', 'datadog', 'tanzu', 'stagemonitor', 'instana', 'skywalking', 'lightstep', 'zipkin', 'inspectit', 'jaeger']

def getTopicNumberListbyTool(thedf_info, toolname):
    theTooldf = thedf_info.loc[thedf_info['tool']==toolname,:]
    topiclist = []
    for i in range(9):
        topiclist.append(theTooldf.loc[theTooldf['dominantTopic'].str.contains(str(i)),:].shape[0])
    return topiclist

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def displayToolTopicPieChart(thedf, toolname):
    smaller_dim = (3.5, 3.5)
    labels = topics
    #licenselist = df['license'].values.tolist()
    counts = getTopicNumberListbyTool(thedf, toolname)
    #for item in labels:
    #    counts.append(len([x for x in licenselist if x==item]))
    #explode = (0, 0)
    fig, ax = plt.subplots(figsize=smaller_dim)
    ax.pie(counts, explode=None, autopct=lambda pct: func(pct, counts), shadow=False,
           startangle=90, radius=sum(counts)) #labels=labels,
    ax.axis('equal')
    plt.title(f"{toolname}")
    plt.savefig(f'{toolname}.png', bbox_inches='tight')
    plt.show()

def displayToolTopicBarChart():
    smaller_dim = (8, 4.5)
    labels = topics
    dfdata = pd.read_csv('temp.csv', index_col='Tool')
    fig, ax = plt.subplots(figsize=(14, 6))
    #ax.set_yticklabels(dfdata.Tool.values.tolist())
    dfdata.plot(kind='barh', stacked=True, ax=ax)
    plt.title(f"Topic Distribution for each Tool")
    plt.savefig(f'barchartstack.png', bbox_inches='tight')
    plt.show()

#for tool in toolList:
#    displayToolTopicPieChart(df, tool)
#displayToolTopicBarChart()

def displayToolTopicSentiment(thedf_info, toolname):
    #a4_dims = (11.7, 8.27)
    #smaller_dim = (8, 4.5)
    #fig, ax = plt.subplots(figsize=smaller_dim)
    theTooldf = thedf_info.loc[thedf_info['tool'] == toolname, :]
    sentslabels = ['Positive', 'Neutral', 'Negative']
    senttopicdict = {}
    senttopicdict['topic'] = topics
    for i in range(3):
        senttopicdict[sentslabels[i]] = []
    for j in range(9):
        senttopicdict[sentslabels[0]].append(theTooldf.loc[(theTooldf['com']>0) & (theTooldf['dominantTopic'].str.contains(str(j))), :].shape[0])
        senttopicdict[sentslabels[1]].append(theTooldf.loc[(theTooldf['com']==0) & (theTooldf['dominantTopic'].str.contains(str(j))), :].shape[0])
        senttopicdict[sentslabels[2]].append(theTooldf.loc[(theTooldf['com']<0) & (theTooldf['dominantTopic'].str.contains(str(j))), :].shape[0])
    outdf = pd.DataFrame.from_dict(senttopicdict)
    df_total = outdf["Positive"] + outdf["Neutral"] + outdf["Negative"]
    df_rel = outdf[outdf.columns[1:]].div(df_total, 0) * 100
    outdf['Positive'] = df_rel['Positive']
    outdf['Negative'] = df_rel['Negative']
    outdf['Neutral'] = df_rel['Neutral']
    print(outdf)
    ax = outdf.plot(
        figsize=(2,8),
        x='topic',
        kind='barh',
        stacked=True,
        title=toolname,
        legend=None,
        color = ['green', 'yellow', 'red'],
        mark_right=True)  # plt.show()
    for n in df_rel:
        for i, (cs, ab, pc) in enumerate(zip(outdf.iloc[:, 1:].cumsum(1)[n],
                                             outdf[n], df_rel[n])):
            #print(i, cs, ab, pc)
            plt.text(cs - ab / 2, i, str(np.round(pc, 1)) + '%',
                     va='center', ha='center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(f'{toolname}_sents.png', bbox_inches='tight')
    plt.show()


displayToolTopicBarChart()