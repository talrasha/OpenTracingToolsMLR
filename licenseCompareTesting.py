import os
from pprint import pprint
import requests
import requests.auth
import pandas as pd
import numpy as np
import time
import csv, json
import itertools
from difflib import SequenceMatcher
import datetime
import matplotlib.pyplot as plt
import operator
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import ssl
from urllib.request import urlopen

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',25)

context = ssl.create_default_context()
cookies = {'birthtime': '568022401'}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36', 'Referer': 'https://steamcommunity.com/', 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'}

df = pd.read_csv('selected40projects.csv')

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def displayLicensePieChart():
    smaller_dim = (8, 4.5)
    labels = list(set(df['license'].values.tolist()))
    licenselist = df['license'].values.tolist()
    counts = []
    for item in labels:
        counts.append(len([x for x in licenselist if x==item]))
    #explode = (0, 0)
    fig, ax = plt.subplots(figsize=smaller_dim)
    ax.pie(counts, explode=None, labels=labels, autopct=lambda pct: func(pct, counts), shadow=False,
           startangle=90)
    ax.axis('equal')
    plt.title("Percentage of Licenses")
    plt.savefig('40LicensePieChart.png', bbox_inches='tight')
    plt.show()

def displayLanguagePieChart():
    smaller_dim = (8, 4.5)
    labels = list(set(df['main language'].values.tolist()))
    languagelist = df['main language'].values.tolist()
    counts = []
    for item in labels:
        counts.append(len([x for x in languagelist if x == item]))
    # explode = (0, 0)
    fig, ax = plt.subplots(figsize=smaller_dim)
    ax.pie(counts, explode=None, labels=labels, autopct=lambda pct: func(pct, counts), shadow=False,
           startangle=90)
    ax.axis('equal')
    plt.title("Percentage of Main Language")
    plt.savefig('40LanguagePieChart.png', bbox_inches='tight')
    plt.show()

commits = df['size'].values.tolist()
commits = [x for x in commits if x!='∞']
print(sum([int(x) for x in commits])/len(commits))
print(commits)