import os
from pprint import pprint
import requests
import requests.auth
import pandas as pd
import numpy as np
import time
import re
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
import glob
import gensim
from gensim.utils import simple_preprocess
import re
import seaborn as sns
from nltk.corpus import stopwords
from time import time
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from pprint import pprint
import tqdm
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from copy import deepcopy
from scipy.sparse import csr_matrix, vstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from scipy.linalg import get_blas_funcs
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import pprint
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.error import HTTPError, URLError
import datetime
import ssl
import requests
import csv
import urllib
import time
import operator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.stats.stats import pearsonr
from sklearn.metrics import *
import glob
from urllib.request import Request, urlopen
import json
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn import metrics

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',25)

monthnumber = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
now = datetime.datetime.now()
rec = {'Recommended': True, 'Not Recommended': False}
context = ssl._create_unverified_context()
cookies = {'birthtime': '568022401'}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36', 'Referer': 'https://steamcommunity.com/', 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'}



def fromtextlist2csv(toolappname):
    with open('./DzoneURLs/'+toolappname+'.txt', 'r', encoding='utf-8') as txtfile:
        linklist = [x.strip('\n') for x in txtfile.readlines()]
    features = ['tag', 'text']
    count = 0
    for item in linklist:
        req = Request(item, headers=headers)
        html = urlopen(req).read()
        bsObj = BeautifulSoup(html, 'lxml')
        thetitle = bsObj.find('h1', {'class': 'article-title'}).get_text()
        thecontent = bsObj.find('div', {'class': 'content-html'}).get_text()
        #thetime = bsObj.find('span', {'class': 'author-date'}).get_text()
        #thetime = pd.to_datetime('20'+thetime.split(', ')[-1]+'-'+monthDict[thetime.split('. ')[0]]+'-'+thetime.split(', ')[0].split('. ')[1])
        with open('Dataset/dzone_td.csv', 'a', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([toolappname, (thetitle + thecontent)])
        count += 1
        print(toolappname+str(count))

fromtextlist2csv('td')
