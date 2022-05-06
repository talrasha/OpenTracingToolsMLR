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

df_medium = pd.read_csv('Dataset/medium_sents_ss.csv')
df_dzone = pd.read_csv('Dataset/dzone_sents_ss.csv')
df_stack = pd.read_csv('Dataset/stackoverflow_sents_ss.csv')
with open('../Dataset/TrainingData/informative.txt', 'r', encoding='utf-8') as txtfile:
    inlist = [x.strip('\n') for x in txtfile.readlines()]
with open('../Dataset/TrainingData/noninformative.txt', 'r', encoding='utf-8') as txtfile:
    nlist = [x.strip('\n') for x in txtfile.readlines()]
with open('../Dataset/TrainingData/motivation.txt', 'r', encoding='utf-8') as txtfile:
    mlist = [x.strip('\n') for x in txtfile.readlines()]
with open('../Dataset/TrainingData/benefit.txt', 'r', encoding='utf-8') as txtfile:
    blist = [x.strip('\n') for x in txtfile.readlines()]
with open('../Dataset/TrainingData/issue.txt', 'r', encoding='utf-8') as txtfile:
    islist = [x.strip('\n') for x in txtfile.readlines()]

train_data_n = []
train_target_n = []
test_data_n = []
test_target_n = []

for i in range(1500):
    train_data_n.append(nlist[i])
    train_data_n.append(inlist[i])
    #train_data_n.append(mlist[i])
    #train_data_n.append(blist[i])
    #train_data_n.append(islist[i])
    train_target_n.extend([0,1])
for i in range(750,1500):
    test_data_n.append(nlist[i])
    test_data_n.append(inlist[i])
    #test_data_n.append(mlist[i])
    #test_data_n.append(blist[i])
    #test_data_n.append(islist[i])
    test_target_n.extend([0,1])

test_target_n = np.array(test_target_n)
train_target_n = np.array(train_target_n)
stop_words = stopwords.words('english')

df = pd.read_csv('sents_info_aspect.csv')
print(df.loc[df['aspect']==0, :].shape)
print(df.loc[df['aspect']==1, :].shape)
print(df.loc[df['aspect']==2, :].shape)
print(df.head())