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

context = ssl.create_default_context()
cookies = {'birthtime': '568022401'}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36', 'Referer': 'https://steamcommunity.com/', 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'}

df_questions = pd.read_csv('../Dataset/Outcomes/questions.csv')
df_answers = pd.read_csv('../Dataset/Outcomes/answers.csv')
df_questions.drop_duplicates(subset=['question_id'], inplace=True)
df_answers.drop_duplicates(subset=['answer_id'], inplace=True)
df_questions_sent = pd.read_csv('Dataset/questions_sents.csv')
df_answers_sent = pd.read_csv('Dataset/answers_sents.csv')

testingString = df_questions.loc[df_questions['question_id']==36596678, 'body'].values[0]
testingDzoneLink = "https://dzone.com/articles/appdynamics-introduces-support-for-sap"

monthDict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

dzone_all_files = glob.glob('Dataset/DzoneURLs/*.txt')
medium_all_files = glob.glob('Dataset/MediumRaw/*.txt')

def displayQuestionAnswerDistribution():
    a4_dims = (11.7, 8.27)
    smaller_dim = (8, 4.5)
    fig, ax = plt.subplots(figsize=smaller_dim)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    df_questions_gbtools = df_questions.groupby('toolname').count()
    df_questions_gbtools.reset_index(inplace=True)
    df_answers_gbtools = df_answers.groupby('toolname').count()
    df_answers_gbtools.reset_index(inplace=True)

    toollabels = df_questions_gbtools.loc[:,'toolname'].values.tolist()
    toolquestioncount = []
    toolanswercount =[]
    for tool in toollabels:
        toolquestioncount.append(df_questions_gbtools.loc[df_questions_gbtools['toolname']==tool, 'question_id'].values[0])
        toolanswercount.append(df_answers_gbtools.loc[df_answers_gbtools['toolname']==tool, 'answer_id'].values[0])
    x = np.arange(len(toollabels))
    width = 0.35
    rect1 = ax.bar(x - width/2, toolquestioncount, width, label='Questions')
    rect2 = ax.bar(x + width/2, toolanswercount, width, label='Answers')

    plt.title("Question and Answer Numbers for each Tool")
    plt.xlabel("Tools")
    plt.ylabel("Number")
    ax.set_xticks(x)
    ax.set_xticklabels(toollabels, rotation='vertical')
    ax.legend()
    ax.bar_label(rect1, padding=3)
    ax.bar_label(rect2, padding=3)
    fig.tight_layout()
    plt.savefig('QAdistribution.png', bbox_inches='tight')
    plt.show()

def displayQuestionAnswerMediumDZoneDistribution():
    a4_dims = (11.7, 8.27)
    smaller_dim = (8, 4.5)
    fig, ax = plt.subplots(figsize=smaller_dim)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    df_questions_gbtools = df_questions.groupby('toolname').count()
    df_questions_gbtools.reset_index(inplace=True)
    df_answers_gbtools = df_answers.groupby('toolname').count()
    df_answers_gbtools.reset_index(inplace=True)

    toollabels = df_questions_gbtools.loc[:,'toolname'].values.tolist()
    toolquestioncount = []
    toolanswercount =[]
    for tool in toollabels:
        toolquestioncount.append(df_questions_gbtools.loc[df_questions_gbtools['toolname']==tool, 'question_id'].values[0])
        toolanswercount.append(df_answers_gbtools.loc[df_answers_gbtools['toolname']==tool, 'answer_id'].values[0])
    x = np.arange(len(toollabels))
    width = 0.15
    rect1 = ax.bar(x - width*1.5, toolquestioncount, width, label='StackOverflow Qs.')
    rect2 = ax.bar(x - width/2, toolanswercount, width, label='StackOverflow As.')

    mediumpostcount = [25,139,4,1,9,65,1,2,1,28,51]
    dzonepostcount = [339,170,70,4,46,104,38,4,1,12,171]
    rect3 = ax.bar(x + width/2, mediumpostcount, width, label='MediumRaw Articles')
    rect4 = ax.bar(x + width*1.5, dzonepostcount, width, label='DZone Articles')

    plt.title("Social Media Content Volume for each Tool")
    plt.xlabel("Tools")
    plt.ylabel("Number")
    ax.set_xticks(x)
    ax.set_xticklabels(toollabels, rotation='vertical')
    ax.legend()
    ax.bar_label(rect1, padding=3, rotation='vertical', fontsize=8)
    ax.bar_label(rect2, padding=3, rotation='vertical', fontsize=8)
    ax.bar_label(rect3, padding=3, rotation='vertical', fontsize=8)
    ax.bar_label(rect4, padding=3, rotation='vertical', fontsize=8)
    fig.tight_layout()
    plt.savefig('QAMDdistribution.png', bbox_inches='tight')
    plt.show()

def fromParagrph2SentenceListUpdated(thestring):
    prelist = thestring.split('</p>')[:-1]
    plist = []
    for item in prelist:
        if '<p>' not in item:
            plist.append(item)
        elif len(item.split('<p>'))>1:
            plist.append(' '.join(item.split('<p>')[1:]))
        else:
            plist.append(item.split('<p>')[1])
    #plist = [x.split('<p>')[1] for x in prelist]
    sentences = []
    for p in plist:
        temp = str(p).split('>')
        temp_cleaned = []
        for piece in temp:
            if '<' in piece:
                if piece.split('<')[0]:
                    temp_cleaned.append(piece.split('<')[0])
                else:
                    continue
            else:
                temp_cleaned.append(piece)
        cleanedsent = ' '.join(temp_cleaned)
        tsents = sent_tokenize(str(cleanedsent))
        for sent in tsents:
            if '<a' in sent:
                if sent.split('<a')[0]:
                    sentences.append(sent.split('<a')[0] + sent.split('/a>')[-1])
                else:
                    continue
            elif '<img' in sent:
                if sent.split('<img')[0]:
                    sentences.append(sent.split('<img')[0] + sent.split('/img>')[-1])
                else:
                    continue
            else:
                sentences.append(sent)
    return sentences

def getSentenceLevelDataset4Questions():
    question_features = ['toolname', 'question_id', 'accepted_answer_id', 'answer_count', 'creation_date',
                         'is_answered', 'last_activity_date', 'last_edit_date', 'owner_id', 'owner_reputation', 'score',
                         'view_count', 'title', 'sentence']
    with open('Dataset/questions_sents.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(question_features)
    questionlen = df_questions.shape[0]
    for i in range(questionlen):
        questionitem = df_questions.iloc[i].values.tolist()
        sentences = fromParagrph2SentenceListUpdated(questionitem[-1])
        for sent in sentences:
            with open('Dataset/questions_sents.csv', 'a', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                questionitem = questionitem[:-1]+[sent]
                writer.writerow(questionitem)

def getSentenceLevelDataset4Answers():
    answer_features = ['toolname', 'answer_id', 'question_id', 'comment_count', 'creation_date', 'is_accepted',
                       'last_activity_date', 'owner_reputation', 'owner_id', 'score', 'sentence']
    with open('Dataset/answers_sents.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(answer_features)
    answerlen = df_answers.shape[0]
    for i in range(answerlen):
        answeritem = df_answers.iloc[i].values.tolist()
        sentences = fromParagrph2SentenceListUpdated(answeritem[-1])
        for sent in sentences:
            with open('Dataset/answers_sents.csv', 'a', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                answeritem = answeritem[:-1]+[sent]
                writer.writerow(answeritem)

def sentimentanalysis(np_textarray):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(np_textarray)
    return ss

def addsentimentvalues(pd_reviews):
    pd_reviews['ss'] = pd_reviews['sentence'].apply(str).apply(sentimentanalysis)
    np_ss = list(pd_reviews['ss'].to_numpy())
    neg = []
    pos = []
    neu = []
    com = []
    for item in np_ss:
        neg.append(item['neg'])
        pos.append(item['pos'])
        neu.append(item['neu'])
        com.append(item['compound'])
    pd_reviews['neg'] = neg
    pd_reviews['pos'] = pos
    pd_reviews['neu'] = neu
    pd_reviews['com'] = com
    return pd_reviews.drop(['ss'], axis=1)

def readStaticHTMLArticle(thelink):
    html = urlopen(thelink, context=context)
    bsObj = BeautifulSoup(html.read(), 'lxml')
    textitems = bsObj.find('script', {'type': 'application/ld+json'}).contents
    #jsondata = json.load(textitems[0])
    #print(type(textitems[0]))
    #print(bsObj.prettify())
    data = json.loads(textitems[0].strip())
    return data['articleBody']
    #for item in textitems:
    #    print(item)

def fromtextlist2csv(toolappname):
    with open('./DzoneURLs/'+toolappname+'.txt', 'r', encoding='utf-8') as txtfile:
        linklist = [x.strip('\n') for x in txtfile.readlines()]
    features = ['tag', 'text']
    count = 0
    for item in linklist:
        html = urlopen(item, context=context)
        bsObj = BeautifulSoup(html.read(), 'lxml')
        thetitle = bsObj.find('h1', {'class': 'article-title'}).get_text()
        thecontent = bsObj.find('div', {'class': 'content-html'}).get_text()
        #thetime = bsObj.find('span', {'class': 'author-date'}).get_text()
        #thetime = pd.to_datetime('20'+thetime.split(', ')[-1]+'-'+monthDict[thetime.split('. ')[0]]+'-'+thetime.split(', ')[0].split('. ')[1])
        with open('Dataset/dzone_td.csv', 'a', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([toolappname, (thetitle + thecontent)])
        count += 1
        print(toolappname+str(count))

def mediumfromtextlist2csv(toolappname):
    ########### Clean Text
    selected = [x for x in medium_all_files if toolappname in x]
    count = 0
    for article in selected:
        with open(article, 'r', encoding='utf-8') as txtfile:
            text = txtfile.read()
        if "min read" in text:
            textmain = text.split("min read")[1]
            textmain = textmain.strip()
        else:
            textmain = text.strip()
        with open('../Dataset/Outcomes/medium.csv', 'a', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([toolappname, textmain])
        count += 1
        print(toolappname+str(count))

def getSentenceLevelDatasetAndTxtMedium():
    df = pd.read_csv('../Dataset/Outcomes/medium.csv')
    for i in range(df.shape[0]):
        print(i+1)
        tempair = df.iloc[i].values.tolist()
        thetext = tempair[1]
        sentlist = sent_tokenize(thetext)
        sentlist = [str(x).strip('\n').strip('\r') for x in sentlist]
        for sent in sentlist:
            with open('Dataset/medium_sents.csv', 'a', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow([tempair[0], sent])
            with open('Dataset/medium_sentlist.txt', 'a', encoding='utf-8') as txtfile:
                txtfile.write(sent + '\n')

def getSentenceLevelDatasetAndTxtDzone():
    df = pd.read_csv('../Dataset/Outcomes/dzone.csv')
    for i in range(df.shape[0]):
        print(i+1)
        tempair = df.iloc[i].values.tolist()
        thetext = tempair[1]
        sentlist = sent_tokenize(thetext)
        sentlist = [str(x).strip('\n').strip('\r') for x in sentlist]
        for sent in sentlist:
            with open('Dataset/dzone_sents.csv', 'a', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow([tempair[0], sent])
            with open('Dataset/dzone_sentlist.txt', 'a', encoding='utf-8') as txtfile:
                txtfile.write(sent + '\n')

#for item in ['appdynamics', 'datadog', 'elasticapm', 'inspectit', 'instana', 'jaeger', 'lightstep', 'skywalking', 'stagemonitor', 'tanzu', 'zipkin']:
#    mediumfromtextlist2csv(item)
# {'Wavefront VMware', 'Jaeger', 'Datadog', 'InspectIT', 'Zipkin', 'Instana', 'SkyWalking', 'LightStep', 'AppDynamics', 'Elastic APM', 'Stagemonitor'}

def regularizeToolname(thestr):
    if thestr == 'Wavefront VMware':
        return 'tanzu'
    else:
        return ''.join(thestr.split()).lower()

def getSentenceLevelDatasetAndTxtStackOverflow():
    df_questions_sent['tool'] = df_questions_sent['toolname'].apply(regularizeToolname)
    df_questions_sent_reg = df_questions_sent.loc[:,['tool', 'sentence']]
    df_answers_sent['tool'] = df_answers_sent['toolname'].apply(regularizeToolname)
    df_answers_sent_reg = df_answers_sent.loc[:, ['tool', 'sentence']]
    df_stack = pd.concat([df_questions_sent_reg,df_answers_sent_reg])
    df_stack.to_csv('stackoverflow_sents.csv', index=False)
    sentlist = df_stack['sentence'].values.tolist()
    for sent in sentlist:
        with open('Dataset/stackoverflow_sentlist.txt', 'a', encoding='utf-8') as txtfile:
            txtfile.write(sent + '\n')

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

def removeSpecialCharacters(s):
    stripped = re.sub('[^\w\s]', '', s)
    stripped = re.sub('_', '', stripped)
    stripped = re.sub('\s+', ' ', stripped)
    stripped = stripped.strip()
    return stripped

def remove_noise2(sentence):
    result = ''
    poster = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stopword_set = set(stopwords.words('english'))
    wordlist = re.sub(r"\n|(\\(.*?){)|}|[!$%^&*#()_+|~\-={}\[\]:\";'<>?,.\/\\]|[0-9]|[@]", ' ', sentence) # remove punctuation
    wordlist = re.sub('\s+', ' ', wordlist) # remove extra space
    return wordlist

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def remove_noise(sentence):
    result = ''
    poster = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stopword_set = set(stopwords.words('english'))
    wordlist = re.sub(r"\n|(\\(.*?){)|}|[!$%^&*#()_+|~\-={}\[\]:\";'<>?,.\/\\]|[0-9]|[@]", ' ', sentence) # remove punctuation
    wordlist = re.sub('\s+', ' ', wordlist) # remove extra space
    wordlist_normal = [poster.stem(word.lower()) for word in wordlist.split()] # restore word to its original form (stemming)
    wordlist_normal = [lemmatizer.lemmatize(word, pos='v') for word in wordlist_normal] # restore word to its root form (lemmatization)
    wordlist_clean = [word for word in wordlist_normal if word not in stopword_set] # remove stopwords
    result = ' '.join(wordlist_clean)
    return result

def cross_validation(clf, data_X, data_y, unlabeled=None, n_folds=5):
    print('=' * 80)
    print("Validation: ")
    print(clf)
    kf = StratifiedKFold(n_splits=n_folds)
    start_time = time()
    train_accuracies= list() # training accuracy
    fold_count = 1
    original_clf = deepcopy(clf)
    for train_ids, valid_ids in kf.split(data_X, data_y):
        cv_clf = deepcopy(original_clf)
        print("Fold # %d" % fold_count)
        fold_count += 1
        #print(train_ids, valid_ids)
        train_X = data_X[train_ids]
        train_y = data_y[train_ids]
        valid_X = data_X[valid_ids]
        valid_y = data_y[valid_ids]
        if unlabeled==None:
            cv_clf.fit(train_X, train_y)
        else:
            cv_clf.fit(train_X, train_y, unlabeled)
        pred = cv_clf.predict(valid_X)
        train_accuracies.append(metrics.accuracy_score(valid_y, pred))
    train_time = time() - start_time
    print("Validation time: %0.3f seconds" % train_time)
    print("Average training accuracy: %0.3f" % np.mean(np.array(train_accuracies)))
    return train_accuracies, train_time

df_all = pd.read_csv('Dataset/data_sents_ss_info.csv', lineterminator='\n')
df_info = df_all.loc[df_all['informative']==1, :]
df_noninfo = df_all.loc[df_all['informative']==0, :]
#print(df_info.shape)

def allSentenceBarChart():
    a4_dims = (11.7, 8.27)
    smaller_dim = (8, 4.5)
    fig, ax = plt.subplots(figsize=smaller_dim)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    labels = list(df_info.groupby('tool').count().index)
    listcountinfosent = df_info.groupby('tool').count().loc[:, 'sentence'].values.tolist()
    listcountnoninfosent = df_noninfo.groupby('tool').count().loc[:, 'sentence'].values.tolist()
    allcount = list(np.array(listcountinfosent)+np.array(listcountnoninfosent))
    #print(licensenumberCount)
    x = np.arange(len(labels))
    width = 0.35
    rect1 = ax.bar(x, allcount, width)
    plt.title("Number of sentences each tool")
    plt.xlabel("Tools")
    plt.ylabel("Number")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.legend()
    ax.bar_label(rect1, padding=3)
    fig.tight_layout()
    plt.savefig('allSentence.png', bbox_inches='tight')
    plt.show()

def infoSentenceBarChart():
    a4_dims = (11.7, 8.27)
    smaller_dim = (8, 4.5)
    fig, ax = plt.subplots(figsize=smaller_dim)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    labels = list(df_info.groupby('tool').count().index)
    listcountinfosent = df_info.groupby('tool').count().loc[:, 'sentence'].values.tolist()
    listcountnoninfosent = df_noninfo.groupby('tool').count().loc[:, 'sentence'].values.tolist()
    #print(licensenumberCount)
    x = np.arange(len(labels))
    width = 0.35
    rect1 = ax.bar(x, listcountinfosent, width)
    plt.title("Number of informative sentences each tool")
    plt.xlabel("Tools")
    plt.ylabel("Number")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.legend()
    ax.bar_label(rect1, padding=3)
    fig.tight_layout()
    plt.savefig('InformativeSentence.png', bbox_inches='tight')
    plt.show()

def infononinfoBarChart():
    a4_dims = (11.7, 8.27)
    smaller_dim = (8, 4.5)
    fig, ax = plt.subplots(figsize=smaller_dim)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    labels = list(df_info.groupby('tool').count().index)
    listcountinfosent = df_info.groupby('tool').count().loc[:, 'sentence'].values.tolist()
    #print(licensenumberCount)
    listcountnoninfosent = df_noninfo.groupby('tool').count().loc[:, 'sentence'].values.tolist()

    x = np.arange(len(labels))
    width = 0.35
    rect1 = ax.bar(x - width / 2, listcountinfosent, width, label='Informative')
    rect2 = ax.bar(x + width / 2, listcountnoninfosent, width, label='Non-informative')

    plt.title("Number of Informative and Noninformative Sentences for each Tool")
    plt.xlabel("Tools")
    plt.ylabel("Number")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.legend()
    ax.bar_label(rect1, padding=3)
    ax.bar_label(rect2, padding=3)
    fig.tight_layout()
    plt.savefig('InfoNoninfoSentencesNumber.png', bbox_inches='tight')
    plt.show()

def infoNegPosBarChart():
    a4_dims = (11.7, 8.27)
    smaller_dim = (8, 4.5)
    fig, ax = plt.subplots(figsize=smaller_dim)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    labels = list(df_info.groupby('tool').count().index)
    listcountinfosent_pos = df_info.loc[df_info['com']>0,:].groupby('tool').count().loc[:, 'sentence'].values.tolist()
    #print(licensenumberCount)
    listcountinfosent_neg = df_info.loc[df_info['com']<0, :].groupby('tool').count().loc[:,
                            'sentence'].values.tolist()
    listcountinfosent_neu = df_info.loc[df_info['com']==0, :].groupby('tool').count().loc[:,
                            'sentence'].values.tolist()

    x = np.arange(len(labels))
    width = 0.2
    rect1 = ax.bar(x - width, listcountinfosent_pos, width, label='Positive')
    rect2 = ax.bar(x, listcountinfosent_neu, width, label='Neutral')
    rect3 = ax.bar(x + width, listcountinfosent_neg, width, label='Negative')

    plt.title("Number of Positive, Neutral and Negative Informative Sentences for each Tool")
    plt.xlabel("Tools")
    plt.ylabel("Number")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.legend()
    ax.bar_label(rect1, padding=2)
    ax.bar_label(rect2, padding=2)
    ax.bar_label(rect3, padding=2)
    fig.tight_layout()
    plt.savefig('SentimentSentencesNumber.png', bbox_inches='tight')
    plt.show()

#df = pd.read_csv('questions_sentiment.csv')
#neglist = df.loc[df['com']<0, 'sentence'].values.tolist()

#with open('TrainingData/issue.txt', 'r') as txtfile:
#    existinglist = [x.strip('\n') for x in txtfile.readlines()]
#thenotyet = [x for x in neglist if x not in existinglist]
#for item in thenotyet:
#    print(item)

#df_t = pd.read_csv('sents_info.csv')
#df_t.drop(columns=['x'], axis=1, inplace=True)
#df_t.to_csv('sents_info.csv', encoding='utf-8', index=False)

#df_info = pd.read_csv('sents_info_aspect.csv')
#print(df_info.shape)

fromtextlist2csv('td')
