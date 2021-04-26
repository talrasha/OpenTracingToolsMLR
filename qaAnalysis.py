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


desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',25)

df_questions = pd.read_csv('questions.csv')
df_answers = pd.read_csv('answers.csv')
df_questions.drop_duplicates(subset=['question_id'], inplace=True)
df_answers.drop_duplicates(subset=['answer_id'], inplace=True)

testingString = df_questions.loc[df_questions['question_id']==36596678, 'body'].values[0]

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

def fromParagrph2SentenceList(thestring):
    prelist = thestring.split('</p>')[:-1]
    plist = [x.split('<p>')[1] for x in prelist]
    sentences = []
    for p in plist:
        tsents = sent_tokenize(str(p))
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

def getSentenceLevelDataset():
    question_features = ['toolname', 'question_id', 'accepted_answer_id', 'answer_count', 'creation_date',
                         'is_answered', 'last_activity_date', 'last_edit_date', 'owner_id', 'owner_reputation', 'score',
                         'view_count', 'title', 'sentence']
    answer_features = ['toolname', 'answer_id', 'question_id', 'comment_count', 'creation_date', 'is_accepted',
                       'last_activity_date', 'owner_reputation', 'owner_id', 'score', 'sentence']
    with open('questions_sents.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(question_features)
    questionlen = df_questions.shape[0]
    for i in range(questionlen):
        questionitem = df_questions.iloc[i].values.tolist()
        sentences = fromParagrph2SentenceList(questionitem[-1])
        for sent in sentences:
            with open('questions_sents.csv', 'a', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                questionitem = questionitem[:-1]+[sent]
                writer.writerow(questionitem)

