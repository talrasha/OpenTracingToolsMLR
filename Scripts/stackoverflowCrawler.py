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

personal_token = "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"
github_token = os.getenv('GITHUB_TOKEN', personal_token)
github_headers = {'Authorization': f'token {github_token}'}
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',25)
key = "89B5kN5OqCqblqKTWBKkjA(("
key2 = "luNgbAqlXjNtw499eJrSBA(("

STACKEXCHANGE = "https://api.stackexchange.com/"
VERSION = "2.3/"
endpoint = STACKEXCHANGE+VERSION+'search/advanced'

#df_toolinfo = pd.read_csv('dataset/tools.csv')
#toolnames = df_toolinfo['tool'].values.tolist()
#toolsearchstr = df_toolinfo['searchstr'].values.tolist()
#tooldict = {}
#for item in toolnames:
#    tooldict[item] = df_toolinfo.loc[df_toolinfo['tool']==item, 'searchstr'].values[0]

question_features = ['tag','question_id', 'accepted_answer_id', 'answer_count', 'creation_date',
                     'is_answered', 'last_activity_date', 'last_edit_date', 'owner_id',
                     'owner_reputation', 'score', 'view_count', 'title', 'body']
answer_features = ['tag', 'answer_id', 'question_id', 'comment_count', 'creation_date', 'is_accepted',
                   'last_activity_date', 'owner_reputation', 'owner_id', 'score', 'body']
comment_features = ['tag', 'comment_id', 'answer_id', 'question_id', 'creation_date', 'edited',
                    'owner_reputation', 'owner_id', 'score', 'body']

def initiateCSVs():
    with open('questions.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(question_features)
    with open('answers.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(answer_features)
    with open('comments.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(comment_features)

def getStackOverFlowDataset(toollist):
    paramsorigin = {
        "key": key,
        "pagesize": 100,
        "sort": "votes",
        "site": "stackoverflow",
        "filter": "!LGdawXSMGS0H5KeF1E6_cH"
    }
    theQuery = STACKEXCHANGE + VERSION + 'search/advanced'
    for tool in toollist:
        toolsearch = tool
        print("----> "+toolsearch)
        for searcharea in ['title', 'body']:
            print("----> "+searcharea)
            params = paramsorigin.copy()
            has_more = 1
            params['page'] = 0
            params[searcharea] = toolsearch
            while has_more:
                params['page'] = params['page'] + 1
                print("----> Page "+str(params['page']))
                theResult = requests.get(theQuery, params=params)
                thejson = theResult.json()
                questionslist = thejson['items']
                count = 0
                for question in questionslist:
                    count = count + 1
                    print("----> Question "+str(count))
                    questionitem = []
                    # ['toolname','question_id', 'accepted_answer_id', 'answer_count', 'creation_date',
                    # 'is_answered', 'last_activity_date', 'last_edit_date', 'owner_id'
                    # 'owner_reputation', 'score', 'view_count', 'title', 'body']
                    #toolname, question_id, accepted_answer_id, answer_count, creation_date,
                    # is_answered, last_activity_date, last_edit_date, owner_id,
                    # owner_reputation, score, view_count, title, body

                    questionitem.append(tool)
                    questionitem.append(question['question_id'])
                    try:
                        questionitem.append(question['accepted_answer_id'])
                    except KeyError:
                        questionitem.append(np.NaN)
                    questionitem.append(question['answer_count'])
                    questionitem.append(
                        datetime.datetime.fromtimestamp(question['creation_date']).strftime('%Y/%m/%d, %H:%M:%S'))
                    questionitem.append(question['is_answered'])
                    try:
                        questionitem.append(datetime.datetime.fromtimestamp(question['last_activity_date']).strftime('%Y/%m/%d, %H:%M:%S'))
                    except KeyError:
                        questionitem.append(np.NaN)
                    try:
                        questionitem.append(datetime.datetime.fromtimestamp(question['last_edit_date']).strftime('%Y/%m/%d, %H:%M:%S'))
                    except KeyError:
                        questionitem.append(np.NaN)
                    try:
                        questionitem.append(question['owner']['user_id'])
                    except KeyError:
                        questionitem.append(np.NaN)
                    try:
                        questionitem.append(question['owner']['reputation'])
                    except KeyError:
                        questionitem.append(np.NaN)
                    questionitem.append(question['score'])
                    questionitem.append(question['view_count'])
                    questionitem.append(question['title'])
                    questionitem.append(question['body'])
                    with open('questions.csv', 'a') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        writer.writerow(questionitem)
                    if question['answer_count'] > 0:
                        answers = question['answers']
                        for answer in answers:
                            # ['toolname', 'answer_id', 'question_id', 'comment_count', 'creation_date', 'is_accepted',
                            # 'last_activity_date', 'owner_reputation', 'owner_id', 'score', 'body']
                            answeritem = []
                            answeritem.append(tool)
                            answeritem.append(answer['answer_id'])
                            answeritem.append(answer['question_id'])
                            answeritem.append(answer['comment_count'])
                            answeritem.append(
                                datetime.datetime.fromtimestamp(answer['creation_date']).strftime('%Y/%m/%d, %H:%M:%S'))
                            answeritem.append(answer['is_accepted'])
                            try:
                                answeritem.append(datetime.datetime.fromtimestamp(answer['last_activity_date']).strftime('%Y/%m/%d, %H:%M:%S'))
                            except KeyError:
                                answeritem.append(np.NaN)
                            try:
                                answeritem.append(answer['owner']['reputation'])
                            except KeyError:
                                answeritem.append(np.NaN)
                            try:
                                answeritem.append(answer['owner']['user_id'])
                            except KeyError:
                                answeritem.append(np.NaN)
                            answeritem.append(answer['score'])
                            answeritem.append(answer['body'])
                            with open('answers.csv', 'a') as csvfile:
                                writer = csv.writer(csvfile, delimiter=',')
                                writer.writerow(answeritem)
                    else:
                        continue
                has_more = thejson['has_more']
            else:
                continue
