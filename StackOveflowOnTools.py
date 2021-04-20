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
from PersonalInfo.updateInfo import your_email, getGithubToken, getStackoverflowKey
import updateFlag

personal_token = "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"
github_token = os.getenv('GITHUB_TOKEN', personal_token)
github_headers = {'Authorization': f'token {github_token}'}
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',25)
key = getStackoverflowKey(your_email)

STACKEXCHANGE = "https://api.stackexchange.com/"
VERSION = "2.2/"
endpoint = STACKEXCHANGE+VERSION+'search/advanced'

def getMaxPageNumberStackOverflow(toolname):
    params = {
        "key": key,
        "pagesize": 100,
        #    "page": 1,
        #    "order": "desc",
        "sort": "votes",
        #    "tagged": "visual-studio-code",
        "site": "stackoverflow",
        #    "title": "996.ICU",
        #    "filter": "withbody"
    }
    params['body'] = toolname
    theQuery = STACKEXCHANGE + VERSION + 'search/advanced'
    startpage = 1
    n = 50
    maxpage = 0
    while True:
        params['page'] = n
        response = requests.get(theQuery, params=params)
        try:
            theItemListPerPage = response.json()['items']
            if len(theItemListPerPage) == 100:
                n = n*2
                continue
            else:
                if len(theItemListPerPage)!=0:
                    return n
                else:
                    maxpage = n
                    break
        except:
            return 0
    while True:
        guesspage = int((maxpage+startpage)/2)
        params['page'] = guesspage
        theResult = requests.get(theQuery, params=params)
        try:
            theItemListPerPage = theResult.json()['items']
            if startpage>maxpage:
                return maxpage
            if len(theItemListPerPage) < 100 and len(theItemListPerPage)!=0:
                return guesspage
            elif len(theItemListPerPage) == 100:
                startpage = guesspage+1
            elif len(theItemListPerPage) == 0:
                maxpage = guesspage-1
        except:
            return 0

def getStackOverFlowQuestionsDAC(toolname):
    themaxpage = getMaxPageNumberStackOverflow(toolname)
    params = {
        "key": key,
        "pagesize": 100,
        "sort": "votes",
        "site": "stackoverflow"
    }
    theQuery = STACKEXCHANGE + VERSION + 'search/advanced'
    params['title'] = toolname
    params['page'] = 1
    theResult = requests.get(theQuery, params=params)
    thejson = theResult.json()
    pprint(thejson)



print(getMaxPageNumberStackOverflow("AppDynamics"))