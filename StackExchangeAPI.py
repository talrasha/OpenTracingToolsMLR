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

personal_token = getGithubToken(your_email)
#personal_token = "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"
github_token = os.getenv('GITHUB_TOKEN', personal_token)
github_headers = {'Authorization': f'token {github_token}'}
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',25)

#client_id = '19145'
#client_secret = 'kRmZ)lSyy3bEzpDZu7aWQA(('
#scope = 'no_expiry'
#key = '89B5kN5OqCqblqKTWBKkjA(('
key = getStackoverflowKey(your_email)
#redirect_uri = 'https://stackexchange.com/oauth/login_success'
#oauth_url = 'https://stackoverflow.com/oauth'
#token_url = 'https://stackoverflow.com/oauth/access_token/json'
#client_auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
#oauth = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scope)
#authorization_url, state = oauth.authorization_url('https://stackexchange.com/oauth/dialog')

STACKEXCHANGE = "https://api.stackexchange.com/"
VERSION = "2.2/"
endpoint = STACKEXCHANGE+VERSION+'search/advanced'

#params = {
#    "key": key,
#    "pagesize": 100,
#    "page": 1,
#    "order": "desc",
#    "sort": "votes",
#    "tagged": "visual-studio-code",
#    "site": "stackoverflow",
#    "title": "vscode",
#    "filter": "withbody"
#}

#response = requests.get(endpoint, params=params)
#result = response.json()
#pprint(result)

def calculateProjectnameTagSimilarity(projectname, tag):
    return SequenceMatcher(None, projectname, tag).ratio()

def getStackOverflowKeyTagfromProjectName(projectname):
    theQuery = STACKEXCHANGE + VERSION + 'search/advanced'
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
    params['page'] = 1
    params['title'] = projectname
    response = requests.get(theQuery, params=params)
    result = response.json()
    try:
        itemlist = result['items']
        taglist = [x['tags'] for x in itemlist]
        merged = list(itertools.chain.from_iterable(taglist))
        #sortedmerged = sorted(set(merged), key=merged.count, reverse=True)
        countdict = {}
        for item in merged:
            if item in countdict.keys():
                countdict[item] = countdict[item]+1
            else:
                countdict[item] = 1
        for feature in countdict.keys():
            countdict[feature] = [countdict[feature], calculateProjectnameTagSimilarity(projectname, feature)]
        #return countdict
        return [k for k in sorted(countdict.items(), key=lambda item: item[1][0]*item[1][1], reverse=True)][0][0]
        #sortedmergedsimilarity = {x:calculateProjectnameTagSimilarity(projectname, x) for x in sortedmerged}
        #return [k for k in sorted(sortedmergedsimilarity.items(), key=lambda item: item[1], reverse=True)][0]
    except:
        return -1

def getStackOverflowQuestiondata(projectname):
    quota_max = 10000
    quota_remaining = 10000
    theProjectQuery = f"https://api.github.com/repos/{projectname}"
    p_search = requests.get(theProjectQuery, headers=github_headers)
    project_info = p_search.json()
    project_id = project_info['id']
    theQuery = STACKEXCHANGE + VERSION + 'search/advanced'
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
    dataitemweneed = ['answer_count', 'creation_date', 'is_answered', 'last_activity_date', 'score', 'view_count']
    page = 1
    result = []
    projecttag = getStackOverflowKeyTagfromProjectName(projectname)
    params['title'] = projectname.split('/')[1]
    if projecttag == -1:
        pass
    else:
        params['tagged'] = projecttag
    while 1==1:
        params['page'] = page
        response = requests.get(theQuery, params=params)


        searchresultperpage = response.json()['items']
        if len(searchresultperpage) == 0:
            break
        else:
            for question in searchresultperpage:
                eachquestiondata = {}
                eachquestiondata['answer_count'] = question['answer_count']
                eachquestiondata['creation_date'] = question['creation_date']
                eachquestiondata['is_answered'] = question['is_answered']
                try:
                    eachquestiondata['last_activity_date'] = question['last_activity_date']
                except:
                    eachquestiondata['last_activity_date'] = question['creation_date']
                eachquestiondata['score'] = question['score']
                eachquestiondata['view_count'] = question['view_count']
                result.append(eachquestiondata)
            #print(page)
            page = page + 1
    thereturnresult = {}
    thereturnedfeatures = ['question_count', 'average_answer_count', 'answered_percentage', 'average_active_time', 'average_score', 'average_view']
    thereturnresult['project_id'] = project_id
    if len(result)==0:
        for feature in thereturnedfeatures:
            thereturnresult[feature] = np.NaN
    else:
        thereturnresult['question_count'] = len(result)
        thereturnresult['average_answer_count'] = sum([x['answer_count'] for x in result])/len(result)
        thereturnresult['answered_percentage'] = sum([x['is_answered'] for x in result])/len(result)
        thereturnresult['average_active_time'] = sum([(x['last_activity_date']-x['creation_date']) for x in result])/len(result)
        thereturnresult['average_score'] = sum([x['score'] for x in result])/len(result)
        thereturnresult['average_view'] = sum([x['view_count'] for x in result])/len(result)
    return thereturnresult

def getStackOverflowDataProjectsInRange(thestackoverflowcsv, fromN, toN):
    with open("dataset/toollist.txt", 'r', encoding='utf-8') as txtfile:
        projectList = [x.strip('\n') for x in txtfile.readlines()][fromN:toN]
    count = fromN + 1
    features = list(getStackOverflowQuestiondata(projectList[0]).keys())
    with open(thestackoverflowcsv, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(features)
    for project in projectList:
        print(count)
        result = getStackOverflowQuestiondata(project)
        with open(thestackoverflowcsv, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([result[x] for x in features])
        count = count + 1

def getMaxPageNumberStackOverflow(projectfullname):
    starttime = time.time()
    projecttag = getStackOverflowKeyTagfromProjectName(projectfullname)
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
    params['title'] = projectfullname.split('/')[1]
    if projecttag == -1:
        pass
    else:
        params['tagged'] = projecttag
    theQuery = STACKEXCHANGE + VERSION + 'search/advanced'
    startpage = 1
    n = 50
    maxpage = 0
    pipeline = []
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
                    endtime = time.time()
                    #print(endtime-starttime)
                    return n
                else:
                    maxpage = n
                    break
        except:
            return 0
    #print(maxpage)
    while True:
        guesspage = int((maxpage+startpage)/2)
        params['page'] = guesspage
        theResult = requests.get(theQuery, params=params)
        try:
            theItemListPerPage = theResult.json()['items']
            if startpage>maxpage:
                return maxpage
            if len(theItemListPerPage) < 100 and len(theItemListPerPage)!=0:
                endtime = time.time()
                #print(endtime-starttime)
                return guesspage
            elif len(theItemListPerPage) == 100:
                #print([startpage, maxpage])
                startpage = guesspage+1
            elif len(theItemListPerPage) == 0:
                #print([startpage, maxpage])
                maxpage = guesspage-1
        except:
            return 0

def getStackOverFlowQuestionsDAC(projectfullname, stackoverflowdatacsv):
    theProjectQuery = f"https://api.github.com/repos/{projectfullname}"
    p_search = requests.get(theProjectQuery, headers=github_headers)
    project_info = p_search.json()
    print(project_info)
    project_id = project_info['id']
    themaxpage = getMaxPageNumberStackOverflow(projectfullname)
    params = {
        "key": key,
        "pagesize": 100,
        "sort": "votes",
        "site": "stackoverflow"
    }
    theQuery = STACKEXCHANGE + VERSION + 'search/advanced'
    projecttag = getStackOverflowKeyTagfromProjectName(projectfullname)
    params['title'] = projectfullname.split('/')[1]
    if projecttag == -1:
        pass
    else:
        params['tagged'] = projecttag
    params['page'] = themaxpage
    theResult = requests.get(theQuery, params=params)
    thejson = theResult.json()
    #pprint(thejson)
    features = ['project_id', 'questions_count']
    if 'error_id' in thejson and thejson['error_name'] == 'throttle_violation' and 'too many requests from this IP' in thejson['error_message']:
        print(thejson['error_message'])
        time.sleep(int(thejson['error_message'].split()[-2])+1)
        return 1
    else:
        try:
            itemsonlastpage = thejson['items']
            #pprint([x['question_id'] for x in itemsonlastpage])
            thereturn = [project_id, 100*(themaxpage-1)+len(itemsonlastpage)]
        except:
            thereturn = [project_id, 0]
        currentdf = pd.read_csv(stackoverflowdatacsv, index_col='project_id')
        existingprojects = currentdf.index.values.tolist()
        #if currentdf.empty:
        #    with open(newissuedatatable, 'a', encoding='utf-8') as csvfile:
        #        writer = csv.writer(csvfile, delimiter=',')
        #        writer.writerow(features)
        if project_id in existingprojects:
            currentdf.loc[project_id] = thereturn[1:]
            currentdf.reset_index(inplace=True)
            currentdf.to_csv(stackoverflowdatacsv, encoding='utf-8', index=False)
        else:
            with open(stackoverflowdatacsv, 'a', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(thereturn)
        return 0

def getStackoverflowQuestionsDACfromProjectsInRange(fromN, toN):
    with open("dataset/toollist.txt", 'r', encoding='utf-8') as txtfile:
        projectList = [x.strip('\n') for x in txtfile.readlines()][fromN:toN]
    count = fromN+1
    skipped = []
    for project in projectList:
        #starttime = time.time()
        #try:
        print("Stackoverflow {}".format(count))
        apilimiterror = getStackOverFlowQuestionsDAC(project, "dataset/newstackoverflowdata.csv")
        if apilimiterror:
            skipped.append(project)
            updateFlag.updateflag("dataset/flag.csv", your_email, 'stackoverflow', count, toN)
            count = count + 1
            continue
        #endtime = time.time()
        #with open("newtimeperproject.csv", 'a', encoding='utf-8') as csvfile:
        #    writer = csv.writer(csvfile, delimiter=',')
        #    writer.writerow([project, endtime-starttime])
        #except KeyError:
        #    print("Project {} NOT exist any more... very sad".format(count))
        else:
            updateFlag.updateflag("dataset/flag.csv", your_email, 'stackoverflow', count, toN)
            count = count + 1
    while True:
        if skipped:
            for project in skipped:
                apilimiterror = getStackOverFlowQuestionsDAC(project, "dataset/newstackoverflowdata.csv")
                if apilimiterror:
                    continue
                else:
                    skipped.remove(project)
        else:
            break

#getStackOverflowDataProjectsInRange("stackoverflowdata.csv", 0, 10)
#print(getStackOverflowQuestiondata("facebook/react"))
#print(getStackOverflowKeyTagfromProjectName("react"))
#getStackOverFlowQuestionsDAC('freeCodeCamp/freeCodeCamp', 'newstackoverflowdata.csv')

