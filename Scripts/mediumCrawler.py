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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.stats.stats import pearsonr
from sklearn.metrics import *
import glob
from urllib.request import Request, urlopen
import json

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

testingurl1 = 'https://medium.com/picus-security-engineering/answers-to-faq-about-being-a-software-engineer-in-picus-ec94d3235f6a'
testingurl2 = 'https://medium.com/@a.minaro/the-not-so-simple-life-of-data-scientists-84da4050328'

def readArticleLink(link):
    req = Request(link, headers=headers)
    html = urlopen(req).read()
    #html = urlopen(link, context=context, header = )
    bsObj = BeautifulSoup(html, 'lxml')
    print(bsObj.prettify())

def getArticleIdDateTitle(link):
    req = Request(link, headers=headers)
    html = urlopen(req).read()
    # html = urlopen(link, context=context, header = )
    bsObj = BeautifulSoup(html, 'lxml')
    title = bsObj.find('script', {'type': 'application/ld+json'})
    maintext = title.get_text()
    jsontext = json.loads(maintext)
    return jsontext['identifier'], jsontext['datePublished'], jsontext['headline'],

def getArticleDate(link):
    req = Request(link, headers=headers)
    html = urlopen(req).read()
    # html = urlopen(link, context=context, header = )
    bsObj = BeautifulSoup(html, 'lxml')
    title = bsObj.find('script', {'type': 'application/ld+json'})
    maintext = title.get_text()
    jsontext = json.loads(maintext)
    return jsontext['datePublished']

def getArticleContent(link):
    req = Request(link, headers=headers)
    html = urlopen(req).read()
    # html = urlopen(link, context=context, header = )
    bsObj = BeautifulSoup(html, 'lxml')
    content = bsObj.find_all('p', {'class': 'pw-post-body-paragraph'})
    return ' '.join([x.get_text() for x in content])

def getArticleUrlListwithTag(thetag):
    theurl = f"https://medium.com/tag/{thetag}/archive"
    req = Request(theurl, headers=headers)
    html = urlopen(req).read()
    bsObj = BeautifulSoup(html, 'lxml')
    years = bsObj.find_all('div', {'class': 'timebucket u-inlineBlock u-width50'})
    #yearhrefs = [x.find('a').get('href') for x in years]
    #articleurls = []
    with open(f"{thetag}-medium.csv", 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id','date','title','text'])
    count = 0
    for year in years:
        year_req = Request(year.find('a').get('href'), headers=headers)
        year_html = urlopen(year_req).read()
        year_bsObj = BeautifulSoup(year_html, 'lxml')
        months = year_bsObj.find_all('div', {'class': 'timebucket u-inlineBlock u-width80'})
        for month in months:
            try:
                month_req = Request(month.find('a').get('href'), headers=headers)
                month_html = urlopen(month_req).read()
                month_bsObj = BeautifulSoup(month_html, 'lxml')
                days = month_bsObj.find_all('div', {'class': 'timebucket u-inlineBlock u-width35'})
                for day in days:
                    try:
                        day_req = Request(day.find('a').get('href'), headers=headers)
                        day_html = urlopen(day_req).read()
                        day_bsObj = BeautifulSoup(day_html, 'lxml')
                        urls = [x.get('href') for x in day_bsObj.find_all('a', {'class': 'button button--smaller button--chromeless u-baseColor--buttonNormal'})]
                        #articleurls.extend(urls)
                        for url in urls:
                            tempinput = []
                            theid,thedate,thetitle = getArticleIdDateTitle(url)
                            tempinput.append(theid)
                            tempinput.append(thedate)
                            tempinput.append(thetitle)
                            tempinput.append(getArticleContent(url))
                            with open(f"{thetag}-medium.csv", 'a') as csvfile:
                                writer = csv.writer(csvfile, delimiter=',')
                                writer.writerow(tempinput)
                            count = count+1
                            print(count)
                    except:
                        continue
            except:
                continue

def getArticleUrlListwithTagContinue(thetag):
    theurl = f"https://medium.com/tag/{thetag}/archive"
    req = Request(theurl, headers=headers)
    html = urlopen(req).read()
    bsObj = BeautifulSoup(html, 'lxml')
    years = bsObj.find_all('div', {'class': 'timebucket u-inlineBlock u-width50'})
    #yearhrefs = [x.find('a').get('href') for x in years]
    #articleurls = []
    dfexist = pd.read_csv(f"{thetag}-medium.csv")
    existingids = dfexist['id'].values.tolist()
    lastdate = dfexist['date'].values.tolist()[-1]
    lastyear = int(lastdate.split('T')[0].split('-')[0])
    count = 0
    for year in [x for x in years if int(x.find('a').get_text())>=lastyear]:
        year_req = Request(year.find('a').get('href'), headers=headers)
        year_html = urlopen(year_req).read()
        year_bsObj = BeautifulSoup(year_html, 'lxml')
        months = year_bsObj.find_all('div', {'class': 'timebucket u-inlineBlock u-width80'})
        for month in months:
            try:
                month_req = Request(month.find('a').get('href'), headers=headers)
                month_html = urlopen(month_req).read()
                month_bsObj = BeautifulSoup(month_html, 'lxml')
                days = month_bsObj.find_all('div', {'class': 'timebucket u-inlineBlock u-width35'})
                for day in days:
                    try:
                        day_req = Request(day.find('a').get('href'), headers=headers)
                        day_html = urlopen(day_req).read()
                        day_bsObj = BeautifulSoup(day_html, 'lxml')
                        urls = [x.get('href') for x in day_bsObj.find_all('a', {'class': 'button button--smaller button--chromeless u-baseColor--buttonNormal'})]
                        #articleurls.extend(urls)
                        for url in urls:
                            tempinput = []
                            theid,thedate,thetitle = getArticleIdDateTitle(url)
                            if theid in existingids:
                                continue
                            else:
                                tempinput.append(theid)
                                tempinput.append(thedate)
                                tempinput.append(thetitle)
                                tempinput.append(getArticleContent(url))
                                with open(f"{thetag}-medium.csv", 'a') as csvfile:
                                    writer = csv.writer(csvfile, delimiter=',')
                                    writer.writerow(tempinput)
                                count = count+1
                                print(count)
                    except:
                        continue
            except:
                continue

def makeupforbefore():
    years = list(range(2010,2015))
    for year in years:
        theurl = f"https://medium.com/tag/technical-debt/archive/{year}"
        req = Request(theurl, headers=headers)
        html = urlopen(req).read()
        bsObj = BeautifulSoup(html, 'lxml')
        articles = bsObj.find_all('a', {'class': 'button button--smaller button--chromeless u-baseColor--buttonNormal'})
        for article in articles:
            url = article.get('href')
            tempinput = []
            theid, thedate, thetitle = getArticleIdDateTitle(url)
            tempinput.append(theid)
            tempinput.append(thedate)
            tempinput.append(thetitle)
            tempinput.append(getArticleContent(url))
            with open(f"technical-debt-medium.csv", 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(tempinput)

getArticleUrlListwithTag('AI')