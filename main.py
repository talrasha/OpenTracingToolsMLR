# This is a sample Python script.
# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import PersonalInfo.updateInfo
import updateFlag, StackExchangeAPI
import pandas as pd
import os
from PersonalInfo.updateInfo import your_email

datatypelist = ['stackoverflow']

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def installPackage():
    global installPythonPackage
    installPythonPackage = ['pip3 install prawcore pandas numpy requests praw']
    os.system('start cmd /k ' + installPythonPackage[0])

def startcrawling():
    #github_personal_token = PersonalInfo.PersonalInfoRedditExample.getGithubToken(PersonalInfo.PersonalInfoRedditExample.your_email)
    flagdf = pd.read_csv('dataset/flag.csv')
    taskdf = flagdf.loc[flagdf['email'] == PersonalInfo.updateInfo.your_email]
    targetdatatypes = taskdf.datatype.values.tolist()
    task_dict = {}
    for item in targetdatatypes:
        task_dict[item] = (taskdf.loc[taskdf['datatype']==item, 'start_point'].values[0], taskdf.loc[taskdf['datatype']==item, 'end_point'].values[0])
    for datatype in task_dict.keys():
        currentlocation = task_dict[datatype][0]
        maxlocation = task_dict[datatype][1]
        print("You have started to crawl: {}".format(datatype))
        if datatype == 'stackoverflow':
            StackExchangeAPI.getStackoverflowQuestionsDACfromProjectsInRange(currentlocation, maxlocation)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi(your_email)
    installPackage()
    print("Now you start crawling data.")
    startcrawling()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
