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

params = {
    "key": key,
    "pagesize": 100,
    "page": 1,
#    "order": "desc",
    "sort": "votes",
#    "tagged": "visual-studio-code",
#    "site": "stackoverflow",
    "title": "vscode",
#    "filter": "withbody"
}

#response = requests.get(endpoint, params=params)
#result = response.json()
#pprint(result)


