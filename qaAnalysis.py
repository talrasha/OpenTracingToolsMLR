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
import datetime

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',25)

df_questions = pd.read_csv('questions.csv')
df_answers = pd.read_csv('answers.csv')

