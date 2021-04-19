import csv
import pandas as pd
import updateFlag

your_email = "xiaozhou.li@tuni.fi"

github_personal_token_input="368402c51b162c2d3bdad9667b528c2a94579396"
stackoverflow_key_input="89B5kN5OqCqblqKTWBKkjA(("
reddit_client_id_input="mSQg7W-mu7jM4Q"
reddit_client_secret_input="yN4K7PcB2KC7ve_ccO-_YiueFnguOA"
reddit_user_agent_input="my user agent again"
reddit_username_input="sideshowli"
reddit_password_input="admin123456"

def claimNewCrawlTask(email, datatype, start, end):
    updateFlag.updateflag('dataset/flag.csv', email, datatype, start, end)

def getGithubToken(youremail):
    github_personal_token = ""
    info_df = pd.read_csv("dataset/info.csv")
    if not info_df[info_df['email'] == youremail].empty:
        github_personal_token = info_df.loc[info_df['email'] == youremail, 'github_personal_token'].values[0]
    return github_personal_token

def getStackoverflowKey(youremail):
    stackoverflow_key = ""
    info_df = pd.read_csv("dataset/info.csv")
    if not info_df[info_df['email'] == youremail].empty:
        stackoverflow_key = info_df.loc[info_df['email'] == youremail, 'stackoverflow_key'].values[0]
    return stackoverflow_key

