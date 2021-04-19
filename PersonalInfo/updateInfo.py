import csv
import pandas as pd
from datacrawling import updateFlag

#your_email = "lixiaozhou725@live.com"
your_email = "savanna.lujan@tuni.fi"

github_personal_token_input="368402c51b162c2d3bdad9667b528c2a94579396"
stackoverflow_key_input="89B5kN5OqCqblqKTWBKkjA(("
reddit_client_id_input="mSQg7W-mu7jM4Q"
reddit_client_secret_input="yN4K7PcB2KC7ve_ccO-_YiueFnguOA"
reddit_user_agent_input="my user agent again"
reddit_username_input="sideshowli"
reddit_password_input="admin123456"

def updatePersonalInfo(email=your_email,
                       github_personal_token=github_personal_token_input,
                       stackoverflow_key=stackoverflow_key_input,
                       reddit_client_id=reddit_client_id_input,
                       reddit_client_secret=reddit_client_secret_input,
                       reddit_user_agent=reddit_user_agent_input,
                       reddit_username=reddit_username_input,
                       reddit_password=reddit_password_input):
    currentdf = pd.read_csv("dataset/info.csv", index_col='email')
    existingemails = currentdf.index.values.tolist()
    if email in existingemails:
        currentdf.loc[email] = [github_personal_token,stackoverflow_key,reddit_client_id,reddit_client_secret,reddit_user_agent,reddit_username,reddit_password]
        currentdf.reset_index(inplace=True)
        currentdf.to_csv("dataset/info.csv", encoding='utf-8', index=False)
    else:
        with open("dataset/info.csv", 'a', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([email,github_personal_token,stackoverflow_key,reddit_client_id,reddit_client_secret,reddit_user_agent,reddit_username,reddit_password])

def claimNewCrawlTask(email, datatype, start, end):
    updateFlag.updateflag('dataset/flag.csv', email, datatype, start, end)

def getGithubToken(youremail):
    info_df = pd.read_csv("dataset/info.csv")
    if not info_df[info_df['email'] == youremail].empty:
        github_personal_token = info_df.loc[info_df['email'] == youremail, 'github_personal_token'].values[0]
    return github_personal_token

def getStackoverflowKey(youremail):
    info_df = pd.read_csv("dataset/info.csv")
    if not info_df[info_df['email'] == youremail].empty:
        stackoverflow_key = info_df.loc[info_df['email'] == youremail, 'stackoverflow_key'].values[0]
    return stackoverflow_key

