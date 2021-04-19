import csv
import pandas as pd

def updateflag(flagcsv, youremail, datatype, currentlocation, maxlocation):
    flagdf = pd.read_csv(flagcsv)
    if not flagdf[(flagdf['email'] == youremail) & (flagdf['datatype'] == datatype)].empty:
        flagdf.loc[(flagdf['email'] == youremail) & (flagdf['datatype'] == datatype), 'start_point'] = currentlocation
        flagdf.loc[(flagdf['email'] == youremail) & (flagdf['datatype'] == datatype), 'end_point'] = maxlocation
        flagdf.to_csv(flagcsv, encoding='utf-8', index=False)
    else:
        with open(flagcsv, 'a', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([youremail,datatype,currentlocation,maxlocation])

def getflag(flagcsv, youremail, datatype):
    flagdf = pd.read_csv(flagcsv)
    if flagdf[(flagdf['email'] == youremail) & (flagdf['datatype'] == datatype)].empty:
        return -1, -1
    else:
        currentlocation = flagdf.loc[(flagdf['email'] == youremail) & (flagdf['datatype'] == datatype), 'start_point'].values[0]
        maxlocation = flagdf.loc[(flagdf['email'] == youremail) & (flagdf['datatype'] == datatype), 'end_point'].values[0]
        return currentlocation, maxlocation
