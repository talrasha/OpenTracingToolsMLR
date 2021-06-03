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
        "filter": "withbody"
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

for item in dzone_all_files:
    thetoolname = item.split('\\')[1].split('.')[0]
    linklist = []
    with open(item, 'r', encoding='utf-8') as txtfile:
        linklist = [x.strip('\n') for x in txtfile.readlines()]
        linklist = list(set(linklist))
    print(thetoolname+': '+str(len(linklist)))



with open('sentlist.txt', 'r', encoding='utf-8') as txtfile:
    sentlist = [x.strip('\n').strip() for x in txtfile.readlines() if x!='']
for sent in sentlist:
        with open('sentlist2.txt', 'a', encoding='utf-8') as txtfile2:
            txtfile2.write(sent + '\n')


with open('sentlist2.txt', 'r', encoding='utf-8') as txtfile:
    sentlist = [x.strip('\n') for x in txtfile.readlines()]
newlist = []
for sent in sentlist:
    try:
        if (sent[-1]=='.' or sent[-1]=='?' or sent[-1]=='!'):
            newlist.append(sent)
        else:
            continue
    except:
        continue
for sent in newlist:
    with open('sentlist3.txt', 'a', encoding='utf-8') as txtfile:
        txtfile.write(sent + '\n')

df_stack = pd.read_csv('stackoverflow_sents.csv')
df_stack_ss = addsentimentvalues(df_stack)
df_stack_ss.to_csv('stackoverflow_sents_ss.csv', index=False)

df_dzone = pd.read_csv('dzone_sents.csv')
df_dzone_ss = addsentimentvalues(df_dzone)
df_dzone_ss.to_csv('dzone_sents_ss.csv', index=False)

df_medium = pd.read_csv('medium_sents.csv')
df_medium_ss = addsentimentvalues(df_medium)
df_medium_ss.to_csv('medium_sents_ss.csv', index=False)


train_data_clean = map(remove_noise, train_data_n)
test_data_clean = map(remove_noise, test_data_n)
# Convert all text data into tf-idf vectors
vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.95)
# vectorizer = TfidfVectorizer()
train_vec = vectorizer.fit_transform(train_data_clean)
test_vec = vectorizer.transform(test_data_clean)
print(train_vec.shape, test_vec.shape)
n_train_data = train_vec.shape[0]
split_ratio = 0.2 # labeled vs total(labeled+unlabeled)
X_l, X_u, y_l, y_u = train_test_split(train_vec, np.array(train_target_n), train_size=split_ratio, stratify=np.array(train_target_n))
print(X_l.shape, X_u.shape)
nb_clf = MultinomialNB(alpha=1e-2)
cross_validation(nb_clf, X_l, y_l)
nb_clf = MultinomialNB(alpha=1e-2).fit(X_l, y_l)
pred = nb_clf.predict(test_vec)
print(metrics.classification_report(np.array(test_target_n), pred, target_names=['non-info','info']))
# pprint(metrics.confusion_matrix(test_Xy.target, pred))
print(metrics.accuracy_score(np.array(test_target_n), pred))