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
