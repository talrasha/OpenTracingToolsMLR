# Open Tracing Tools. An Overview and Critical Comparison (the Replication Package)

## Table of Contents
* **[Introduction](#Introduction)**
* **[How To Reference the Work](#How-to-reference-the-work)**
* Datasets
  * **[DzoneURLs](#Dataset)**
  * **[MediumRaw](#Dataset)**
  * **[TrainingData](#Dataset)**
  * **[Outcomes](#Dataset)**
* Scripts
  * **[dzoneCrawler.py](#Scripts)**
  * **[mediumCrawler.py](#Scripts)**
  * **[stackoverflowCrawler.py](#Scripts)**
  * **[dataPreprocess.py](#Scripts)**
  * **[topicModeling.py](#Scripts)**
* How to Use
  * **[Step 0. Data Crawling](#step-0-data-crawling)**
  * **[Step 1. Preprocessing](#step-1-preprocessing)**
  * **[Step 2. Filtering](#step-2-filtering)**
  * **[Step 3. Topic Modeling](#step-3-topic-modeling)**
  * **[Step 4. Topic Mapping](#step-4-topic-mapping)**
  * **[Step 5. Opinion Mining](#step-5-opinion-mining)**


## Introduction
Distributed tracing can help to pinpoint where failures occur and what causes poor performance in the system. DevOps teams can monitor, debug and optimize their code of modern distributed software architectures, such as microservices or serverless functions. The publication "*Open Tracing Tools. An Overview and Critical Comparison*" provided an overview and performed a
critical comparison of eleven Open Tracing Tools. 

This replication package provides the necessary tools towards collecting and analyzing the "gray" literatures from popular sources, i.e., [Medium](https://medium.com/), [Dzone](https://dzone.com/), and [StackOverflow](https://stackoverflow.com/), as well as the raw data collected for this publication.

## How To Reference this Project
Please cite as *Open Tracing Tools. An Overview and Critical Comparison* [1]

[1] To be updated when the submission is officially published.

## Dataset
* [DzoneURLs](https://github.com/talrasha/OpenTracingToolsMLR/tree/main/Dataset/DzoneURLs): For each open tracing tool, a list of article URLs from Dzone are saved in a separate .txt file with the tool name as the filename in this folder. For example, *datadog.txt* contains the URLs of all the Dzone articles regarding *DataDog*. 
* [MediumRaw](https://github.com/talrasha/OpenTracingToolsMLR/tree/main/Dataset/MediumRaw): For each open tracing tool, each article from Medium regarding it is saved as a separate .txt file with the tool name and an index together as the filename in this folder. For example, *appdynamics-23.txt* contains the 23rd article from Medium regarding *AppDynamics*.
* [TrainingData](https://github.com/talrasha/OpenTracingToolsMLR/tree/main/Dataset/TrainingData): The training data for classifying the *informative* and *non-informative* sentences using Naive Bayes Classifier. Therein, 1500 informative and 1500 non-informative sentences are manually labelled.
* [OutComes](https://github.com/talrasha/OpenTracingToolsMLR/tree/main/Dataset/Outcomes): The collected and organized textual data from each source.

## Scripts
* [dzoneCrawler.py](https://github.com/talrasha/OpenTracingToolsMLR/blob/main/Scripts/dzoneCrawler.py): The script to crawl articles from the Dzone URL lists. 
* [mediumCrawler.py](https://github.com/talrasha/OpenTracingToolsMLR/blob/main/Scripts/mediumCrawler.py): The script to crawl Medium articles based on tags (new feature, not used in this publication)
* [stackoverflowCrawler.py](https://github.com/talrasha/OpenTracingToolsMLR/blob/main/Scripts/stackoverflowCrawler.py): The script to crawl questions and answers from StackOverflow using APIs.
* [dataPreprocess.py](https://github.com/talrasha/OpenTracingToolsMLR/blob/main/Scripts/dataPreprocess.py): The script to pre-process textual data.
* [filteringTopicModeling.ipynb](https://github.com/talrasha/OpenTracingToolsMLR/blob/main/Scripts/filteringTopicModeling.ipynb): The script to conduct filtering and topic modeling on textual data as well as related plotting functions 

## How to Use

### Step 0. Data Crawling

For crawling data from **Medium**, use the *getArticleUrlListwithTag()* function in *[mediumCrawler.py](https://github.com/talrasha/OpenTracingToolsMLR/blob/main/Scripts/mediumCrawler.py)*, if the tool name is archived as tag in Medium, e.g., ['datadog'](https://medium.com/tag/datadog/archive) as a tag. For the tool names that are not tags, the data can be extracted manually. To be noted, in the submission, Medium data was crawled manually.

For crawling data from **Dzone**, first manually extract the URL list of the target articles by searching from the Dzone website and save it as TXT with the title of the tool name. For example, the URL list of articles about *'Zipkin'* is saved in *[zipkin.txt](https://github.com/talrasha/OpenTracingToolsMLR/tree/main/Dataset/DzoneURLs/zipkin.txt)*. Then use the *fromtextlist2csv()* function in *[dzoneCrawler.py](https://github.com/talrasha/OpenTracingToolsMLR/blob/main/Scripts/dzoneCrawler.py)* to extract and format the data into CSV file.

For crawling data from **Stack Overflow**, use the *getStackOverFlowDataset()* function in *[stackoverflowCrawler.py](https://github.com/talrasha/OpenTracingToolsMLR/blob/main/Scripts/stackoverflowCrawler.py)* to get the questions and answers data for a list of tools, whose names are the input of the function. 

### Step 1. Preprocessing

For the datasets crawled from different sources in *[OutComes](https://github.com/talrasha/OpenTracingToolsMLR/tree/main/Dataset/Outcomes)*, use the functions, *getSentenceLevelDatasetAndTxtStackOverflow()*, *getSentenceLevelDatasetAndTxtMedium()* and *getSentenceLevelDatasetAndTxtDzone()* to tokenize and organize the text data into sentence level. 

### Step 2. Filtering

To ease the testing and visualization process, the source code is drafted in Jupyter Notebook (i.e., *[filteringTopicModeling.ipynb](https://github.com/talrasha/OpenTracingToolsMLR/blob/main/Scripts/filteringTopicModeling.ipynb)*) and was ran in Google Colaboratory. Use the manually labelled training data, [informative.txt](https://github.com/talrasha/OpenTracingToolsMLR/tree/main/Dataset/TrainingData/informative.txt) and [noninformative.txt](https://github.com/talrasha/OpenTracingToolsMLR/tree/main/Dataset/TrainingData/noninformative.txt), class *Semi_EM_MultinomialNB* provides the EMNB model training function. *MultinomialNB* model training function is imported from *sklearn.naive_bayes*. Block [13] - [14] test and visualize the different sizes of train-test data and the according performances (generating Figure 5). Block [25] - [27] train the NB model and mark each sentence by either information (i.e., 1) or noninformative (i.e., 0).

### Step 3. Topic Modeling

In the Block [40] of *[filteringTopicModeling.ipynb](https://github.com/talrasha/OpenTracingToolsMLR/blob/main/Scripts/filteringTopicModeling.ipynb)* function *compute_coherence_values()* calculates the *c_v* coherence measure. Using the *c_v* coherence measure, function *testTopicNumberK()* in Block [49] tests the different numbers of topics, e.g., Block [54] calculates and visulize such outcomes, which is also the method to obtain Figure 6. With the detected topic number (i.e., 9 topics in this case), Block [90] trains the LDA model. The topic model is saved in [TopicModel](https://github.com/talrasha/OpenTracingToolsMLR/tree/main/Dataset/TopicModel) folder.

### Step 4. Topic Mapping

### Step 5. Opinion Mining




