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
  * **[Step 0. Data Crawling](#datacrawling)**
  * **[Step 1. Preprocessing](#preprocessing)**
  * **[Step 2. Filtering](#filtering)**
  * **[Step 3. Topic Modeling](#topicmodeling)**
  * **[Step 4. Topic Mapping](#topicmapping)**
  * **[Step 5. Opinion Mining](#opinionmining)**


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
* [topicModeling.py](https://github.com/talrasha/OpenTracingToolsMLR/blob/main/Scripts/topicModeling.py): The script to conduct topic modeling on textual data as well as related plotting functions 

## How to Use

