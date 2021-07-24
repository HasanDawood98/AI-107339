# AI-107338 Summer 2021 : Assignment 03 Folder #

### PROJECT MEMBERS ###
StdID | Name
------------ | -------------
**64181** | **Hafiz Ali Hammad Ansari (338)**
62886 | Hasan Dawood (339)
64290 | Ubaid Ullah (339)

## Description ##
<p> This folder contains assignment03 submitted to AI-107338. </p>

## Multinomial Naive Bayes ##
Multinomial Naive Bayes is one of the most popular supervised learning classifications that is used for the analysis of the categorical text data. Multinomial Naive Bayes algorithm is a probabilistic learning method that is mostly used in Natural Language Processing (NLP). The algorithm is based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output.

## Assignment Description ##
In this task,
* We download the train.csv and test.csv file from the Kaggle Titanic competition data.
* We extract features from sownloaded data.
* Then delete the name column and other text based columns which were not easily converted to numeric data. 
* For gender column we replaced male and female with 0s and 1s respectively.
* And replaced S, Q and C in the embarked column to 0s, 1s and 2s. 
* Then Import the features in python.
* Then used sci-kit learn and multinomial Naïve Bayes to train our first model on the complete train.csv file
* Used the test.csv file and the trained model to make prediction for each passenger in test.csv file.
* And In the end, Submit predictions on Kaggle and got Score 0.6533

## Kaggle Score Screenshot ##
![score](https://user-images.githubusercontent.com/38988469/126226170-9a657280-efc4-47b6-a454-060a1d43a69e.png)

#### Libraries Used ####
[pandas] (https://pandas.pydata.org/)
