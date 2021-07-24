from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC, LinearSVC 
from sklearn.model_selection import cross_val_score 
from sklearn.naive_bayes import MultinomialNB

import pandas as pd
import re 
import numpy as np

trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

total = trainData.isnull().sum().sort_values(ascending=False) 
percent_1 = trainData.isnull().sum()/trainData.isnull().count()*100 
percent_2 = (round(percent_1, 1)).sort_values(ascending=False) 
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(missing_data)

trainData = trainData.drop(['PassengerId'], axis=1)


deck = {"U": 1, "C": 2, "B": 3, "D": 4, "E": 5, "F": 6, "A": 7, "G": 8} 
data = [trainData, testData]

for dataset in data: 
    dataset['Cabin'][dataset.Cabin.isnull()] = 'U0' 
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group()) 
    dataset['Deck'] = dataset['Deck'].map(deck) 
    dataset['Deck'] = dataset['Deck'].fillna(0) 
    dataset['Deck'] = dataset['Deck'].astype(int) 
trainData = trainData.drop(['Cabin'], axis=1) 
testData = testData.drop(['Cabin'], axis=1)


data = [trainData, testData] 
for dataset in data: 
    mean = dataset["Age"].mean() 
    std = dataset["Age"].std() 
    is_null = dataset["Age"].isnull().sum() 
    rand_age = np.random.randint(mean - std, mean + std, size = is_null) 
    age_slice = dataset["Age"].copy() 
    age_slice[np.isnan(age_slice)] = rand_age 
    dataset["Age"] = age_slice 
    dataset["Age"] = trainData["Age"].astype(int) 
    trainData["Age"].isnull().sum()

trainData['Embarked'].describe() 
common_value = 'S' 
data = [trainData, testData]

for dataset in data: 
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

data = [trainData, testData] 
for dataset in data: 
    dataset['Fare'] = dataset['Fare'].fillna(0) 
    dataset['Fare'] = dataset['Fare'].astype(int)

data = [trainData, testData] 
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 
for dataset in data: # extract titles 
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+).', expand=False) 
    # replace titles with a more common title or as Rare 
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare') 
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss') 
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss') 
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs') 
    # convert titles into numbers 
    dataset['Title'] = dataset['Title'].map(titles) 
    # filling NaN with 0, to get safe 
    dataset['Title'] = dataset['Title'].fillna(0) 
trainData = trainData.drop(['Name'], axis=1) 
testData = testData.drop(['Name'], axis=1)

genders = {"male": 0, "female": 1} 
data = [trainData, testData] 
for dataset in data: 
    dataset['Sex'] = dataset['Sex'].map(genders)

trainData = trainData.drop(['Ticket'], axis=1) 
testData = testData.drop(['Ticket'], axis=1)

ports = {"S": 0, "C": 1, "Q": 2} 
data = [trainData, testData]

for dataset in data: 
    dataset['Embarked'] = dataset['Embarked'].map(ports)

data = [trainData, testData] 
for dataset in data: 
    dataset['Age'] = dataset['Age'].astype(int) 
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0 
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 22), 'Age'] = 1 
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 33), 'Age'] = 2 
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 44), 'Age'] = 3 
    dataset.loc[(dataset['Age'] > 44) & (dataset['Age'] <= 55), 'Age'] = 4 
    dataset.loc[(dataset['Age'] > 55) & (dataset['Age'] <= 66), 'Age'] = 5 
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
print(trainData['Age'].value_counts())

trainData['Fare'] = trainData['Fare'].astype(int) 
trainData['FareBand'] = pd.qcut(trainData['Fare'], 6) 
trainData = trainData.drop(['FareBand'], axis=1) 
data = [trainData, testData] 
for dataset in data: 
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0 
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1 
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2 
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare'] = 3 
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare'] = 4 
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5 
    dataset['Fare'] = dataset['Fare'].astype(int)

data = [trainData, testData] 
for dataset in data: 
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']


data = [trainData, testData] 
for dataset in data: 
    '''dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)'''

X_train = trainData.drop("Survived", axis=1) 
Y_train = trainData["Survived"] 
X_test = testData.drop("PassengerId", axis=1).copy()


knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train) 
Y_pred = knn.predict(X_test) 
acc_knn = round(knn.score(X_train, Y_train) * 100, 2) 
print("kNN accuracy =",round(acc_knn,2,), "%") 
print(Y_pred.shape) 
print(Y_pred)

submission = pd.DataFrame({ "PassengerId": testData["PassengerId"], "Survived": Y_pred }) 
submission.to_csv('KNN_submission.csv', index=False)

knn = KNeighborsClassifier(n_neighbors = 3) 
scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring = "accuracy") 
print("Scores:\n", pd.Series(scores)) 
print("Mean:", scores.mean()) 
print("Standard Deviation:", scores.std())