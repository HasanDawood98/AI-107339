import pandas as pd
from sklearn.naive_bayes import MultinomialNB

#Read CSV Files
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

#Get Labels and remove them from Train data
YTrain = trainData.Survived;
trainData.drop('Survived',inplace=True,axis=1)

#Convert Data to Features
# trainData.drop('Name',inplace=True,axis=1)
# trainData.drop('Cabin',inplace=True,axis=1)
# trainData.drop('Ticket',inplace=True,axis=1)
# testData.drop('Name',inplace=True,axis=1)
# testData.drop('Cabin',inplace=True,axis=1)
# testData.drop('Ticket',inplace=True,axis=1)
#Convert String based columns to integer classes
trainData["Sex"] = trainData["Sex"].replace(['female','male'],[0,1])
testData["Sex"] = testData["Sex"].replace(['female','male'],[0,1])
trainData["Embarked"] = trainData["Embarked"].replace(['S','Q','C'],[0,1,2])
testData["Embarked"] = testData["Embarked"].replace(['S','Q','C'],[0,1,2])

#Remove Nan Values from Train
trainData.fillna(value=0,inplace=True)
#Dummy values in Test for all NaN
testData.fillna(value=trainData['Age'].mean(),inplace=True)
testData.fillna(value=trainData['Fare'].mean(),inplace=True)

#Test
print(YTrain.shape)
print(trainData)
print(testData.shape)

#Train
mnb = MultinomialNB();
mnb.fit(trainData,YTrain)
#Predictions
predictions = mnb.predict(testData)
print(predictions.shape)

submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('submission.csv', index=False)