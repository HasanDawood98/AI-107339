# AI-107338 Summer 2021  # 
# Final Ai-Project-Repository #
## Titanic - Machine Learning from Disaster  ##

### PROJECT MEMBERS ###
StdID | Name
------------ | -------------
**64181** | **Hafiz Ali Hammad Ansari (338)**
62886 | Hasan Dawood (339)
64290 | Ubaid Ullah (339)

## Description ##
In this phase we are expolring three new models to achieve better accuracy and get familiar with them.

### 1. SVM ###
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

#### The advantages of support vector machines are: ####

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

#### The disadvantages of support vector machines include: ####

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

### 2. Linear  Classifiers(LR) ###
Linear classifiers classify data into labels based on a linear combination of input features. Therefore, these classifiers separate data using a line or plane or a hyperplane (a plane in more than 2 dimensions). They can only be used to classify data that is linearly separable. They can be modified to classify non-linearly separable data
#### Logistic Regression ####
In Logistic regression, we take weighted linear combination of input features and pass it through a sigmoid function which outputs a number between 1 and 0. Unlike perceptron, which just tells us which side of the plane the point lies on, logistic regression gives a probability of a point lying on a particular side of the plane. The probability of classification will be very close to 1 or 0 as the point goes far away from the plane. The probability of classification of points very close to the plane is close to 0.5


### 3. KNN ###
The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.
* Birds of a feather flock together. *

#### Advantages ####
The algorithm is simple and easy to implement.
There’s no need to build a model, tune several parameters, or make additional assumptions.
The algorithm is versatile. It can be used for classification, regression, and search (as we will see in the next section).
#### Disadvantages ####
The algorithm gets significantly slower as the number of examples and/or predictors/independent variables increase.

** In order to achieve better accuracy and cross check our parameters and model we implement cross validation **
### Cross-validation ###
Here we describe cross-validation: one of the fundamental methods in machine learning for method assessment and picking parameters in a prediction or machine learning task. Suppose we have a set of observations with many features and each observation is associated with a label. We will call this set our training data. Our task is to predict the label of any new samples by learning patterns from the training data. For a concrete example, let’s consider gene expression values, where each gene acts as a feature. We will be given a new set of unlabeled data (the test data) with the task of predicting the tissue type of the new samples.

If we choose a machine learning algorithm with a tunable parameter, we have to come up with a strategy for picking an optimal value for this parameter. We could try some values, and then just choose the one which performs the best on our training data, in terms of the number of errors the algorithm would make if we apply it to the samples we have been given for training. However, we have seen how this leads to over-fitting.

** we are using K-fold **

#### K-Fold ####

K-fold cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.

The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=10 becoming 10-fold cross-validation.
 
##### Algo we used for K-fold #####
model-name = ModelFunctionScikit() 


scores = cross_val_score(Mul, X_train, Y_train, cv=10, scoring = "accuracy")


print  Scores


print  Mean


print  Standard Deviation



### Accuracy Results We Acheived From Above Mentioned Models ####

#### Linear Classifiers(LR) ####

![LinearClassification_Score](https://user-images.githubusercontent.com/38988469/126864543-a16fb3ba-abf1-4ce7-a916-c5b3ddbd3541.PNG)


#### KNN ####

![knn](https://user-images.githubusercontent.com/38988469/126864553-b16db85e-d278-4156-943e-73d21085c94c.png)


#### SVM ####

![SVM_Score](https://user-images.githubusercontent.com/38988469/126864559-f39b6cda-5384-4390-9c07-e9ba0a9a260b.PNG)

