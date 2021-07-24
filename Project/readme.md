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



