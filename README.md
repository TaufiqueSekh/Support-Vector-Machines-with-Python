# Support-Vector-Machines-with-Python

## What is an SVM?
Support vector machines are a set of supervised learning methods used for classification, regression, and outliers detection. All of these are common tasks in machine learning.

You can use them to detect cancerous cells based on millions of images or you can use them to predict future driving routes with a well-fitted regression model.

There are specific types of SVMs you can use for particular machine learning problems, like support vector regression (SVR) which is an extension of support vector classification (SVC).

The main thing to keep in mind here is that these are just math equations tuned to give you the most accurate answer possible as quickly as possible.

SVMs are different from other classification algorithms because of the way they choose the decision boundary that maximizes the distance from the nearest data points of all the classes. The decision boundary created by SVMs is called the maximum margin classifier or the maximum margin hyper plane.

## How an SVM works
A simple linear SVM classifier works by making a straight line between two classes. That means all of the data points on one side of the line will represent a category and the data points on the other side of the line will be put into a different category. This means there can be an infinite number of lines to choose from.

What makes the linear SVM algorithm better than some of the other algorithms, like k-nearest neighbors, is that it chooses the best line to classify your data points. It chooses the line that separates the data and is the furthest away from the closet data points as possible.

A 2-D example helps to make sense of all the machine learning jargon. Basically you have some data points on a grid. You're trying to separate these data points by the category they should fit in, but you don't want to have any data in the wrong category. That means you're trying to find the line between the two closest points that keeps the other data points separated.

So the two closest data points give you the support vectors you'll use to find that line. That line is called the decision boundary.
![image](https://user-images.githubusercontent.com/13853670/232039198-c1574fa9-893d-41d8-8a92-d032343cf4d5.png)

linear SVM

The decision boundary doesn't have to be a line. It's also referred to as a hyperplane because you can find the decision boundary with any number of features, not just two.
![image](https://user-images.githubusercontent.com/13853670/232039469-c7963130-630c-4168-a43c-62057a3ca358.png)

non-linear SVM using RBF kernel

## Types of SVMs
There are two different types of SVMs, each used for different things:

### Simple SVM: 
Typically used for linear regression and classification problems.
### Kernel SVM:
Has more flexibility for non-linear data because you can add more features to fit a hyperplane instead of a two-dimensional space.

### Why SVMs are used in machine learning
SVMs are used in applications like handwriting recognition, intrusion detection, face detection, email classification, gene classification, and in web pages. This is one of the reasons we use SVMs in machine learning. It can handle both classification and regression on linear and non-linear data.

Another reason we use SVMs is because they can find complex relationships between your data without you needing to do a lot of transformations on your own. It's a great option when you are working with smaller datasets that have tens to hundreds of thousands of features. They typically find more accurate results when compared to other algorithms because of their ability to handle small, complex datasets.

Here are some of the pros and cons for using SVMs.

 ### Pros
-- Effective on datasets with multiple features, like financial or medical data.
-- Effective in cases where number of features is greater than the number of data points.
-- Uses a subset of training points in the decision function called support vectors which makes it memory efficient.
-- Different kernel functions can be specified for the decision function. You can use common kernels, but it's also possible to specify custom kernels.
### Cons
-- If the number of features is a lot bigger than the number of data points, avoiding over-fitting when choosing kernel functions and regularization term is crucial.
-- SVMs don't directly provide probability estimates. Those are calculated using an expensive five-fold cross-validation.
-- Works best on small sample sets because of its high training time.
-- Since SVMs can use any number of kernels, it's important that you know about a few of them.

## Kernel functions

### Linear
These are commonly recommended for text classification because most of these types of classification problems are linearly separable.

The linear kernel works really well when there are a lot of features, and text classification problems have a lot of features. Linear kernel functions are faster than most of the others and you have fewer parameters to optimize.

Here's the function that defines the linear kernel:

f(X) = w^T * X + b

In this equation, w is the weight vector that you want to minimize, X is the data that you're trying to classify, and b is the linear coefficient estimated from the training data. This equation defines the decision boundary that the SVM returns.

### Polynomial
The polynomial kernel isn't used in practice very often because it isn't as computationally efficient as other kernels and its predictions aren't as accurate.

Here's the function for a polynomial kernel:

f(X1, X2) = (a + X1^T * X2) ^ b

This is one of the more simple polynomial kernel equations you can use. f(X1, X2) represents the polynomial decision boundary that will separate your data. X1 and X2 represent your data.

### Gaussian Radial Basis Function (RBF)
One of the most powerful and commonly used kernels in SVMs. Usually the choice for non-linear data.

Here's the equation for an RBF kernel:

f(X1, X2) = exp(-gamma * ||X1 - X2||^2)

In this equation, gamma specifies how much a single training point has on the other data points around it. ||X1 - X2|| is the dot product between your features.

### Sigmoid
More useful in neural networks than in support vector machines, but there are occasional specific use cases.

Here's the function for a sigmoid kernel:

f(X, y) = tanh(alpha * X^T * y + C)
In this function, alpha is a weight vector and C is an offset value to account for some mis-classification of data that can happen.

Others
There are plenty of other kernels you can use for your project. This might be a decision to make when you need to meet certain error constraints, you want to try and speed up the training time, or you want to super tune parameters.

Some other kernels include: ANOVA radial basis, hyperbolic tangent, and Laplace RBF.
