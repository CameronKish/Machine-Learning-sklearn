#Purpose: To make a simple example of creating a prediction using a SVM with sklearn. Will predict the number based on the image provided.

import matplotlib.pyplot as plt
#sklearn used for the machine learning
from sklearn import datasets
from sklearn import svm

#our dataset
digits = datasets.load_digits()

#gamma is "leap size" down graph to get to the answer accurately and quickly. Smaller means more steps to get to answer.
clf = svm.SVC(gamma = 0.001, C=100)

#training the data, leaving the last 10 for testing
x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

#this number does not actually represent the number we are predicting. Just a reference to an image in the dataset at sklearn
#pick any number from 0 to 1797 (negatives also work)
prediction_digit = 555

print('Prediction:', clf.predict(digits.data[[prediction_digit]].reshape(1,-1)))

#show the image
plt.imshow(digits.images[prediction_digit], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
