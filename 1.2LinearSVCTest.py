#Purpose: To make a simple example of creating a prediction using a SVM with sklearn. Will pick which grouping a datapoint fits into.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use("ggplot")

'''
#quick example to graph scatter plot
x = [1,5,1.5,8,1,9]
y = [2,8,1.8,8,0.6,11]
plt.scatter(x,y)
plt.show()
'''

X = np.array([[1,2],
            [5,8],
            [1.5,1.8],
            [8,8],
            [1,0.6],
            [9,11]])

y = [0,1,0,1,0,1]
#svm stands for support vector machine. C is a magical number default at 1 anyways
clf = svm.SVC(kernel ='linear', C = 1.0)
clf.fit(X,y)

#can change the 2 numbers below and it will predict whether a 1 or 0 group based on the test data above.
#Smaller numbers (closer to 0) will result in a 0 prediction. Larger numbers (closer to 10) will result in a 1 prediction
print(clf.predict([[2.58,1.76]]))




#all of this below isn't really important for svm. wouldn't usually graph it unless only a few features
##w is a special coefficient
#w = clf.coef_[0]
#print(w)

##textbook algorithm for making the line
#a = -w[0]/w[1]

#xx = np.linspace(0,12)
#yy = a * xx - clf.intercept_[0]/w[1]

#h0 = plt.plot(xx,yy,'k-',label ="non weighted div")

#plt.scatter(X[:,0], X[:, 1], c = y)
#plt.legend()
#plt.show()