import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, svm
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score, LeavePOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#load datasets
iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
data = iris.data
target = iris.target


print (data.shape)
print (target.shape)

new_column = np.ones([150,1], dtype = 'int')

#NewA i.e. after adding Ones
new_data = np.concatenate((new_column, data),1)
print (new_data.shape)

#ATranspose
new_data_transpose = np.transpose(new_data)
print (new_data_transpose.shape)

#(ATranspose * A)
product = np.matmul(new_data_transpose, new_data)
print (product.shape)

#INV(A)
matrixInverse = np.linalg.inv(product)
print (matrixInverse.shape)

#INV(AT.A).ATranspose
secondProd = np.matmul(matrixInverse, new_data_transpose)
print (secondProd.shape)

#Beta-Value
betaProduct = np.matmul(secondProd,target)
print (betaProduct.shape)

#prediction
prediction = np.matmul(new_data,betaProduct)
print (prediction.shape)

#sum-squared-error
sum_squared_error = 0
for i in range(0,150):
    squared_error = np.square(prediction[i] - target[i])
    sum_squared_error =  sum_squared_error + squared_error

print (sum_squared_error)

#accuracy
accuracy = 100 - sum_squared_error
print (accuracy)


#K - Fold Cross Validation

#Train_Test_Split is done and SVM algortihm is used
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.6, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clfagain = clf.score(X_test, y_test)
print (clfagain)

#When cv = 5
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print (scores)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))






from sklearn import metrics
scores = cross_val_score(
clf, iris.data, iris.target, cv=5, scoring='f1_macro')
print (scores)

from sklearn.model_selection import ShuffleSplit
n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
cross_val_score(clf, iris.data, iris.target, cv=cv)

#When cv =10
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
print (metrics.accuracy_score(iris.target, predicted))











