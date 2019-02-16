import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

#load datasets
iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
y = iris.target
z = iris.data

#split the training and test sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

#fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
predictions[0:5]
print (predictions)

# The coefficients
print('Coefficients: \n', lm.coef_)

#Mean Squared Error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, predictions))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, predictions))

#accuracy_Score
score = model.score(X_test, y_test)
print (score)




