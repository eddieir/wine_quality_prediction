from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import time

tic = time.clock()

df = pd.read_csv('winequality-white.csv', header=0, sep=';')

X = df[list(df.columns)[:-1]]
y = df['quality']
print ("1")
X_train, X_test, y_train, y_test = train_test_split(X, y)
print ("splitted")
model_lin = svm.LinearSVC()
model_rbf = svm.SVC()
print ("made model")
model_lin.fit(X_train, y_train)
model_rbf.fit(X_train, y_train)
print ("fitting")
y_predict_lin = model_lin.predict(X_test)
y_predict_rbf = model_rbf.predict(X_test)
print ("predicted")
mse_lin = mean_squared_error(y_predict_lin, y_test)
mse_rbf = mean_squared_error(y_predict_rbf, y_test)
print ("mserd")
print (model_lin.score(X_test, y_test), "RMSE: " + str(mse_lin ** 0.5))
print (model_rbf.score(X_test, y_test), "RMSE: " + str(mse_rbf ** 0.5))
toc = time.clock()
print  ((toc - tic))