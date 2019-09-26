from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import pandas as pd

df = pd.read_csv('winequality-white.csv', header=0, sep=';')



X = df[list(df.columns)[:-1]]
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print ('Score:', model.score(X_test, y_test))
print ('RMSE:', mean_squared_error(y_predict, y_test) ** 0.5)


#parameters = [{'weights': ['uniform', 'distance'], 'n_neighbors': [40, 60, 80, 100, 120, 140]}]
#grid = GridSearchCV(model, parameters)
#grid.fit(X_train, y_train)
#print(grid)

#print(grid.best_score_)
#print(grid.best_estimator_.weights)
#print(grid.best_estimator_.n_neighbors)