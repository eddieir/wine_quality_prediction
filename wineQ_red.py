import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error 

df = pd.read_csv('winequality-redT.csv')

print(df.head())

print(df.describe())

# there are no categorical variables. each feature is a number. Regression problem. 
# Given the set of values for features, we have to predict the quality of wine.
# finding correlation of each feature with our target variable - quality
correlations = df.corr()['quality']
print(correlations)



sns.heatmap(df.corr())
#plt.show()

def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations

# taking features with correlation more than 0.05 as input x and quality as target variable y
features = get_features(0.05)
print(features)
x = df[features]
y = df['quality']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)
# x_train.shape
# x_test.shape
# y_train.shape
y_test.shape




# fitting linear regression to training data
regressor = LinearRegression()
regressor.fit(x_train,y_train)



LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



# this gives the coefficients of the 10 features selected above. 
regressor.coef_


train_pred = regressor.predict(x_train)


test_pred = regressor.predict(x_test)


train_rmse = mean_squared_error(train_pred, y_train) ** 0.5
print(train_rmse)

test_rmse = mean_squared_error(test_pred, y_test) ** 0.5
print(test_rmse)

# The root-mean-square error (RMSE) is a frequently used measure of the differences between values (sample and population values) predicted by a model and the values actually observed. 
# The RMSE for your training and your test sets should be very similar if you have built a good model. 
# If the RMSE for the test set is much higher than that of the training set, it is likely that you've badly over fit the data

# rounding off the predicted values for test set
predicted_data = np.round_(test_pred)
predicted_data

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))

'''Mean Absolute Error: 0.484434075598
Mean Squared Error: 0.393804134629
Root Mean Squared Error: 0.627538153923
'''


coeffecients = pd.DataFrame(regressor.coef_,features)
coeffecients.columns = ['Coeffecient']
print(coeffecients)
#These numbers mean that holding all other features fixed, a 1 unit increase in suplhates will lead to an increase of 0.8 in Quality of wine, and similarly for the other features
#These numbers mean that holding all other features fixed, a 1 unit increase in volatile acidity will lead to a decrease of 0.99 in Quality of wine, and similarly for the other features


