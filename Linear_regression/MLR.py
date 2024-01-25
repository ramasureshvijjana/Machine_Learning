import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import  numpy as np

data = pd.read_csv("/ML/Linear_regression/data/Fish.csv")
x_y_values = data.iloc[:, 1:].values
x= x_y_values[:,1:5]
y= x_y_values[:,0].reshape(-1,1)
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=.3)

mlg = LinearRegression()
mlg = mlg.fit(x_train, y_train)
y_pred = mlg.predict(x_test)
print(x_train.shape, x_test.shape,y_train.shape, y_test.shape, y_pred.shape)
m = mlg.coef_
c = mlg.intercept_
print("Slope of line or weight : ", m)
print("intercepting on Y-Axis : ", c)


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("accuracy of model : ", metrics.explained_variance_score(y_test, y_pred))