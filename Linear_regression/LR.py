import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Loading data from csv file
data = pd.read_csv("/ML/Linear_regression/data/Fish.csv")

# Selecting required features
X_y_values = data.iloc[:, 5:7].values
# Fish Height
X = X_y_values[:, 0].reshape(-1, 1)
# Fish width
Y = X_y_values[:, 1].reshape(-1, 1)
print(X.shape, Y.shape)

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=5)

# Model creation
reg = LinearRegression()
# Model training
reg = reg.fit(x_train, y_train)
# Model testing
Y_pred = reg.predict(x_test)

print("Slope of line (weight) : ", reg.coef_)
print("intercepting on Y-Axis : ", reg.intercept_)
print("accuracy of model : ", metrics.explained_variance_score(y_test, Y_pred))

val = float(input('Enter Height of the fish: \n'))
print("Width = ",float(reg.predict([[val]])))



