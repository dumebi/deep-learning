import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


#read data

dataframe = pd.read_csv('challenge_dataset.txt',  sep=',', names = ["a", "b"])

x_values = dataframe[['a']]
y_values = dataframe[['b']]

X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.5, shuffle=True,  random_state=42)


#training
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)

score = r2_score(y_test, pred)
print score
#visualize
plt.scatter(x_values, y_values)
plt.plot(x_values, regressor.predict(x_values))
plt.show()