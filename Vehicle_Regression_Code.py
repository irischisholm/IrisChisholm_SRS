import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score #used for LinearRegression
from sklearn.metrics import accuracy_score #used for LogisticRegression



#PART A: Linear Regression (predicts price of car)

df = pd.read_csv("/Users/irisyura/Downloads/vehicle_price_prediction.csv")
X = df[['year', 'engine_hp']] #df means DataFrame so that it actually collects data from a table, not a list of strings
y = df['mileage']

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

print(model.intercept_)
print(model.coef_)

y_pred = model.predict(x_test)

predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions.head())

r2 = r2_score(y_test, y_pred)
print(f"R-squared score: {r2}") #the closer the R^2 score is to 1, the more accurate the model is!



#PART B: Logistic Regression (predicts manual vs. automatic transmission)

X = df[['year', 'mileage', 'engine_hp']]
y = df['transmission'] #BINARY because it's either manual or automatic

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

print(model.intercept_)
print(model.coef_)

y_pred = model.predict(x_test)

predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions.head())

print("Accuracy:", accuracy_score(y_test, y_pred)) #the closer the accuracy score is to 1, the more accurate the model is!