import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score #used for LinearRegression
from sklearn.metrics import accuracy_score #used for LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold



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



#LEAVE-ONE-OUT CROSS VALIDATION (LOOCV):

X_loo = X.head(10)
y_loo = y.head(10)

loo = LeaveOneOut()
loo_errors = []

for train_idx, val_idx in loo.split(X_loo):
    X_train_loo, X_val_loo = X_loo.iloc[train_idx], X_loo.iloc[val_idx]
    y_train_loo, y_val_loo = y_loo.iloc[train_idx], y_loo.iloc[val_idx]

    model_loo = LinearRegression()
    model_loo.fit(X_train_loo, y_train_loo)

    pred = model_loo.predict(X_val_loo)
    loo_errors.append((y_val_loo.values[0] - pred[0]) ** 2)

loo_mse = np.mean(loo_errors)
loo_rmse = np.sqrt(loo_mse)

print(f"LOOCV MSE: {loo_mse:.4f}")
print(f"LOOCV RMSE: {loo_rmse:.4f}")



#K-FOLD CROSS VALIDATION:

kf = KFold(n_splits= 10, shuffle=True, random_state=42)
kf_errors = []

for train_idx, val_idx in kf.split(X):
    X_train_kf, X_val_kf = X.iloc[train_idx], X.iloc[val_idx]
    y_train_kf, y_val_kf = y.iloc[train_idx], y.iloc[val_idx]

    model_kf = LinearRegression()
    model_kf.fit(X_train_kf, y_train_kf)

    pred = model_kf.predict(X_val_kf)
    kf_errors.append(mean_squared_error(y_val_kf, pred))

kf_mse = np.mean(kf_errors)
kf_rmse = np.sqrt(kf_mse)

print(f"KF MSE: {kf_mse:.4f}")
print(f"KF RMSE: {kf_rmse:.4f}")



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