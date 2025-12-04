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
from sklearn.linear_model import Ridge, RidgeCV, Lasso, ElasticNet



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

X_loo = X.head(10000)
y_loo = y.head(10000)

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




#RIDGE REGRESSION:

ridgeReg = Ridge(alpha=10) #having a varying lambda means that the value of lambda is being changed to find the "best" model; as lambda increases, it shrinks the model's irrelevant coefficients to make the model more accurate
ridgeReg.fit(x_train, y_train)
train_score_ridge = ridgeReg.score(x_train, y_train)
test_score_ridge = ridgeReg.score(x_test, y_test)
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))


#LASSO:

LassoReg = Lasso(alpha=10)
LassoReg.fit(x_train, y_train)
train_score_lasso = LassoReg.score(x_train, y_train)
test_score_lasso = LassoReg.score(x_test, y_test)
print("The train score for LASSO model is {}".format(train_score_lasso))
print("The test score for LASSO model is {}".format(test_score_lasso))

Lasso_r2 = r2_score(y_test, LassoReg.predict(x_test))
print(f"The R-squared score for LASSO model is: {Lasso_r2}")


#ELASTIC NET:

ElasticNetReg = ElasticNet(alpha=10, l1_ratio=0.5) #weight is the parameter that controls whether the Ridge-Regression model or the LASSO model is more prominent
ElasticNetReg.fit(x_train, y_train)
train_score_elastic = ElasticNetReg.score(x_train, y_train)
test_score_elastic = ElasticNetReg.score(x_test, y_test)
print("The train score for Elastic Net model is {}".format(train_score_elastic))
print("The test score for Elastic Net model is {}".format(test_score_elastic))

Elastic_r2 = r2_score(y_test, ElasticNetReg.predict(x_test))
print(f"The R-squared score for Elastic Net model is: {Elastic_r2}")
