import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

data = pd.read_csv("SalaryData.csv")

X = data["YearsExperience"].values
Y = data["Salary"].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
mean_x = np.mean(X_train)
mean_y = np.mean(Y_train)

numerator = np.sum((X_train - mean_x) * (Y_train - mean_y))
denominator = np.sum((X_train - mean_x)**2)
b1 = numerator / denominator
b0 = mean_y - b1 * mean_x

print("Custom Model Coefficients")
print("Slope:", b1)
print("Intercept:", b0)
Y_pred_custom = b0 + b1 * X_test

rmse_custom = np.sqrt(mean_squared_error(Y_test, Y_pred_custom))
mae_custom = mean_absolute_error(Y_test, Y_pred_custom)
r2_custom = r2_score(Y_test, Y_pred_custom)

print("\nCustom Model Performance")
print("RMSE:", rmse_custom)
print("MAE:", mae_custom)
print("R2 Score:", r2_custom)

X_train_reshape = X_train.reshape(-1,1)
X_test_reshape = X_test.reshape(-1,1)

model = LinearRegression()
model.fit(X_train_reshape, Y_train)

Y_pred_sklearn = model.predict(X_test_reshape)

print("\nSklearn Model Coefficients")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

rmse_sklearn = np.sqrt(mean_squared_error(Y_test, Y_pred_sklearn))
mae_sklearn = mean_absolute_error(Y_test, Y_pred_sklearn)
r2_sklearn = r2_score(Y_test, Y_pred_sklearn)

print("\nSklearn Model Performance")
print("RMSE:", rmse_sklearn)
print("MAE:", mae_sklearn)
print("R2 Score:", r2_sklearn)

plt.scatter(X, Y, color='blue')
plt.plot(X, b0 + b1*X, color='red')
plt.title("Custom Linear Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X, Y, color='blue')
plt.plot(X, model.predict(X.reshape(-1,1)), color='green')
plt.title("Sklearn Linear Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()