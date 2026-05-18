import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("transactions.csv")

# -----------------------------------
# FEATURES & TARGET
# -----------------------------------

X = df[["count", "year", "quarter"]]
y = df["amount"]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# MODEL TRAINING
# -----------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------------
# PREDICTIONS
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# EVALUATION METRICS
# -----------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION RESULTS")
print("--------------------------------")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)

# -----------------------------------
# ACTUAL vs PREDICTED VISUALIZATION
# -----------------------------------

plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual Amount")
plt.ylabel("Predicted Amount")

plt.title("Actual vs Predicted Transaction Amount")

plt.show()

# -----------------------------------
# RESIDUAL ANALYSIS
# -----------------------------------

residuals = y_test - y_pred

plt.figure(figsize=(8,5))

sns.histplot(residuals, bins=50, kde=True, color="purple")

plt.title("Residual Distribution")

plt.show()