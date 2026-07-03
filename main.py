import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# Student Marks Prediction using Linear Regression
# ==========================================================

# Dataset
data = {
    "Study-Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Marks": [40, 45, 50, 56, 60, 64, 70, 85, 98]
}

df = pd.DataFrame(data)

print("=" * 50)
print("DATASET")
print("=" * 50)
print(df)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nCorrelation Matrix")
print(df.corr())

# ==========================================================
# Features and Target
# ==========================================================

X = df[["Study-Hours"]]
y = df["Marks"]

# ==========================================================
# Train Model
# ==========================================================

model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# ==========================================================
# Comparison Table
# ==========================================================

comparison = pd.DataFrame({
    "Study Hours": X.values.flatten(),
    "Actual Marks": y,
    "Predicted Marks": y_pred.round(2)
})

print("\n")
print("=" * 50)
print("ACTUAL VS PREDICTED")
print("=" * 50)
print(comparison)

# ==========================================================
# Model Evaluation
# ==========================================================

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n")
print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ==========================================================
# Model Equation
# ==========================================================

m = model.coef_[0]
c = model.intercept_

print("\nRegression Equation")
print(f"Marks = {m:.2f} × Study Hours + {c:.2f}")

# ==========================================================
# Visualization
# ==========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    X,
    y,
    color="blue",
    s=80,
    label="Actual Data"
)

plt.plot(
    X,
    y_pred,
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.title("Student Marks Prediction using Linear Regression")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.grid(True)
plt.legend()

plt.show()

# ==========================================================
# User Prediction
# ==========================================================

while True:

    study_hours = float(input("\nEnter Study Hours (0-9): "))

    if study_hours < 0 or study_hours > 9:
        print("Invalid Input! Please enter values between 0 and 9.")
    else:
        prediction = model.predict([[study_hours]])
        print(f"Predicted Marks = {prediction[0]:.2f}")
        break