import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

# =========================================================
# STEP 1: Dataset Generation (Energy Consumption Data)
# =========================================================
np.random.seed(42)
num_samples = 200

# Features: Building Size (sq ft) and Average Temperature (°C)
size = np.random.uniform(500, 3500, num_samples)
temp = np.random.uniform(10, 35, num_samples)

# Target: Continuous Energy Consumption (kWh)
energy = (size * 0.05) + (temp * 1.5) + np.random.normal(0, 10, num_samples)

df_energy = pd.DataFrame({'Size': size, 'Temperature': temp, 'Energy_kWh': energy})

# =========================================================
# PART 1: Linear Regression (Predicting Continuous Value)
# =========================================================
X = df_energy[['Size', 'Temperature']]
y_reg = df_energy['Energy_kWh']

# Train-Test Split for Regression
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Train Model
linear_model = LinearRegression()
linear_model.fit(X_train_r, y_train_r)

# Predictions & Metrics
y_pred_r = linear_model.predict(X_test_r)
mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print("--- PART 1: LINEAR REGRESSION RESULTS ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-Squared Score (R²):    {r2:.4f}")
print(f"Coefficients:             {linear_model.coef_}")
print(f"Intercept:                {linear_model.intercept_:.2f}")
print("\n" + "="*50 + "\n")

# =========================================================
# PART 2: Logistic Regression (Classification Task)
# =========================================================
# Binary target: 1 if Energy > Median, else 0
y_cls = (y_reg > y_reg.median()).astype(int)

# Train-Test Split for Classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)

# Train Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_c, y_train_c)

# Predictions & Metrics
y_pred_c = logistic_model.predict(X_test_c)
acc = accuracy_score(y_test_c, y_pred_c)
cm = confusion_matrix(y_test_c, y_pred_c)

print("--- PART 2: LOGISTIC REGRESSION RESULTS ---")
print(f"Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test_c, y_pred_c))

# Display Confusion Matrix Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Energy', 'High Energy'])
disp.plot(cmap='Blues')
plt.title("Logistic Regression - Confusion Matrix")
plt.show()