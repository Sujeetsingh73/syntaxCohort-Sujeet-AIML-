import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# =========================================================
# STEP 1: Create Synthetic Spam Detection Dataset
# =========================================================
np.random.seed(42)
num_samples = 500

# Features extracted from emails:
# 1. Word_Count: Total words in email
# 2. Contains_URL: 1 if link present, 0 otherwise
# 3. Urgent_Words_Count: Occurrences of words like 'free', 'win', 'urgent'
# 4. Capital_Letters_Ratio: Percentage of uppercase characters
word_count = np.random.randint(20, 500, num_samples)
contains_url = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
urgent_words = np.random.poisson(lam=1.5, size=num_samples)
capital_ratio = np.random.uniform(0.01, 0.40, num_samples)

# Rule for Spam (Target = 1 for Spam, 0 for Ham)
spam_probability = (contains_url * 0.4) + (urgent_words * 0.2) + (capital_ratio * 0.3)
y = (spam_probability > 0.45).astype(int)

X = pd.DataFrame({
    'Word_Count': word_count,
    'Contains_URL': contains_url,
    'Urgent_Words_Count': urgent_words,
    'Capital_Letters_Ratio': capital_ratio
})

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# STEP 2: Baseline Decision Tree Classifier
# =========================================================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("--- 1. BASIC DECISION TREE ACCURACY ---")
print(f"Accuracy: {accuracy_score(y_test, dt_pred) * 100:.2f}%\n")

# =========================================================
# STEP 3: Random Forest with Hyperparameter Tuning (GridSearchCV)
# =========================================================
# Hyperparameter grid to prevent overfitting
param_grid = {
    'n_estimators': [50, 100, 150],      # Number of trees in forest
    'max_depth': [3, 5, 10, None],       # Max depth of each tree
    'min_samples_split': [2, 5, 10],     # Minimum samples required to split a node
    'criterion': ['gini', 'entropy']     # Split quality criterion
}

rf_base = RandomForestClassifier(random_state=42)

# Set up Grid Search with 5-Fold Cross Validation
grid_search = GridSearchCV(
    estimator=rf_base, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best Random Forest Model
best_rf_model = grid_search.best_estimator_
rf_pred = best_rf_model.predict(X_test)

print("--- 2. HYPERPARAMETER TUNED RANDOM FOREST ---")
print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, rf_pred))

# =========================================================
# STEP 4: Feature Importance & Visualizations
# =========================================================
# Extract Feature Importances from tuned Random Forest
importances = best_rf_model.feature_importances_
feature_names = X.columns

# Plotting Confusion Matrix and Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit (Ham)', 'Spam'])
disp.plot(cmap='Greens', ax=axes[0])
axes[0].set_title("Random Forest - Confusion Matrix")

# Plot 2: Feature Importance Bar Chart
axes[1].barh(feature_names, importances, color='teal')
axes[1].set_xlabel("Importance Score")
axes[1].set_title("Feature Importance in Spam Detection")

plt.tight_layout()
plt.show()