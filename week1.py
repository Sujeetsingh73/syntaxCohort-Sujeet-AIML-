import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# Step 1: Create a Dummy "Messy" Housing Dataset
# ---------------------------------------------------------
raw_data = {
    'SquareFeet': [1500, 2000, np.nan, 2500, 12000, 1800, 2200, 1600],  # 12000 is an outlier, NaN is missing
    'Bedrooms': [3, 4, 3, np.nan, 5, 3, 4, 2],                        # Missing value
    'Neighborhood': ['Suburbs', 'City', 'City', 'Suburbs', 'Rural', np.nan, 'Rural', 'City'], # Categorical + missing
    'Price': [300000, 400000, 320000, 500000, 1500000, 350000, 420000, 280000] # Target variable
}

df = pd.DataFrame(raw_data)
print("--- Raw Messy Dataset ---")
print(df)
print("\n" + "="*50 + "\n")

# ---------------------------------------------------------
# Step 2: Handle Outliers (IQR Method)
# ---------------------------------------------------------
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Removing extreme outliers from numerical feature 'SquareFeet'
df_cleaned = remove_outliers_iqr(df.dropna(subset=['SquareFeet']), 'SquareFeet')
# Re-align original index
df = df.loc[df_cleaned.index]

# ---------------------------------------------------------
# Step 3: Separate Features (X) and Target (y)
# ---------------------------------------------------------
X = df.drop(columns=['Price'])
y = df['Price']

# Define numerical and categorical columns
num_features = ['SquareFeet', 'Bedrooms']
cat_features = ['Neighborhood']

# ---------------------------------------------------------
# Step 4: Build Preprocessing Pipeline (Scikit-Learn)
# ---------------------------------------------------------
# Numerical Pipeline: Impute missing values with Median -> Standardize scale
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Pipeline: Impute missing values with Most Frequent -> One-Hot Encode
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine Pipelines using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# ---------------------------------------------------------
# Step 5: Fit and Transform the Data
# ---------------------------------------------------------
X_processed = preprocessor.fit_transform(X)

# Retrieve feature names after One-Hot Encoding
encoded_cat_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_features)
all_feature_names = num_features + list(encoded_cat_cols)

# Convert back to clean DataFrame for inspection
X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

print("--- Preprocessed & Normalized Features (Training Ready) ---")
print(X_processed_df)

# ---------------------------------------------------------
# Step 6: Train-Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed_df, y, test_size=0.2, random_state=42
)

print("\n" + "="*50 + "\n")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape:  {X_test.shape}")