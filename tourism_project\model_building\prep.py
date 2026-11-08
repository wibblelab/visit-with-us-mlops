# for data manipulation
import pandas as pd
# for splitting the data into train and test sets
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# Load the registered dataset from the repository data folder
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# --- Cleaning rules (from the EDA summary) ---
# Each rule is fixed rather than data-specific, so the same cleaning applies
# consistently whenever the pipeline is re-run on a new version of the dataset.

# 1. Drop identifier columns and Designation
cols_to_drop = ["Unnamed: 0", "CustomerID", "Designation"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 2. Correct inconsistent category label for 'Fe Male'
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 3. Remove duplicate records (checked after the ID columns are dropped)
n_before = len(df)
df = df.drop_duplicates()
print(f"Duplicate records removed: {n_before - len(df)}")

# Report missing values: none in the current data, and the model pipeline
# (XGBoost + OneHotEncoder) can handle them if a future dataset contains any
print(f"Missing values: {df.isnull().sum().sum()}")

# Define predictor (X) and target variable (y)
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split into train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,   # for reproducibility
    stratify=y,        # keeps the purchase rate consistent across both sets
)

# Save the splits to the working directory; the GitHub Actions workflow
# uploads these as an artifact for the model training job
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Cleaned dataset: {df.shape[0]} rows, {X.shape[1]} features")
print(f"Train set: {Xtrain.shape[0]} rows | Test set: {Xtest.shape[0]} rows")
print(f"Purchase rate - train: {ytrain.mean():.1%} | test: {ytest.mean():.1%}")
