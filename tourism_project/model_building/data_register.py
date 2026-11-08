import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns (from the data dictionary) are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Flag any columns not in the data dictionary (e.g. a saved index column) for removal in data prep
extra = [c for c in df.columns if c not in expected_columns]

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Columns not in data dictionary:", extra if extra else "None")
print(f"Total missing values: {df.isnull().sum().sum()}")
print("Package purchased (ProdTaken) distribution:")
print(df["ProdTaken"].value_counts())
