# for data manipulation
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization and experiment tracking
import joblib
import mlflow

# MLflow server runs locally on port 5000 (started in a terminal locally, and by the workflow in GitHub Actions)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-wellness-package")

# Xtrain/Xtest/ytrain/ytest are produced by prep.py (downloaded from the data-prep job's artifact in GitHub Actions)
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

# List of numerical features (ordinal and binary features kept numeric, as their order is meaningful)
numeric_features = [
    "Age",                       # Age of the customer
    "CityTier",                  # City category (1-3), ordinal
    "DurationOfPitch",           # Duration of the sales pitch
    "NumberOfPersonVisiting",    # Total people on the trip
    "NumberOfFollowups",         # Follow-ups after the pitch
    "PreferredPropertyStar",     # Preferred hotel rating (3-5), ordinal
    "NumberOfTrips",             # Average trips per year
    "Passport",                  # Holds a valid passport (0/1)
    "PitchSatisfactionScore",    # Satisfaction with the pitch (1-5), ordinal
    "OwnCar",                    # Owns a car (0/1)
    "NumberOfChildrenVisiting",  # Children under 5 on the trip
    "MonthlyIncome",             # Gross monthly income
]

# List of categorical features (one-hot encoded)
categorical_features = [
    "TypeofContact",   # Company Invited / Self Enquiry
    "Occupation",      # Customer's occupation
    "Gender",          # Customer's gender
    "ProductPitched",  # Package pitched to the customer
    "MaritalStatus",   # Customer's marital status
]

# Set the class weight to handle class imbalance (ratio of non-buyers to buyers, ~4.2)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Class weight (scale_pos_weight): {class_weight:.2f}")

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter grid, kept modest so the pipeline runs quickly on GitHub Actions
param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning, optimising for recall so that as few likely buyers as possible are missed
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        # Log each combination as a separate (nested) MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)

    classification_threshold = 0.45
    mlflow.log_param("classification_threshold", classification_threshold)

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)
    print("Training set performance:")
    print(classification_report(ytrain, y_pred_train))
    print("Test set performance:")
    print(classification_report(ytest, y_pred_test))

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })

    # Save next to app.py so the Streamlit app can load it directly, and log
    # it as an MLflow artifact for traceability
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
