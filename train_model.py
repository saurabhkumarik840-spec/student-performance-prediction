# ============================================================
# STUDENT PERFORMANCE PREDICTION
# STEP 3.2 - MODEL TRAINING & COMPARISON
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_PATH = "data/StudentPerformanceFactors.csv"

MODEL_DIR = "models"
PLOT_DIR = "plots"

TARGET_COLUMN = "Exam_Score"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STUDENT PERFORMANCE PREDICTION")
print("MODEL TRAINING & COMPARISON")
print("=" * 60)

print("\n[1/8] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"\nDataset loaded successfully!")
print(f"Dataset Shape: {df.shape}")


# ------------------------------------------------------------
# 3. BASIC DATA ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 4. DATA CLEANING
# ------------------------------------------------------------

print("\n[2/8] Cleaning dataset...")

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Remove rows where target is missing
df = df.dropna(subset=[TARGET_COLUMN])

print(f"\nDataset Shape After Cleaning: {df.shape}")


# ------------------------------------------------------------
# 5. SPLIT FEATURES AND TARGET
# ------------------------------------------------------------

print("\n[3/8] Preparing features and target...")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print("\nTarget Column:")
print(TARGET_COLUMN)

print("\nNumber of Features:")
print(X.shape[1])


# ------------------------------------------------------------
# 6. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ------------------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ------------------------------------------------------------
# 7. TRAIN TEST SPLIT
# ------------------------------------------------------------

print("\n[4/8] Splitting dataset into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")


# ------------------------------------------------------------
# 8. PREPROCESSING PIPELINE
# ------------------------------------------------------------

print("\n[5/8] Creating preprocessing pipeline...")

# Numerical preprocessing
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# 9. DEFINE MACHINE LEARNING MODELS
# ------------------------------------------------------------

print("\n[6/8] Creating machine learning models...")

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE
    )
}


# ------------------------------------------------------------
# 10. TRAIN AND EVALUATE MODELS
# ------------------------------------------------------------

print("\n[7/8] Training models...")

results = []

trained_models = {}

predictions = {}


for model_name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training: {model_name}")
    print("-" * 60)

    # Create complete pipeline
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # Train model
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    # Store model
    trained_models[model_name] = pipeline

    # Store predictions
    predictions[model_name] = y_pred

    # Store results
    results.append(
        {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        }
    )

    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R2 Score : {r2:.4f}")


# ------------------------------------------------------------
# 11. MODEL COMPARISON
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

# Sort by R2 Score (higher is better)
results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# Save comparison results
comparison_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print(
    f"\nModel comparison saved to: {comparison_path}"
)


# ------------------------------------------------------------
# 12. SELECT BEST MODEL
# ------------------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

best_predictions = predictions[
    best_model_name
]


print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"\nBest Model: {best_model_name}"
)

print(
    f"Best R2 Score: {results_df.iloc[0]['R2 Score']:.4f}"
)

print(
    f"Best MAE: {results_df.iloc[0]['MAE']:.4f}"
)

print(
    f"Best RMSE: {results_df.iloc[0]['RMSE']:.4f}"
)


# ------------------------------------------------------------
# 13. SAVE ALL TRAINED MODELS
# ------------------------------------------------------------

print("\n[8/8] Saving trained models...")

for model_name, model in trained_models.items():

    # Convert model name into filename
    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".pkl"
    )

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Saved: {model_path}"
    )


# ------------------------------------------------------------
# 14. SAVE BEST MODEL
# ------------------------------------------------------------

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

joblib.dump(
    best_model,
    best_model_path
)

print(
    f"\nBest model saved to: {best_model_path}"
)


# ------------------------------------------------------------
# 15. SAVE MODEL INFORMATION
# ------------------------------------------------------------

model_info = {
    "best_model": best_model_name,
    "target_column": TARGET_COLUMN,
    "features": X.columns.tolist(),
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE
}

model_info_path = os.path.join(
    MODEL_DIR,
    "model_info.pkl"
)

joblib.dump(
    model_info,
    model_info_path
)

print(
    f"Model information saved to: {model_info_path}"
)


# ------------------------------------------------------------
# 16. ACTUAL VS PREDICTED GRAPH
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.6
)

# Perfect prediction line
min_value = min(
    y_test.min(),
    best_predictions.min()
)

max_value = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual Exam Score"
)

plt.ylabel(
    "Predicted Exam Score"
)

plt.title(
    f"Actual vs Predicted Exam Score\nBest Model: {best_model_name}"
)

plt.tight_layout()


actual_predicted_path = os.path.join(
    PLOT_DIR,
    "actual_vs_predicted_best_model.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300
)

plt.close()

print(
    f"\nActual vs Predicted graph saved to: "
    f"{actual_predicted_path}"
)


# ------------------------------------------------------------
# 17. MODEL PERFORMANCE COMPARISON GRAPH
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    results_df["Model"],
    results_df["R2 Score"]
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "R2 Score"
)

plt.title(
    "Model Performance Comparison"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()


comparison_plot_path = os.path.join(
    PLOT_DIR,
    "model_comparison_r2.png"
)

plt.savefig(
    comparison_plot_path,
    dpi=300
)

plt.close()

print(
    f"Model comparison graph saved to: "
    f"{comparison_plot_path}"
)


# ------------------------------------------------------------
# 18. FEATURE IMPORTANCE
# ------------------------------------------------------------

if best_model_name in [
    "Random Forest",
    "Gradient Boosting"
]:

    print(
        "\nGenerating feature importance graph..."
    )

    # Get trained model
    trained_estimator = best_model.named_steps[
        "model"
    ]

    # Get preprocessing step
    fitted_preprocessor = best_model.named_steps[
        "preprocessor"
    ]

    # Get feature names after encoding
    try:

        feature_names = (
            fitted_preprocessor
            .get_feature_names_out()
        )

        importances = (
            trained_estimator
            .feature_importances_
        )

        feature_importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances
            }
        )

        feature_importance_df = (
            feature_importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(15)
        )

        # Save feature importance CSV
        feature_importance_path = os.path.join(
            MODEL_DIR,
            "feature_importance.csv"
        )

        feature_importance_df.to_csv(
            feature_importance_path,
            index=False
        )

        # Plot
        plt.figure(
            figsize=(10, 7)
        )

        plt.barh(
            feature_importance_df["Feature"][::-1],
            feature_importance_df["Importance"][::-1]
        )

        plt.xlabel(
            "Importance"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            f"Top 15 Feature Importance\n{best_model_name}"
        )

        plt.tight_layout()

        feature_plot_path = os.path.join(
            PLOT_DIR,
            "feature_importance.png"
        )

        plt.savefig(
            feature_plot_path,
            dpi=300
        )

        plt.close()

        print(
            f"Feature importance saved to: "
            f"{feature_plot_path}"
        )

    except Exception as e:

        print(
            "\nCould not generate feature importance:"
        )

        print(e)


# ------------------------------------------------------------
# 19. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"\nDataset Size        : {df.shape}"
)

print(
    f"Training Samples    : {len(X_train)}"
)

print(
    f"Testing Samples     : {len(X_test)}"
)

print(
    f"Best Model          : {best_model_name}"
)

print(
    f"Best R2 Score       : "
    f"{results_df.iloc[0]['R2 Score']:.4f}"
)

print(
    f"Best MAE            : "
    f"{results_df.iloc[0]['MAE']:.4f}"
)

print(
    f"Best RMSE           : "
    f"{results_df.iloc[0]['RMSE']:.4f}"
)

print(
    f"\nBest Model File     : {best_model_path}"
)

print(
    f"Comparison CSV      : {comparison_path}"
)

print(
    f"Plots Directory     : {PLOT_DIR}"
)

print("\n" + "=" * 60)
print("READY FOR PREDICTION")
print("=" * 60)