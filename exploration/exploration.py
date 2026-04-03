# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: vroom-forecast
#     language: python
#     name: vroom-forecast
# ---

# %% [markdown]
# # Vroom Forecast — Exploration Notebook
#
# **Goal:** Understand which vehicle attributes drive the total number of reservations.
#
# Datasets:
# - `vehicles.csv` — 1000 vehicles with their attributes
# - `reservations.csv` — ~6400 completed reservations
#
# Plan:
# 1. Load and inspect both datasets
# 2. Aggregate reservations per vehicle and merge with vehicle features
# 3. Exploratory analysis and visualizations
# 4. Train a model to identify the most important features
# 5. Log experiments to MLflow
# 6. Present key insights

# %% [markdown]
# ## 1. Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

DATA_DIR = Path("data")

# %% [markdown]
# ## 2. MLflow Setup

# %%
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("vroom-forecast-exploration")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: {mlflow.get_experiment_by_name('vroom-forecast-exploration')}")

# %% [markdown]
# ## 3. Load and Inspect Data

# %% [markdown]
# ### 3.1 Vehicles

# %%
vehicles = pd.read_csv(DATA_DIR / "vehicles.csv")
print(f"Shape: {vehicles.shape}")
vehicles.head()

# %%
vehicles.info()

# %%
vehicles.describe()

# %% [markdown]
# ### 3.2 Reservations

# %%
reservations = pd.read_csv(DATA_DIR / "reservations.csv", parse_dates=["created_at"])
print(f"Shape: {reservations.shape}")
reservations.head()

# %%
reservations.info()

# %%
reservations["created_at"].describe()

# %% [markdown]
# ## 4. Data Preparation
#
# Aggregate reservations per vehicle and merge with vehicle attributes.

# %%
# Count reservations per vehicle
res_counts = reservations.groupby("vehicle_id").size().reset_index(name="num_reservations")
print(f"Vehicles with at least 1 reservation: {len(res_counts)}")
res_counts.describe()

# %%
# Merge with vehicle features (left join to keep all vehicles)
df = vehicles.merge(res_counts, on="vehicle_id", how="left")
df["num_reservations"] = df["num_reservations"].fillna(0).astype(int)
print(f"Final dataset shape: {df.shape}")
print(f"Vehicles with 0 reservations: {(df['num_reservations'] == 0).sum()}")
df.head()

# %% [markdown]
# ## 5. Exploratory Data Analysis

# %% [markdown]
# ### 5.1 Target Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["num_reservations"], bins=30, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Number of Reservations")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Reservations per Vehicle")

axes[1].hist(
    df.loc[df["num_reservations"] > 0, "num_reservations"],
    bins=30,
    edgecolor="black",
    alpha=0.7,
)
axes[1].set_xlabel("Number of Reservations")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution (excluding zero)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.2 Feature Distributions

# %%
feature_cols = [
    "technology",
    "actual_price",
    "recommended_price",
    "num_images",
    "street_parked",
    "description",
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), feature_cols):
    if df[col].nunique() <= 5:
        df[col].value_counts().sort_index().plot.bar(ax=ax, edgecolor="black", alpha=0.7)
    else:
        ax.hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
    ax.set_title(col)
    ax.set_xlabel("")

plt.suptitle("Feature Distributions", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.3 Correlation Matrix

# %%
corr = df[feature_cols + ["num_reservations"]].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.4 Reservations by Categorical Features

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, col in zip(axes, ["technology", "street_parked"]):
    sns.boxplot(data=df, x=col, y="num_reservations", ax=ax)
    ax.set_title(f"Reservations by {col}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.5 Reservations vs. Continuous Features

# %%
continuous_cols = ["actual_price", "recommended_price", "num_images", "description"]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, col in zip(axes, continuous_cols):
    ax.scatter(df[col], df["num_reservations"], alpha=0.3, s=10)
    ax.set_xlabel(col)
    ax.set_ylabel("num_reservations")
    ax.set_title(f"Reservations vs {col}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.6 Price Analysis
#
# How does the difference between actual and recommended price affect reservations?

# %%
df["price_diff"] = df["actual_price"] - df["recommended_price"]
df["price_ratio"] = df["actual_price"] / df["recommended_price"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(df["price_diff"], df["num_reservations"], alpha=0.3, s=10)
axes[0].set_xlabel("Price Difference (actual - recommended)")
axes[0].set_ylabel("Reservations")
axes[0].set_title("Reservations vs Price Difference")
axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5)

axes[1].scatter(df["price_ratio"], df["num_reservations"], alpha=0.3, s=10)
axes[1].set_xlabel("Price Ratio (actual / recommended)")
axes[1].set_ylabel("Reservations")
axes[1].set_title("Reservations vs Price Ratio")
axes[1].axvline(x=1, color="red", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.7 Reservations Over Time

# %%
reservations_ts = reservations.set_index("created_at").resample("W").size()

fig, ax = plt.subplots(figsize=(14, 4))
reservations_ts.plot(ax=ax)
ax.set_ylabel("Reservations per Week")
ax.set_title("Reservation Volume Over Time")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Feature Importance — Model Training with MLflow
#
# Train a Random Forest and a Gradient Boosting model to predict `num_reservations`,
# extract feature importances, and log everything to MLflow.

# %%
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score

# Prepare features
feature_cols_model = [
    "technology",
    "actual_price",
    "recommended_price",
    "num_images",
    "street_parked",
    "description",
    "price_diff",
    "price_ratio",
]

X = df[feature_cols_model].copy()
y = df["num_reservations"].copy()

print(f"Features shape: {X.shape}")
print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")

# %% [markdown]
# ### 6.1 Random Forest

# %%
with mlflow.start_run(run_name="random_forest"):
    rf_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1,
    }
    mlflow.log_params(rf_params)
    mlflow.set_tag("model_type", "RandomForestRegressor")

    rf = RandomForestRegressor(**rf_params)

    # Cross-validation
    rf_scores = cross_val_score(rf, X, y, cv=5, scoring="neg_mean_absolute_error")
    rf_cv_mae = -rf_scores.mean()
    rf_cv_std = rf_scores.std()
    mlflow.log_metric("cv_mae_mean", rf_cv_mae)
    mlflow.log_metric("cv_mae_std", rf_cv_std)
    print(f"Random Forest CV MAE: {rf_cv_mae:.3f} (+/- {rf_cv_std:.3f})")

    # Fit on full data
    rf.fit(X, y)
    rf_preds = rf.predict(X)
    mlflow.log_metric("train_mae", mean_absolute_error(y, rf_preds))
    mlflow.log_metric("train_rmse", root_mean_squared_error(y, rf_preds))
    mlflow.log_metric("train_r2", r2_score(y, rf_preds))

    # Log feature importances
    rf_importances = pd.Series(rf.feature_importances_, index=feature_cols_model).sort_values(
        ascending=True
    )
    for feat, imp in rf_importances.items():
        mlflow.log_metric(f"importance_{feat}", imp)

    # Log feature importance plot
    fig, ax = plt.subplots(figsize=(8, 5))
    rf_importances.plot.barh(ax=ax, edgecolor="black", alpha=0.7)
    ax.set_title("Random Forest — Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    mlflow.log_figure(fig, "rf_feature_importances.png")
    plt.show()

    # Log model
    mlflow.sklearn.log_model(rf, artifact_path="model", input_example=X.head(1))
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# %% [markdown]
# ### 6.2 Gradient Boosting

# %%
with mlflow.start_run(run_name="gradient_boosting"):
    gb_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    mlflow.log_params(gb_params)
    mlflow.set_tag("model_type", "GradientBoostingRegressor")

    gb = GradientBoostingRegressor(**gb_params)

    # Cross-validation
    gb_scores = cross_val_score(gb, X, y, cv=5, scoring="neg_mean_absolute_error")
    gb_cv_mae = -gb_scores.mean()
    gb_cv_std = gb_scores.std()
    mlflow.log_metric("cv_mae_mean", gb_cv_mae)
    mlflow.log_metric("cv_mae_std", gb_cv_std)
    print(f"Gradient Boosting CV MAE: {gb_cv_mae:.3f} (+/- {gb_cv_std:.3f})")

    # Fit on full data
    gb.fit(X, y)
    gb_preds = gb.predict(X)
    mlflow.log_metric("train_mae", mean_absolute_error(y, gb_preds))
    mlflow.log_metric("train_rmse", root_mean_squared_error(y, gb_preds))
    mlflow.log_metric("train_r2", r2_score(y, gb_preds))

    # Log feature importances
    gb_importances = pd.Series(gb.feature_importances_, index=feature_cols_model).sort_values(
        ascending=True
    )
    for feat, imp in gb_importances.items():
        mlflow.log_metric(f"importance_{feat}", imp)

    # Log feature importance plot
    fig, ax = plt.subplots(figsize=(8, 5))
    gb_importances.plot.barh(ax=ax, edgecolor="black", alpha=0.7)
    ax.set_title("Gradient Boosting — Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    mlflow.log_figure(fig, "gb_feature_importances.png")
    plt.show()

    # Log model
    mlflow.sklearn.log_model(gb, artifact_path="model", input_example=X.head(1))
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# %% [markdown]
# ### 6.3 Comparison of Feature Importances

# %%
importance_df = pd.DataFrame(
    {
        "Random Forest": rf_importances,
        "Gradient Boosting": gb_importances,
    }
)

fig, ax = plt.subplots(figsize=(10, 5))
importance_df.plot.barh(ax=ax, edgecolor="black", alpha=0.7)
ax.set_title("Feature Importances — Model Comparison")
ax.set_xlabel("Importance")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.4 Permutation Importance

# %%
from sklearn.inspection import permutation_importance

perm_result = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
perm_importances = pd.Series(perm_result.importances_mean, index=feature_cols_model).sort_values(
    ascending=True
)

fig, ax = plt.subplots(figsize=(8, 5))
perm_importances.plot.barh(ax=ax, edgecolor="black", alpha=0.7, xerr=perm_result.importances_std)
ax.set_title("Permutation Importance (Random Forest)")
ax.set_xlabel("Mean decrease in score")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Key Insights
#
# **Summary of findings** (to be filled after running the notebook):
#
# 1. **Top drivers of reservations:** _[fill after running]_
# 2. **Technology effect:** Does having technology installed increase bookings?
# 3. **Pricing:** How does the gap between actual and recommended price impact demand?
# 4. **Images:** Do more photos lead to more reservations?
# 5. **Description length:** Does a longer description help?
# 6. **Street parking:** Any significant effect?
#
# **Best model:** _[fill after comparing RF vs GB MAE scores]_
#
# View all runs at: http://localhost:5001

# %%
# Summary statistics for top insights
print("=== Mean reservations by technology ===")
print(df.groupby("technology")["num_reservations"].mean())
print()
print("=== Mean reservations by street_parked ===")
print(df.groupby("street_parked")["num_reservations"].mean())
print()
print("=== Correlation with num_reservations ===")
print(
    df[feature_cols_model + ["num_reservations"]]
    .corr()["num_reservations"]
    .sort_values(ascending=False)
)
print()
print("=== Model Comparison ===")
print(f"Random Forest  CV MAE: {rf_cv_mae:.3f} (+/- {rf_cv_std:.3f})")
print(f"Gradient Boost CV MAE: {gb_cv_mae:.3f} (+/- {gb_cv_std:.3f})")
