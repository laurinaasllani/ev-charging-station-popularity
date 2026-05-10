import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

try:
    from xgboost import XGBClassifier
except ImportError:
    print("Instalo XGBoost:")
    print("pip install xgboost")
    XGBClassifier = None


# ======================================
# 1. Leximi i datasetit
# ======================================

df = pd.read_csv("EVChargingStationUsage.csv")

df.columns = df.columns.str.strip()

print("\nDataset Information:\n")

print(df.info())


# ======================================
# 2. Funksion për kohëzgjatjen
# ======================================

def duration_to_minutes(value):

    try:
        h, m, s = str(value).split(":")

        return (
            int(h) * 60 +
            int(m) +
            int(s) / 60
        )

    except:
        return 0


# ======================================
# 3. Përpunimi i kolonave
# ======================================

df["User ID"] = df["User ID"].fillna("Unknown")


# Charging Time
if "Charging Time (hh:mm:ss)" in df.columns:

    df["Charging Time Minutes"] = (
        df["Charging Time (hh:mm:ss)"]
        .apply(duration_to_minutes)
    )


# Total Duration
if "Total Duration (hh:mm:ss)" in df.columns:

    df["Total Duration Minutes"] = (
        df["Total Duration (hh:mm:ss)"]
        .apply(duration_to_minutes)
    )


# Start Date
if "Start Date" in df.columns:

    df["Start Date"] = pd.to_datetime(
        df["Start Date"],
        errors="coerce"
    )

    df["Start Hour"] = df["Start Date"].dt.hour

    df["Start Day"] = df["Start Date"].dt.day

    df["Start Month"] = df["Start Date"].dt.month

    df["Start Weekday"] = (
        df["Start Date"].dt.weekday
    )


# ======================================
# 4. Krijimi i target-it Popular
# ======================================

station_popularity = (
    df.groupby("Station Name")["User ID"]
    .nunique()
    .reset_index()
)

station_popularity.columns = [
    "Station Name",
    "Unique Users"
]


threshold = station_popularity[
    "Unique Users"
].quantile(0.75)


station_popularity["Popular"] = np.where(

    station_popularity["Unique Users"] >= threshold,

    1,

    0
)


df = df.merge(

    station_popularity[
        ["Station Name", "Popular"]
    ],

    on="Station Name",

    how="left"
)


print("\nPopularity Threshold:")

print(threshold)

print("\nClass Distribution:")

print(df["Popular"].value_counts())


# ======================================
# 5. Feature Selection
# ======================================
# IMPORTANT:
# Station Name, Latitude dhe Longitude
# HIQEN për të shmangur leakage

selected_features = [

    "Org Name",

    "Port Type",

    "Port Number",

    "Plug Type",

    "City",

    "State/Province",

    "Postal Code",

    "Country",

    "Fee",

    "Ended By",

    "Driver Postal Code",

    "Charging Time Minutes",

    "Total Duration Minutes",

    "Start Hour",

    "Start Day",

    "Start Month",

    "Start Weekday"
]


selected_features = [

    col for col in selected_features

    if col in df.columns
]


data = df[
    selected_features + ["Popular"]
].copy()


# Sigurohemi që target-i është 0/1
data["Popular"] = (
    data["Popular"]
    .fillna(0)
    .astype(int)
)


# ======================================
# 6. Missing Values
# ======================================

for col in data.columns:

    if col == "Popular":
        continue

    if data[col].dtype == "object":

        data[col] = data[col].fillna(
            data[col].mode()[0]
        )

    else:

        data[col] = data[col].fillna(
            data[col].mean()
        )


# ======================================
# 7. X dhe y
# ======================================

X = data.drop(columns=["Popular"])

y = data["Popular"]


# Label Encoding
for col in X.columns:

    if X[col].dtype == "object":

        le = LabelEncoder()

        X[col] = le.fit_transform(
            X[col].astype(str)
        )


# ======================================
# 8. Train/Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n==============================")

print("DATA SPLIT")

print("==============================")

print("80% Training")

print("20% Testing")


# ======================================
# 9. StandardScaler
# ======================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ======================================
# 10. MODELS
# ======================================

models = {

    "Lasso Logistic Regression":

        LogisticRegression(

            penalty="l1",

            solver="liblinear",

            max_iter=1000,

            random_state=42
        ),

    "Random Forest":

        RandomForestClassifier(

            n_estimators=50,

            random_state=42,

            class_weight="balanced"
        ),

    "Gradient Boosting":

        GradientBoostingClassifier(

            n_estimators=50,

            learning_rate=0.1,

            random_state=42
        )
}


# XGBoost
if XGBClassifier is not None:

    models["XGBoost"] = XGBClassifier(

        n_estimators=50,

        learning_rate=0.1,

        max_depth=5,

        random_state=42,

        eval_metric="logloss"
    )


# ======================================
# 11. Training & Evaluation
# ======================================

results = []


for name, model in models.items():

    print(f"\nTraining Model: {name}")


    if name == "Lasso Logistic Regression":

        model.fit(
            X_train_scaled,
            y_train
        )

        y_pred = model.predict(
            X_test_scaled
        )

    else:

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1
    })


    print("\nConfusion Matrix:")

    print(confusion_matrix(
        y_test,
        y_pred
    ))


    print("\nClassification Report:")

    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))


# ======================================
# 12. Final Results
# ======================================

results_df = pd.DataFrame(results)


print("\n==============================")

print("MODEL PERFORMANCE")

print("==============================")

print(results_df)


results_df.to_csv(

    "classification_model_results.csv",

    index=False
)


# ======================================
# 13. Feature Importance
# ======================================

rf_model = RandomForestClassifier(

    n_estimators=50,

    random_state=42,

    class_weight="balanced"
)


rf_model.fit(
    X_train,
    y_train
)


feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        rf_model.feature_importances_
})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\n==============================")

print("TOP FACTORS")

print("==============================")

print(feature_importance.head(15))


feature_importance.to_csv(

    "classification_feature_importance.csv",

    index=False
)


# ======================================
# 14. Better Visualization
# ======================================

top_features = (
    feature_importance
    .head(10)
    .sort_values("Importance")
)


plt.figure(figsize=(11, 7))


plt.barh(

    top_features["Feature"],

    top_features["Importance"]
)


for index, value in enumerate(

    top_features["Importance"]
):

    plt.text(

        value,

        index,

        f"{value:.3f}",

        va="center"
    )


plt.xlabel("Importance Score")

plt.title(
    "Most Important Factors Affecting EV Charging Station Popularity"
)

plt.tight_layout()

plt.savefig(
    "feature_importance_chart.png",
    dpi=300
)

plt.show()


print("\nAnalysis Completed Successfully.")

print("\nGenerated Files:")

print("classification_model_results.csv")

print("classification_feature_importance.csv")

print("feature_importance_chart.png")