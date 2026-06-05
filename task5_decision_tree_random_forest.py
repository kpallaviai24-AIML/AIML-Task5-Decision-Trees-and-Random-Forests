# ==========================================================
# TASK 5 : DECISION TREES AND RANDOM FORESTS
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# CREATE PROJECT FOLDERS
# ==========================================================

os.makedirs("Output_Visualizations", exist_ok=True)
os.makedirs("Model_Output", exist_ok=True)
os.makedirs("Predictions", exist_ok=True)

print("=" * 60)
print("TASK 5 : DECISION TREES AND RANDOM FORESTS")
print("=" * 60)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("dataset/heart.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="target",
    data=df
)

plt.title("Target Distribution")

plt.savefig(
    "Output_Visualizations/target_distribution.png"
)

plt.close()

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "Output_Visualizations/correlation_heatmap.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop(
    "target",
    axis=1
)

y = df["target"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# ==========================================================
# DECISION TREE MODEL
# ==========================================================

dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

print("\nDecision Tree Training Completed")

# ==========================================================
# DECISION TREE PREDICTIONS
# ==========================================================

dt_pred = dt_model.predict(
    X_test
)

# ==========================================================
# DECISION TREE METRICS
# ==========================================================

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

dt_precision = precision_score(
    y_test,
    dt_pred
)

dt_recall = recall_score(
    y_test,
    dt_pred
)

dt_f1 = f1_score(
    y_test,
    dt_pred
)

print("\nDecision Tree Accuracy :", dt_accuracy)
print("Decision Tree Precision:", dt_precision)
print("Decision Tree Recall   :", dt_recall)
print("Decision Tree F1 Score :", dt_f1)

# ==========================================================
# DECISION TREE VISUALIZATION
# ==========================================================

plt.figure(figsize=(20,10))

plot_tree(
    dt_model,
    filled=True,
    feature_names=X.columns,
    class_names=["No Disease","Disease"]
)

plt.savefig(
    "Output_Visualizations/decision_tree.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# DECISION TREE CONFUSION MATRIX
# ==========================================================

dt_cm = confusion_matrix(
    y_test,
    dt_pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    dt_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Decision Tree Confusion Matrix")

plt.savefig(
    "Output_Visualizations/confusion_matrix_dt.png"
)

plt.close()

# ==========================================================
# DECISION TREE CLASSIFICATION REPORT
# ==========================================================

dt_report = classification_report(
    y_test,
    dt_pred
)

with open(
    "Model_Output/classification_report_dt.txt",
    "w"
) as f:

    f.write(dt_report)

with open(
    "Model_Output/decision_tree_metrics.txt",
    "w"
) as f:

    f.write(
        f"Accuracy : {dt_accuracy}\n"
    )

    f.write(
        f"Precision: {dt_precision}\n"
    )

    f.write(
        f"Recall   : {dt_recall}\n"
    )

    f.write(
        f"F1 Score : {dt_f1}\n"
    )

# ==========================================================
# TREE DEPTH ANALYSIS
# ==========================================================

depths = range(1,21)

train_scores = []
test_scores = []

for depth in depths:

    temp_model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    temp_model.fit(
        X_train,
        y_train
    )

    train_scores.append(
        temp_model.score(
            X_train,
            y_train
        )
    )

    test_scores.append(
        temp_model.score(
            X_test,
            y_test
        )
    )

plt.figure(figsize=(8,5))

plt.plot(
    depths,
    train_scores,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    depths,
    test_scores,
    marker="o",
    label="Test Accuracy"
)

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Tree Depth Analysis")
plt.legend()

plt.savefig(
    "Output_Visualizations/tree_depth_analysis.png"
)

plt.close()

# ==========================================================
# RANDOM FOREST MODEL
# ==========================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

print(
    "\nRandom Forest Training Completed"
)

# ==========================================================
# RANDOM FOREST PREDICTIONS
# ==========================================================

rf_pred = rf_model.predict(
    X_test
)

# ==========================================================
# RANDOM FOREST METRICS
# ==========================================================

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred
)

rf_recall = recall_score(
    y_test,
    rf_pred
)

rf_f1 = f1_score(
    y_test,
    rf_pred
)

print(
    "\nRandom Forest Accuracy :",
    rf_accuracy
)

print(
    "Random Forest Precision:",
    rf_precision
)

print(
    "Random Forest Recall   :",
    rf_recall
)

print(
    "Random Forest F1 Score :",
    rf_f1
)

# ==========================================================
# RANDOM FOREST CONFUSION MATRIX
# ==========================================================

rf_cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.savefig(
    "Output_Visualizations/confusion_matrix_rf.png"
)

plt.close()

# ==========================================================
# RANDOM FOREST REPORT
# ==========================================================

rf_report = classification_report(
    y_test,
    rf_pred
)

with open(
    "Model_Output/classification_report_rf.txt",
    "w"
) as f:

    f.write(
        rf_report
    )

with open(
    "Model_Output/random_forest_metrics.txt",
    "w"
) as f:

    f.write(
        f"Accuracy : {rf_accuracy}\n"
    )

    f.write(
        f"Precision: {rf_precision}\n"
    )

    f.write(
        f"Recall   : {rf_recall}\n"
    )

    f.write(
        f"F1 Score : {rf_f1}\n"
    )

# ==========================================================
# FEATURE IMPORTANCE - DECISION TREE
# ==========================================================

dt_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": dt_model.feature_importances_
    }
)

dt_importance = dt_importance.sort_values(
    by="Importance",
    ascending=False
)

dt_importance.to_csv(
    "Model_Output/feature_importance_dt.csv",
    index=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=dt_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Decision Tree Feature Importance"
)

plt.savefig(
    "Output_Visualizations/feature_importance_dt.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# FEATURE IMPORTANCE - RANDOM FOREST
# ==========================================================

rf_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    }
)

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

rf_importance.to_csv(
    "Model_Output/feature_importance_rf.csv",
    index=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=rf_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.savefig(
    "Output_Visualizations/feature_importance_rf.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# CROSS VALIDATION
# ==========================================================

dt_cv = cross_val_score(
    dt_model,
    X,
    y,
    cv=5
)

rf_cv = cross_val_score(
    rf_model,
    X,
    y,
    cv=5
)

with open(
    "Model_Output/cross_validation_results.txt",
    "w"
) as f:

    f.write(
        "Decision Tree Cross Validation Scores\n"
    )

    f.write(
        str(dt_cv)
    )

    f.write(
        "\n\nRandom Forest Cross Validation Scores\n"
    )

    f.write(
        str(rf_cv)
    )

    f.write(
        f"\n\nDecision Tree Mean Accuracy : {dt_cv.mean()}"
    )

    f.write(
        f"\nRandom Forest Mean Accuracy : {rf_cv.mean()}"
    )

# ==========================================================
# CROSS VALIDATION GRAPH
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    range(1,6),
    dt_cv,
    marker="o",
    label="Decision Tree"
)

plt.plot(
    range(1,6),
    rf_cv,
    marker="o",
    label="Random Forest"
)

plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross Validation Scores")
plt.legend()

plt.savefig(
    "Output_Visualizations/cross_validation_scores.png"
)

plt.close()

# ==========================================================
# MODEL COMPARISON
# ==========================================================

models = [
    "Decision Tree",
    "Random Forest"
]

accuracies = [
    dt_accuracy,
    rf_accuracy
]

plt.figure(figsize=(8,5))

sns.barplot(
    x=models,
    y=accuracies
)

plt.title(
    "Model Comparison"
)

plt.ylabel(
    "Accuracy"
)

plt.savefig(
    "Output_Visualizations/model_comparison.png"
)

plt.close()

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

dt_predictions = pd.DataFrame(
    {
        "Actual": y_test,
        "Predicted": dt_pred
    }
)

dt_predictions.to_csv(
    "Predictions/decision_tree_predictions.csv",
    index=False
)

rf_predictions = pd.DataFrame(
    {
        "Actual": y_test,
        "Predicted": rf_pred
    }
)

rf_predictions.to_csv(
    "Predictions/random_forest_predictions.csv",
    index=False
)

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"Decision Tree Accuracy : {dt_accuracy:.4f}"
)

print(
    f"Random Forest Accuracy : {rf_accuracy:.4f}"
)

print(
    f"Decision Tree CV Mean : {dt_cv.mean():.4f}"
)

print(
    f"Random Forest CV Mean : {rf_cv.mean():.4f}"
)

print("\nTop 5 Decision Tree Features:")
print(
    dt_importance.head()
)

print("\nTop 5 Random Forest Features:")
print(
    rf_importance.head()
)

print("\nProject Outputs Generated Successfully!")

print(
    "\nOutput_Visualizations Folder Created"
)

print(
    "Model_Output Folder Created"
)

print(
    "Predictions Folder Created"
)

print(
    "\n✅Task 5 Completed Successfully!"
)

print("=" * 60)   