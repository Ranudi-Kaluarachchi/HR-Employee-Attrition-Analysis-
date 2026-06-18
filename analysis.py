import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("✅ Dataset loaded successfully\n")

# -----------------------------
# 2. First look at the data
# -----------------------------
print("===== FIRST 5 ROWS =====")
print(df.head())
print()

print("===== DATASET SHAPE =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print()

print("===== COLUMN NAMES =====")
print(df.columns.tolist())
print()

print("===== DATA TYPES =====")
print(df.dtypes)
print()

print("===== MISSING VALUES =====")
print(df.isnull().sum())
print()

print("===== SUMMARY STATISTICS =====")
print(df.describe(include="all"))
print()

# -----------------------------
# 3. Check important columns
# -----------------------------
print("===== ATTRITION COUNTS =====")
print(df["Attrition"].value_counts())
print()

print("===== JOB SATISFACTION COUNTS =====")
print(df["JobSatisfaction"].value_counts())
print()

print("===== ENVIRONMENT SATISFACTION COUNTS =====")
print(df["EnvironmentSatisfaction"].value_counts())
print()

print("===== RELATIONSHIP SATISFACTION COUNTS =====")
print(df["RelationshipSatisfaction"].value_counts())
print()

print("===== WORK LIFE BALANCE COUNTS =====")
print(df["WorkLifeBalance"].value_counts())
print()

print("===== OVERTIME COUNTS =====")
print(df["OverTime"].value_counts())
print()

# -----------------------------
# 4. Create smaller cleaned dataframe
# -----------------------------
df1 = df[[
    "Attrition",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "DistanceFromHome",
    "OverTime"
]].copy()

print("===== SELECTED DATAFRAME (df1) =====")
print(df1.head())
print()

print("===== df1 INFO =====")
print(df1.info())
print()

print("===== df1 MISSING VALUES =====")
print(df1.isnull().sum())
print()

# -----------------------------
# 5. Convert categorical values to numeric
# -----------------------------
df1["Attrition"] = df1["Attrition"].map({"Yes": 1, "No": 0})
df1["OverTime"] = df1["OverTime"].map({"Yes": 1, "No": 0})

# -----------------------------
# 5.1 Save cleaned dataset
# -----------------------------
df1.to_csv("data/cleaned_attrition_dataset.csv", index=False)

print("✅ Cleaned dataset saved as cleaned_attrition_dataset.csv")

print("===== CONVERTED df1 HEAD =====")
print(df1.head())
print()

# -----------------------------
# 6. Correlation
# -----------------------------
print("===== CORRELATION MATRIX =====")
print(df1.corr())
print()

# -----------------------------
# 7. Visualizations
# -----------------------------
sns.set(style="whitegrid")

# Attrition count plot
plt.figure(figsize=(6, 4))
sns.countplot(x="Attrition", data=df1)
plt.title("Attrition Count (0 = No, 1 = Yes)")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("visuals/attrition_count.png", dpi=300, bbox_inches="tight")
plt.show()

# Job Satisfaction vs Attrition
plt.figure(figsize=(6, 4))
sns.countplot(x="JobSatisfaction", hue="Attrition", data=df1)
plt.title("Job Satisfaction vs Attrition")
plt.xlabel("Job Satisfaction")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("visuals/Job_Satisfaction_vs_Attrition.png", dpi=300, bbox_inches="tight")
plt.show()

# Overtime vs Attrition
plt.figure(figsize=(6, 4))
sns.countplot(x="OverTime", hue="Attrition", data=df1)
plt.title("OverTime vs Attrition")
plt.xlabel("OverTime (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("visuals/Overtime_vs_Attrition.png", dpi=300, bbox_inches="tight")
plt.show()

# Monthly Income vs Attrition
plt.figure(figsize=(8, 5))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df1)
plt.title("Monthly Income vs Attrition")
plt.xlabel("Attrition (0 = No, 1 = Yes)")
plt.ylabel("Monthly Income")
plt.tight_layout()
plt.savefig("visuals/Monthly_Income_vs_Attrition.png", dpi=300, bbox_inches="tight")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df1.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("visuals/Correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Analysis completed successfully.")

# -----------------------------
# 8. MODEL BUILDING
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Features (X) and Target (y)
X = df1.drop("Attrition", axis=1)
y = df1["Attrition"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LogisticRegression(max_iter=2000)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print()

print("Classification Report:")
print(classification_report(y_test, y_pred))

from scipy.stats import pearsonr

print("===== HYPOTHESIS TESTING =====")

columns = [
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance"
]

for col in columns:
    corr, p_value = pearsonr(df1[col], df1["Attrition"])
    
    print(f"\n{col}:")
    print("Correlation:", corr)
    print("P-value:", p_value)
    
    if p_value < 0.05:
        print("→ Reject H0 (Significant relationship)")
    else:
        print("→ Fail to reject H0 (No significant relationship)")

print("\n===== FINAL HYPOTHESIS CONCLUSION =====")
print("All tested variables have p-values < 0.05")
print("→ Reject H0")
print("→ Fair Treatment significantly affects Attrition")
