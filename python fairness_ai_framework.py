# ======================================
# FAIRNESS EVALUATION AND MITIGATION TOOL
# Using your uploaded dataset: adult.csv
# ======================================

# ---- Install dependencies (run once) ----
# pip install aif360 fairlearn scikit-learn pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Fairness tools
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# -----------------------------------------
# 1. LOAD AND CLEAN DATA
# -----------------------------------------
df = pd.read_csv(
    r"C:\Users\Mr. Louis Obadiah\Desktop\FALL SEMISTER\AIE501 SEMINAR\the_project\adult.csv"
)

print("✅ Dataset loaded successfully!")
print(df.head())

# Replace placeholders for missing values and clean
df = df.replace('?', np.nan)
df = df.replace([np.inf, -np.inf], np.nan)

# Drop or fill missing values
for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

print(f"\n✅ After cleaning, dataset shape: {df.shape}")
print(f"🔍 Remaining missing values: {df.isna().sum().sum()}")

# Standardize column names
df.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in df.columns]

# Define target and protected attribute
target = 'income'
protected_attr = 'sex'

# Convert target to binary
df[target] = df[target].apply(lambda x: 1 if '>50K' in str(x) else 0)

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Final NaN check
df = df.fillna(0)

# Separate features and label
X = df.drop(columns=[target])
y = df[target]

# -----------------------------------------
# 2. TRAIN BASELINE MODEL
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

acc_before = accuracy_score(y_test, y_pred)
print(f"\n⚙️ Model Accuracy (Before Mitigation): {acc_before:.3f}")

# -----------------------------------------
# 3. CONVERT TO AIF360 FORMAT (SAFE VERSION)
# -----------------------------------------
privileged_groups = [{protected_attr: 1}]   # male
unprivileged_groups = [{protected_attr: 0}] # female

# Combine X and y for AIF360
train_df = pd.concat(
    [pd.DataFrame(X_train, columns=X.columns).reset_index(drop=True),
     y_train.reset_index(drop=True)], axis=1
)
test_df = pd.concat(
    [pd.DataFrame(X_test, columns=X.columns).reset_index(drop=True),
     y_test.reset_index(drop=True)], axis=1
)

# Final cleaning before AIF360 conversion
for df_part, name in [(train_df, "Train"), (test_df, "Test")]:
    df_part.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df_part.columns:
        if df_part[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_part[col]):
                df_part[col].fillna(df_part[col].median(), inplace=True)
            else:
                df_part[col].fillna(df_part[col].mode()[0], inplace=True)
    print(f"🧾 {name} data NaNs after final cleaning: {df_part.isna().sum().sum()}")

# Build AIF360 datasets
train_bld = BinaryLabelDataset(
    df=train_df,
    label_names=[target],
    protected_attribute_names=[protected_attr]
)
test_bld = BinaryLabelDataset(
    df=test_df,
    label_names=[target],
    protected_attribute_names=[protected_attr]
)

# Create prediction dataset for fairness evaluation
test_pred_bld = test_bld.copy()
test_pred_bld.labels = y_pred.reshape(-1, 1)

metric_test = ClassificationMetric(
    test_bld, test_pred_bld,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print(f"\n📊 Fairness Metrics (Before Mitigation):")
stat_par_before = metric_test.statistical_parity_difference()
eq_opp_before = metric_test.equal_opportunity_difference()
print(f"  Statistical Parity Difference: {stat_par_before:.3f}")
print(f"  Equal Opportunity Difference: {eq_opp_before:.3f}")

# -----------------------------------------
# 4. APPLY REWEIGHING (BIAS MITIGATION)
# -----------------------------------------
RW = Reweighing(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
RW.fit(train_bld)
train_transf = RW.transform(train_bld)

# Retrain model using reweighed samples
model_rw = LogisticRegression(max_iter=500)
model_rw.fit(train_transf.features, train_transf.labels.ravel(),
             sample_weight=train_transf.instance_weights)
y_pred_rw = model_rw.predict(X_test_scaled)

acc_after = accuracy_score(y_test, y_pred_rw)
print(f"\n⚙️ Model Accuracy (After Mitigation): {acc_after:.3f}")

# Evaluate fairness again
test_pred_rw = test_bld.copy()
test_pred_rw.labels = y_pred_rw.reshape(-1, 1)
metric_test_rw = ClassificationMetric(
    test_bld, test_pred_rw,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
stat_par_after = metric_test_rw.statistical_parity_difference()
eq_opp_after = metric_test_rw.equal_opportunity_difference()

print(f"\n📊 Fairness Metrics (After Mitigation):")
print(f"  Statistical Parity Difference: {stat_par_after:.3f}")
print(f"  Equal Opportunity Difference: {eq_opp_after:.3f}")

# -----------------------------------------
# 5. VISUALIZE & EXPORT RESULTS
# -----------------------------------------
labels = ['Statistical Parity Diff', 'Equal Opportunity Diff']
before = [stat_par_before, eq_opp_before]
after = [stat_par_after, eq_opp_after]

plt.figure(figsize=(6, 4))
x = np.arange(len(labels))
plt.bar(x - 0.15, before, width=0.3, label='Before')
plt.bar(x + 0.15, after, width=0.3, label='After')
plt.xticks(x, labels)
plt.ylabel('Metric Value (Closer to 0 = Fairer)')
plt.title('Fairness Improvement after Reweighing')
plt.legend()
plt.tight_layout()
plt.show()

# Export key results to CSV
results = pd.DataFrame({
    "Metric": ["Accuracy", "Statistical Parity Diff", "Equal Opportunity Diff"],
    "Before Mitigation": [acc_before, stat_par_before, eq_opp_before],
    "After Mitigation": [acc_after, stat_par_after, eq_opp_after]
})
results.to_csv("fairness_results.csv", index=False)
print("\n💾 Results saved to 'fairness_results.csv'")
