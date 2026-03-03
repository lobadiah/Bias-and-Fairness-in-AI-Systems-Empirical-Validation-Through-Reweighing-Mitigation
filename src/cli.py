# Fairness Evaluation and Mitigation Tool

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


def load_and_clean_data(filepath):
    """Loads and performs initial cleaning of the dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully from {filepath}")

    # Replace placeholders for missing values and clean
    df = df.replace('?', np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop or fill missing values
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    print(f"After cleaning, dataset shape: {df.shape}")

    # Standardize column names
    df.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in df.columns]
    return df

def preprocess_data(df, target='income', protected_attr='sex'):
    """Encodes categorical variables and splits data into train/test sets."""
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Keep protected attribute before scaling for AIF360
    train_protected = X_train[protected_attr].reset_index(drop=True)
    test_protected = X_test[protected_attr].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create DataFrames for scaled features
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Replace scaled protected attribute with unscaled values for AIF360
    X_train_scaled_df[protected_attr] = train_protected
    X_test_scaled_df[protected_attr] = test_protected

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, protected_attr, target

def evaluate_fairness(X_test_df, y_test, y_pred, target, protected_attr, privileged_groups, unprivileged_groups):
    """Computes fairness metrics using AIF360."""
    test_df = pd.concat([X_test_df.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    test_bld = BinaryLabelDataset(
        df=test_df,
        label_names=[target],
        protected_attribute_names=[protected_attr]
    )

    test_pred_bld = test_bld.copy()
    test_pred_bld.labels = y_pred.reshape(-1, 1)

    metric = ClassificationMetric(
        test_bld, test_pred_bld,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    return metric.statistical_parity_difference(), metric.equal_opportunity_difference()


def main():
    protected_attr = 'sex'
    target = 'income'

    privileged_groups = [{protected_attr: 1.0}]
    unprivileged_groups = [{protected_attr: 0.0}]

    # 1. Load and clean data
    df = load_and_clean_data("data/adult_v2.csv")

    # 2. Preprocess
    X_train_scaled, X_test_scaled, y_train, y_test, protected_attr, target = preprocess_data(df, target, protected_attr)

    # 3. Train baseline model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc_before = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy (before mitigation): {acc_before:.3f}")

    stat_par_before, eq_opp_before = evaluate_fairness(X_test_scaled, y_test, y_pred, target, protected_attr, privileged_groups, unprivileged_groups)
    print("Fairness metrics (before mitigation):")
    print(f"  Statistical Parity Difference: {stat_par_before:.3f}")
    print(f"  Equal Opportunity Difference: {eq_opp_before:.3f}")

    # 4. Apply reweighing (bias mitigation)
    train_df = pd.concat([X_train_scaled.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    train_bld = BinaryLabelDataset(
        df=train_df,
        label_names=[target],
        protected_attribute_names=[protected_attr]
    )

    RW = Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    RW.fit(train_bld)
    train_transf = RW.transform(train_bld)

    # Retrain model using reweighed samples and scaled features
    model_rw = LogisticRegression(max_iter=1000)
    model_rw.fit(train_transf.features, train_transf.labels.ravel(),
                 sample_weight=train_transf.instance_weights)
    y_pred_rw = model_rw.predict(X_test_scaled)

    acc_after = accuracy_score(y_test, y_pred_rw)
    print(f"\nModel accuracy (after mitigation): {acc_after:.3f}")

    stat_par_after, eq_opp_after = evaluate_fairness(X_test_scaled, y_test, y_pred_rw, target, protected_attr, privileged_groups, unprivileged_groups)
    print("Fairness metrics (after mitigation):")
    print(f"  Statistical Parity Difference: {stat_par_after:.3f}")
    print(f"  Equal Opportunity Difference: {eq_opp_after:.3f}")

    # 5. Visualize and export results
    labels = ['Statistical Parity Diff', 'Equal Opportunity Diff']
    before = [stat_par_before, eq_opp_before]
    after = [stat_par_after, eq_opp_after]

    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    before_bars = plt.bar(x - 0.15, before, width=0.3, label='Before', color='skyblue')
    after_bars = plt.bar(x + 0.15, after, width=0.3, label='After', color='orange')
    plt.xticks(x, labels)
    plt.ylabel('Metric Value (Closer to 0 = Fairer)')
    plt.title('Fairness Improvement after Reweighing')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()

    for bar in list(before_bars) + list(after_bars):
        height = bar.get_height()
        y_offset = 0.005 if height >= 0 else -0.005
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            f"{height:.3f}",
            ha='center',
            va='bottom' if height >= 0 else 'top',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig('results/bias_visualization.png')
    print("\nVisualization saved to 'results/bias_visualization.png'")

    # Export key results to CSV
    results = pd.DataFrame({
        "Metric": ["Accuracy", "Statistical Parity Diff", "Equal Opportunity Diff"],
        "Before Mitigation": [acc_before, stat_par_before, eq_opp_before],
        "After Mitigation": [acc_after, stat_par_after, eq_opp_after]
    })
    results.to_csv("results/fairness_results.csv", index=False)
    print("Results saved to 'results/fairness_results.csv'")

if __name__ == "__main__":
    main()
