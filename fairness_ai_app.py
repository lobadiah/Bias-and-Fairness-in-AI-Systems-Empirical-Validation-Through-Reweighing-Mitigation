# ============================================================
# FAIRNESS EVALUATION & MITIGATION STREAMLIT APP
# Author: Louis Obadiah (MSc AI Engineering)
# Supervisor: Asst. Prof. Dr. John Olaifa, Okan University
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# ============================================================
# 1️⃣ STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Fairness Evaluation Framework", layout="wide")
st.title("⚖️ Fairness Evaluation and Mitigation Framework")
st.caption("Louis Obadiah (MSc AI Engineering) — Supervised by Asst. Prof. Dr. John Olaifa, Okan University")

# ============================================================
# 2️⃣ LOAD DATASET
# ============================================================
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")

    with st.expander("🔍 View Raw Data (First 5 rows)"):
        st.dataframe(df.head())

    # ============================================================
    # DATA CLEANING
    # ============================================================
    st.markdown("### 🧹 Step 1: Data Cleaning")

    # Replace placeholders for missing values
    df = df.replace('?', np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill missing values
    missing_cols = []
    for col in df.columns:
        if df[col].isna().any():
            missing_cols.append(col)
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    if missing_cols:
        st.info(f"📊 Filled missing values in columns: {', '.join(missing_cols)}")

    # Standardize column names
    df.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in df.columns]

    # Selection of target and protected attribute
    col_a, col_b = st.columns(2)
    with col_a:
        target = st.selectbox("🎯 Select target column:", df.columns, index=list(df.columns).index('income') if 'income' in df.columns else 0)
    with col_b:
        protected_attr = st.selectbox("🛡️ Select protected attribute:", df.columns, index=list(df.columns).index('sex') if 'sex' in df.columns else 0)

    # ============================================================
    # TARGET CONVERSION
    # ============================================================
    # Convert target to binary (1 = positive class, 0 = negative class)
    # Detect if it's already binary or needs conversion
    unique_targets = df[target].unique()
    if len(unique_targets) == 2 and set(unique_targets) == {0, 1}:
        st.info(f"Target '{target}' is already binary.")
    else:
        # For Adult dataset specifically, but generic enough for strings
        pos_val = st.text_input("Value for positive class (1):", value='>50K' if 'income' in target else str(unique_targets[0]))
        df[target] = df[target].apply(lambda x: 1 if pos_val in str(x) else 0)

    # Display target distribution
    target_dist = df[target].value_counts()
    st.write(f"**Target Distribution:** {target_dist.get(0, 0)} (0), {target_dist.get(1, 0)} (1)")

    # ============================================================
    # ENCODING
    # ============================================================
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        st.info(f"📝 Encoded {len(categorical_cols)} categorical columns")

    df = df.fillna(0)

    # ============================================================
    # DEFINE PRIVILEGED GROUPS
    # ============================================================
    unique_vals = sorted(df[protected_attr].unique())
    st.write(f"**Protected attribute '{protected_attr}' encoded values:** {unique_vals}")

    privileged_val = st.selectbox(f"Select value for PRIVILEGED group in '{protected_attr}':", unique_vals, index=min(1, len(unique_vals)-1))

    privileged_groups = [{protected_attr: privileged_val}]
    unprivileged_groups = [{protected_attr: v} for v in unique_vals if v != privileged_val]

    # ============================================================
    # 3️⃣ TRAIN / TEST SPLIT & SCALING
    # ============================================================
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Keep protected attribute unscaled for AIF360
    train_protected = X_train[protected_attr].reset_index(drop=True)
    test_protected = X_test[protected_attr].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    X_train_scaled_df[protected_attr] = train_protected
    X_test_scaled_df[protected_attr] = test_protected

    # ============================================================
    # 4️⃣ BASELINE MODEL
    # ============================================================
    st.markdown("### ⚙️ Step 2: Baseline Model Training")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled_df, y_train)
    y_pred = model.predict(X_test_scaled_df)

    acc_before = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy (Before Mitigation):** `{acc_before:.3f}`")

    # ============================================================
    # 5️⃣ FAIRNESS EVALUATION (Before)
    # ============================================================
    st.markdown("### 📊 Step 3: Fairness Evaluation")

    def get_metrics(X_df, y_true, y_pred_vals):
        bld_df = pd.concat([X_df.reset_index(drop=True), y_true.reset_index(drop=True)], axis=1)
        bld = BinaryLabelDataset(df=bld_df, label_names=[target], protected_attribute_names=[protected_attr])

        pred_bld = bld.copy()
        pred_bld.labels = y_pred_vals.reshape(-1, 1)

        metric = ClassificationMetric(bld, pred_bld, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        return metric.statistical_parity_difference(), metric.equal_opportunity_difference()

    try:
        stat_par_before, eq_opp_before = get_metrics(X_test_scaled_df, y_test, y_pred)

        st.write("#### 📈 Fairness Metrics (Before Mitigation)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Statistical Parity Difference", value=f"{stat_par_before:.3f}")
        with col2:
            st.metric(label="Equal Opportunity Difference", value=f"{eq_opp_before:.3f}")

    except Exception as e:
        st.error(f"❌ Failed to compute fairness metrics: {str(e)}")
        stat_par_before, eq_opp_before = 0.0, 0.0

    # ============================================================
    # 6️⃣ APPLY REWEIGHING
    # ============================================================
    st.markdown("### 🧩 Step 4: Apply Bias Mitigation (Reweighing)")

    try:
        train_bld_df = pd.concat([X_train_scaled_df.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        train_bld = BinaryLabelDataset(df=train_bld_df, label_names=[target], protected_attribute_names=[protected_attr])

        RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        RW.fit(train_bld)
        train_transf = RW.transform(train_bld)

        model_rw = LogisticRegression(max_iter=1000)
        model_rw.fit(train_transf.features, train_transf.labels.ravel(), sample_weight=train_transf.instance_weights)

        y_pred_rw = model_rw.predict(X_test_scaled_df)
        acc_after = accuracy_score(y_test, y_pred_rw)

        stat_par_after, eq_opp_after = get_metrics(X_test_scaled_df, y_test, y_pred_rw)

        st.write(f"**Model Accuracy (After Mitigation):** `{acc_after:.3f}`")
        st.write("#### 📈 Fairness Metrics (After Mitigation)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Statistical Parity Difference", value=f"{stat_par_after:.3f}", delta=f"{stat_par_after - stat_par_before:+.3f}")
        with col2:
            st.metric(label="Equal Opportunity Difference", value=f"{eq_opp_after:.3f}", delta=f"{eq_opp_after - eq_opp_before:+.3f}")

    except Exception as e:
        st.warning(f"⚠️ Reweighing failed: {str(e)}")
        acc_after, stat_par_after, eq_opp_after = acc_before, stat_par_before, eq_opp_before

    # ============================================================
    # 7️⃣ VISUALIZATION
    # ============================================================
    st.markdown("### 📊 Step 5: Results Visualization")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ['Stat. Parity', 'Eq. Opportunity']
    x = np.arange(len(labels))
    ax1.bar(x - 0.15, [stat_par_before, eq_opp_before], width=0.3, label='Before', color='skyblue')
    ax1.bar(x + 0.15, [stat_par_after, eq_opp_after], width=0.3, label='After', color='orange')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('Fairness Metrics Comparison')
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.legend()

    ax2.bar(['Before', 'After'], [acc_before, acc_after], color=['skyblue', 'orange'])
    ax2.set_title('Accuracy Comparison')
    ax2.set_ylim([0, 1])
    for i, v in enumerate([acc_before, acc_after]):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha='center')

    st.pyplot(fig)

    # ============================================================
    # 8️⃣ EXPORT
    # ============================================================
    if st.button("💾 Export Results to CSV"):
        res_df = pd.DataFrame({
            "Metric": ["Accuracy", "Stat. Parity Diff", "Eq. Opportunity Diff"],
            "Before": [acc_before, stat_par_before, eq_opp_before],
            "After": [acc_after, stat_par_after, eq_opp_after]
        })
        res_df.to_csv("fairness_results_app.csv", index=False)
        st.success("Results exported to 'fairness_results_app.csv'")

else:
    st.info("👆 Please upload your dataset to begin.")
