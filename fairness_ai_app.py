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
    # DATA CLEANING (ALIGNED WITH TRAINING CODE)
    # ============================================================
    st.markdown("### 🧹 Step 1: Data Cleaning")
    
    # Replace placeholders for missing values
    df = df.replace('?', np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values (ALIGNED: median for numeric, mode for categorical)
    missing_cols = []
    for col in df.columns:
        if df[col].isna().any():
            missing_cols.append(col)
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    if missing_cols:
        st.info(f"📊 Filled missing values in columns: {', '.join(missing_cols)}")
    
    # Standardize column names
    df.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in df.columns]
    
    # Define target and protected attribute
    target = 'income'
    protected_attr = 'sex'
    
    # Check if columns exist
    if target not in df.columns:
        st.error(f"❌ Target column '{target}' not found!")
        target = st.selectbox("Select target column:", df.columns)
    
    if protected_attr not in df.columns:
        st.error(f"❌ Protected attribute '{protected_attr}' not found!")
        protected_attr = st.selectbox("Select protected attribute:", df.columns)
    
    # ============================================================
    # TARGET CONVERSION (ALIGNED WITH TRAINING CODE)
    # ============================================================
    # Convert target to binary (1 = >50K, 0 = <=50K)
    # ALIGNED: Uses '>50K' in str(x) exactly like training code
    df[target] = df[target].apply(lambda x: 1 if '>50K' in str(x) else 0)
    
    # Display target distribution
    target_dist = df[target].value_counts()
    st.write(f"**Target Distribution:** {target_dist[0]} ≤50K, {target_dist[1]} >50K")
    
    # ============================================================
    # ENCODING (ALIGNED WITH TRAINING CODE)
    # ============================================================
    # Encode ALL categorical variables using LabelEncoder
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    encoders = {}
    
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        
        st.info(f"📝 Encoded {len(categorical_cols)} categorical columns")
    
    # Final NaN check and fill
    df = df.fillna(0)
    
    # ============================================================
    # DEFINE PRIVILEGED GROUPS (CRITICAL ALIGNMENT)
    # ============================================================
    # ALIGNED: Hardcoded values matching training code
    privileged_groups = [{protected_attr: 1}]   # male = 1
    unprivileged_groups = [{protected_attr: 0}] # female = 0
    
    # Verify encoding
    unique_sex_vals = df[protected_attr].unique()
    st.write(f"**Protected attribute '{protected_attr}' unique values:** {sorted(unique_sex_vals)}")
    
    if 0 not in unique_sex_vals or 1 not in unique_sex_vals:
        st.warning("⚠️ Warning: Expected values 0 (female) and 1 (male) not found in encoded data!")
    
    # ============================================================
    # 3️⃣ TRAIN / TEST SPLIT (ALIGNED)
    # ============================================================
    # Separate features and label
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================================
    # 4️⃣ BASELINE MODEL (Before Mitigation)
    # ============================================================
    st.markdown("### ⚙️ Step 2: Baseline Model Training")
    
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc_before = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy (Before Mitigation):** `{acc_before:.3f}`")
    
    # ============================================================
    # 5️⃣ CONVERT TO AIF360 FORMAT (ALIGNED WITH TRAINING CODE)
    # ============================================================
    st.markdown("### 📊 Step 3: Fairness Evaluation")
    
    # ALIGNED: Combine X and y like training code
    train_df = pd.concat(
        [pd.DataFrame(X_train, columns=X.columns).reset_index(drop=True),
         y_train.reset_index(drop=True)], axis=1
    )
    test_df = pd.concat(
        [pd.DataFrame(X_test, columns=X.columns).reset_index(drop=True),
         y_test.reset_index(drop=True)], axis=1
    )
    
    # ALIGNED: Final cleaning before AIF360 conversion
    for df_part, name in [(train_df, "Train"), (test_df, "Test")]:
        df_part.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df_part.columns:
            if df_part[col].isna().any():
                if pd.api.types.is_numeric_dtype(df_part[col]):
                    df_part[col].fillna(df_part[col].median(), inplace=True)
                else:
                    df_part[col].fillna(df_part[col].mode()[0], inplace=True)
    
    # Build AIF360 datasets
    try:
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
        
        # Create prediction dataset
        test_pred_bld = test_bld.copy()
        test_pred_bld.labels = y_pred.reshape(-1, 1)
        
        # Compute fairness metrics
        metric_test = ClassificationMetric(
            test_bld, test_pred_bld,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        
        # ALIGNED: Use same metric names as training code
        stat_par_before = metric_test.statistical_parity_difference()
        eq_opp_before = metric_test.equal_opportunity_difference()
        
        # Handle NaN values safely
        def safe_float(value):
            return float(value) if not np.isnan(value) else 0.0
        
        stat_par_before = safe_float(stat_par_before)
        eq_opp_before = safe_float(eq_opp_before)
        
        st.write("#### 📈 Fairness Metrics (Before Mitigation)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Statistical Parity Difference",
                value=f"{stat_par_before:.3f}",
                delta="Closer to 0 = Fairer" if abs(stat_par_before) < 0.1 else "Bias Detected"
            )
        with col2:
            st.metric(
                label="Equal Opportunity Difference",
                value=f"{eq_opp_before:.3f}",
                delta="Closer to 0 = Fairer" if abs(eq_opp_before) < 0.1 else "Bias Detected"
            )
        
        # Interpretation
        if abs(stat_par_before) > 0.1 or abs(eq_opp_before) > 0.1:
            st.warning("⚠️ Significant fairness bias detected in the baseline model.")
        
    except Exception as e:
        st.error(f"❌ Failed to compute fairness metrics: {str(e)}")
        stat_par_before = 0.0
        eq_opp_before = 0.0
    
    # ============================================================
    # 6️⃣ APPLY REWEIGHING (ALIGNED WITH TRAINING CODE)
    # ============================================================
    st.markdown("### 🧩 Step 4: Apply Bias Mitigation (Reweighing)")
    
    try:
        RW = Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        RW.fit(train_bld)
        train_transf = RW.transform(train_bld)
        
        # Retrain model with reweighed samples
        model_rw = LogisticRegression(max_iter=500)
        model_rw.fit(train_transf.features, train_transf.labels.ravel(),
                     sample_weight=train_transf.instance_weights)
        
        y_pred_rw = model_rw.predict(X_test_scaled)
        acc_after = accuracy_score(y_test, y_pred_rw)
        
        st.write(f"**Model Accuracy (After Mitigation):** `{acc_after:.3f}`")
        
        # Compute post-mitigation fairness metrics
        test_pred_rw = test_bld.copy()
        test_pred_rw.labels = y_pred_rw.reshape(-1, 1)
        
        metric_test_rw = ClassificationMetric(
            test_bld, test_pred_rw,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        
        stat_par_after = safe_float(metric_test_rw.statistical_parity_difference())
        eq_opp_after = safe_float(metric_test_rw.equal_opportunity_difference())
        
        st.write("#### 📈 Fairness Metrics (After Mitigation)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Statistical Parity Difference",
                value=f"{stat_par_after:.3f}",
                delta=f"{stat_par_after - stat_par_before:+.3f}"
            )
        with col2:
            st.metric(
                label="Equal Opportunity Difference",
                value=f"{eq_opp_after:.3f}",
                delta=f"{eq_opp_after - eq_opp_before:+.3f}"
            )
        
    except Exception as e:
        st.warning(f"⚠️ Reweighing failed: {str(e)}. Using baseline model.")
        model_rw = model
        y_pred_rw = y_pred
        acc_after = acc_before
        stat_par_after = stat_par_before
        eq_opp_after = eq_opp_before
    
    # ============================================================
    # 7️⃣ VISUALIZATION (ALIGNED WITH TRAINING CODE)
    # ============================================================
    st.markdown("### 📊 Step 5: Results Visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chart 1: Fairness Metrics Comparison
    labels = ['Statistical Parity Diff', 'Equal Opportunity Diff']
    before = [stat_par_before, eq_opp_before]
    after = [stat_par_after, eq_opp_after]
    x = np.arange(len(labels))
    
    ax1.bar(x - 0.15, before, width=0.3, label='Before', color='skyblue')
    ax1.bar(x + 0.15, after, width=0.3, label='After', color='lightcoral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel('Metric Value (Closer to 0 = Fairer)')
    ax1.set_title('Fairness Improvement after Reweighing')
    ax1.legend()
    ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    
    # Chart 2: Accuracy Comparison
    accuracy_labels = ['Before Mitigation', 'After Mitigation']
    accuracy_values = [acc_before, acc_after]
    colors = ['skyblue', 'lightcoral']
    
    ax2.bar(accuracy_labels, accuracy_values, color=colors, alpha=0.7)
    ax2.set_ylabel('Accuracy Score')
    ax2.set_title('Model Accuracy Comparison')
    ax2.set_ylim([0, 1])
    
    # Add value labels on bars
    for i, v in enumerate(accuracy_values):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ============================================================
    # 8️⃣ RESULTS SUMMARY AND EXPORT
    # ============================================================
    st.markdown("### 📋 Step 6: Summary Report")
    
    # Create summary table
    summary_data = {
        "Metric": ["Accuracy", "Statistical Parity Diff", "Equal Opportunity Diff"],
        "Before Mitigation": [f"{acc_before:.3f}", f"{stat_par_before:.3f}", f"{eq_opp_before:.3f}"],
        "After Mitigation": [f"{acc_after:.3f}", f"{stat_par_after:.3f}", f"{eq_opp_after:.3f}"],
        "Change": [
            f"{acc_after - acc_before:+.3f}",
            f"{stat_par_after - stat_par_before:+.3f}",
            f"{eq_opp_after - eq_opp_before:+.3f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Improvement assessment
    st.markdown("#### 🎯 Fairness Improvement Assessment")
    
    spd_improvement = abs(stat_par_after) < abs(stat_par_before)
    eod_improvement = abs(eq_opp_after) < abs(eq_opp_before)
    
    if spd_improvement and eod_improvement:
        st.success("✅ Excellent! Both fairness metrics improved after mitigation.")
    elif spd_improvement or eod_improvement:
        st.info("🔄 Mixed results: One fairness metric improved, the other did not.")
    else:
        st.warning("⚠️ No improvement in fairness metrics. Consider trying other mitigation techniques.")
    
    # Export option
    if st.button("💾 Export Results to CSV"):
        results_df = pd.DataFrame({
            "Metric": ["Accuracy", "Statistical_Parity_Diff", "Equal_Opportunity_Diff"],
            "Before_Mitigation": [acc_before, stat_par_before, eq_opp_before],
            "After_Mitigation": [acc_after, stat_par_after, eq_opp_after],
            "Change": [acc_after - acc_before, stat_par_after - stat_par_before, eq_opp_after - eq_opp_before]
        })
        
        results_df.to_csv("fairness_results_streamlit.csv", index=False)
        st.success("✅ Results exported to 'fairness_results_streamlit.csv'")
    
    st.success("🎉 Fairness evaluation completed successfully!")

else:
    st.info("👆 Please upload your dataset to begin (e.g., `adult.csv`).")
    st.markdown("---")
    st.markdown("### ℹ️ Expected Dataset Format")
    st.markdown("""
    The app expects a CSV file with:
    - A target column named **'income'** (with values like '<=50K', '>50K')
    - A protected attribute column named **'sex'** (with values 'Male', 'Female')
    - Other demographic and employment features
    """)
    