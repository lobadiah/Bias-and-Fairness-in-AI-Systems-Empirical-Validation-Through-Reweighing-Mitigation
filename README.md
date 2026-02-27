# ⚖️ Fairness Evaluation and Mitigation Framework

A comprehensive tool for identifying and mitigating bias in machine learning models, developed by **Louis Obadiah** (MSc AI Engineering) and supervised by **Asst. Prof. Dr. John Olaifa** at Okan University.

This project provides both a web-based interactive application and a command-line interface (CLI) script to evaluate fairness metrics and apply bias mitigation techniques (specifically Reweighing) using the `aif360` library.

## 🚀 Features

- **Automated Data Cleaning:** Handles missing values and standardizes column names.
- **Dynamic Feature Selection:** Support for custom datasets by allowing users to select target and protected attributes.
- **Fairness Metrics:** Computes key metrics including:
  - **Statistical Parity Difference:** Measuring the difference in selection rates between groups.
  - **Equal Opportunity Difference:** Measuring the difference in true positive rates between groups.
- **Bias Mitigation:** Implements the **Reweighing** algorithm to balance datasets before model training.
- **Comparative Analysis:** Visualizes performance (accuracy) and fairness improvements side-by-side.
- **Exportable Results:** Save evaluation metrics and comparison charts for reporting.

## 🛠️ Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:

```bash
pip install aif360 fairlearn scikit-learn pandas numpy matplotlib seaborn streamlit
```

*Note: If you encounter issues with `aif360`, you might need to install additional dependencies like `tensorflow` for certain algorithms (though not required for the Reweighing algorithm used here).*

## 💻 Usage

### 1. Interactive Web App (Streamlit)
The Streamlit app provides a user-friendly way to upload a dataset and visualize the fairness impact.

```bash
streamlit run fairness_ai_app.py
```

1. Upload your CSV file (e.g., `adult.csv`).
2. Select the **Target Column** (e.g., `income`) and the **Protected Attribute** (e.g., `sex`).
3. Define the **Positive Class** and **Privileged Group**.
4. Review the baseline metrics and the improvements after mitigation.

### 2. Command Line Interface (CLI)
For automated workflows, use the standalone script:

```bash
python fairness_ai_framework.py
```

This script will process the `adult.csv` file in the current directory, perform the analysis, save a visualization as `bias_visualization.png`, and export results to `fairness_results.csv`.

## 📁 Project Structure

- `fairness_ai_app.py`: The Streamlit web application.
- `fairness_ai_framework.py`: The modular CLI script for fairness evaluation.
- `adult.csv`: Sample dataset (Census Income dataset).
- `fairness_results.csv`: Exported metrics from the CLI script.
- `bias_visualization.png`: Visual comparison of metrics.

## 📊 Methodology

1. **Preprocessing:** Data is cleaned, categorical variables are encoded, and features are scaled. Crucially, protected attributes are preserved as unscaled integers to ensure valid fairness metric calculations.
2. **Baseline Training:** A Logistic Regression model is trained on the original dataset.
3. **Fairness Assessment:** Metrics are calculated using `aif360` to detect bias against unprivileged groups.
4. **Mitigation:** The `Reweighing` algorithm is applied to generate sample weights that compensate for historical bias.
5. **Final Evaluation:** The model is retrained using the calculated weights, and metrics are re-evaluated to measure the reduction in bias.

## 🎓 Credits

- **Author:** Louis Obadiah (MSc AI Engineering)
- **Supervisor:** Asst. Prof. Dr. John Olaifa
- **Institution:** Okan University
