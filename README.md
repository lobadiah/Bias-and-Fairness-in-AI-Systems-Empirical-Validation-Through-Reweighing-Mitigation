# Fairness Evaluation and Mitigation Framework

This project evaluates and mitigates bias in machine learning models using the AIF360 library. It provides a command-line interface for automated analysis and a Streamlit dashboard for interactive evaluation.

## Features
- Bias detection using Statistical Parity Difference and Equal Opportunity Difference.
- Bias mitigation using the Reweighing preprocessing algorithm.
- Visual comparison of model accuracy and fairness metrics before and after mitigation.
- Two interfaces: command-line workflow and Streamlit web dashboard.
- CSV export for generated results.

## Project Structure
```text
.
├── data/               # Raw and processed datasets
├── results/            # Generated reports and visualizations
├── src/                # Source code
│   ├── app.py          # Streamlit Web Application
│   └── cli.py          # Command-Line Interface Tool
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Interface
Run the automated evaluation on the provided Adult dataset:
```bash
python src/cli.py
```
Results will be saved in the `results/` directory.

### Streamlit Dashboard
Launch the interactive dashboard to upload and analyze your own datasets:
```bash
streamlit run src/app.py
```

## Methodology
The framework follows this fairness-aware machine learning pipeline:
1. Data cleaning to handle missing values and encode categorical features.
2. Baseline evaluation with Logistic Regression and fairness measurement.
3. Bias mitigation with AIF360 Reweighing on training data.
4. Final evaluation after mitigation and comparison of outcomes.

---
**Author:** Louis Obadiah (MSc AI Engineering)
**Supervisor:** Asst. Prof. Dr. John Olaifa, Okan University
