# ⚖️ Fairness Evaluation and Mitigation Framework

A professional tool for evaluating and mitigating bias in machine learning models using the **AIF360** library. This framework provides both a **CLI tool** for automated analysis and a **Streamlit Web App** for interactive exploration.

## 🚀 Features
- **Bias Detection:** Computes fairness metrics like *Statistical Parity Difference* and *Equal Opportunity Difference*.
- **Bias Mitigation:** Implements the *Reweighing* algorithm to balance datasets before model training.
- **Visual Insights:** Generates comparative visualizations of accuracy and fairness metrics.
- **Dual Interface:** Choose between a command-line interface or a user-friendly web application.
- **Exportable Results:** Save your findings directly to CSV.

## 📁 Project Structure
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

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Command-Line Interface (CLI)
Run the automated evaluation on the provided Adult dataset:
```bash
python src/cli.py
```
Results will be saved in the `results/` directory.

### 2. Streamlit Web App
Launch the interactive dashboard to upload and analyze your own datasets:
```bash
streamlit run src/app.py
```

## 📊 Methodology
The framework follows a standard fairness-aware machine learning pipeline:
1. **Data Cleaning:** Handling missing values and encoding categorical features.
2. **Baseline Evaluation:** Training a Logistic Regression model and measuring its fairness.
3. **Mitigation:** Applying AIF360's `Reweighing` algorithm to the training data.
4. **Final Evaluation:** Retraining the model on reweighed data and comparing results.

---
**Author:** Louis Obadiah (MSc AI Engineering)
**Supervisor:** Asst. Prof. Dr. John Olaifa, Okan University
