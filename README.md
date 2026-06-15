# Customer Churn Prediction (Binary Classification)

This repository contains a complete, beginner-friendly machine learning project designed to predict customer churn using the official **IBM Telco Customer Churn dataset** (popularly hosted on Kaggle). 

Customer churn occurs when a customer cancels their subscription or stops doing business with a service provider. In industries like telecommunications, predicting churn allows companies to proactively reach out to at-risk customers with retention offers, directly helping to increase customer lifetime value and protect revenue.

---

## Project Structure

```
Customer-Churn-Prediction/
├── customer_churn.csv       # IBM Telco Customer Churn dataset (7,043 rows)
├── download_data.py         # Python script to download the dataset from public sources
├── churn_prediction.ipynb   # Complete Jupyter Notebook with code, plots, and outputs
├── verify_notebook.py       # Programmatic validation script to test all code cells
└── README.md                # Project documentation and summary
```

---

## Technical Stack
- **Language**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn (Logistic Regression, Decision Trees, Random Forests)

---

## Machine Learning Pipeline

The project follows a standard industrial machine learning workflow, structured into clear, readable sections inside the [churn_prediction.ipynb](churn_prediction.ipynb) notebook:

### 1. Exploratory Data Analysis (EDA)
- **Class Imbalance check**: Visualized the distribution of the target variable `Churn` (73.46% stay, 26.54% leave).
- **Feature Distributions**: Analyzed continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) using histograms and kernel density estimations (KDE).
- **Multicollinearity & Correlation**: Calculated Pearson correlation coefficients for continuous attributes and plotted them on a Seaborn heatmap.
- **Categorical Relationships**: Discovered factors driving churn (e.g., *Month-to-month* contract structures and *Fiber optic* internet services are linked to higher churn rates).

### 2. Data Cleaning & Preprocessing
- **Handling Formatting Anomalies**: Converted the `TotalCharges` column (initially loaded as a string due to empty space strings for customers with `tenure = 0`) into a float type.
- **Imputation**: Filled the resulting `NaN` values in `TotalCharges` using the dataset's median.
- **Encoding Categorical Variables**: Converted binary features (`gender`, `Partner`, etc.) into `0`/`1` and multi-class categorical features (`Contract`, `PaymentMethod`, etc.) using **One-Hot Encoding** (dropping the first column to avoid the dummy variable trap).
- **Feature Scaling**: Scaled continuous numerical variables using Scikit-Learn's `StandardScaler` to help gradient-based models converge.

### 3. Model Training
We trained and compared three different models:
- **Logistic Regression**: Serves as a fast, linear baseline.
- **Decision Tree Classifier**: An interpretable tree-based model (depth capped at 5 to avoid overfitting).
- **Random Forest Classifier**: Robust ensemble method combining 100 decision trees.

### 4. Evaluation Metrics
Evaluated all models using:
- **Accuracy**: Overall correctness rate.
- **Precision**: Minimizes false positives (identifies actual churners out of predicted churners).
- **Recall (Sensitivity)**: Minimizes false negatives (identifies actual churners out of all real churners).
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Evaluates the model's ability to distinguish classes at all thresholds.

---

## Results & Model Comparison

The models achieved the following results on the hold-out test set (20% stratified split):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 81.00% | 66.00% | **56.00%** | **60.00%** | 0.8421 |
| **Random Forest** | **81.00%** | **67.00%** | 53.00% | **60.00%** | **0.8427** |
| **Decision Tree** | 79.00% | 63.00% | 55.00% | 58.00% | 0.8284 |

### Key Takeaways
- **Recall vs. Precision Trade-off**: In customer churn applications, **Recall** is often prioritized over Precision. Missing a customer who is about to leave (False Negative) is much more costly than sending a discount to a customer who intends to stay (False Positive). Logistic Regression provides the highest Recall (**56.00%**).
- **Best Overall Classifier**: **Random Forest** performed best in terms of **ROC-AUC (0.8427)** and **Precision (67.00%)**.

---

## Feature Importance

Using the Random Forest model, we extracted and plotted the top features that drive predictions. The analysis reveals that:
1. **tenure**: Total length of stay is the strongest predictor of churn (new customers are far more likely to leave).
2. **Contract Type**: Long-term contracts (Two-year and One-year) heavily reduce churn risk compared to month-to-month contracts.
3. **Monthly Charges & Total Charges**: Financial commitment levels have a major impact on decision-making.

---

## Real-World Predictions on New Samples

The notebook includes a pre-run evaluation of two dummy customer profiles using the final Random Forest classifier:
- **Customer A (High Risk)**: A customer with a Month-to-month contract, Fiber Optic internet, and Electronic check payment.
  - **Prediction**: **CHURN (1)** (Probability: **71.86%**)
- **Customer B (Low Risk)**: A customer with a Two-year contract, DSL internet, and Auto credit-card payment.
  - **Prediction**: **NO CHURN (0)** (Probability: **1.61%**)

---

## How to Get Started

### 1. Clone the repository
```bash
git clone https://github.com/hadiamazhar26/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. (Optional) Download dataset
If the dataset is missing, you can re-fetch it using:
```bash
python download_data.py
```

### 4. Run the Jupyter Notebook
Open the notebook in your browser:
```bash
jupyter notebook churn_prediction.ipynb
```
Select **Cell > Run All** in the menu to see the live results, visualizations, and predictions!
