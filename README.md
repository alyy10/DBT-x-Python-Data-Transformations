# DBT-x-Python-Data-Transformations

This repository contains a prototype for a profit scoring model designed for digital loan underwriting. The project leverages dbt, PostgreSQL, and Python to build a comprehensive data pipeline, from data ingestion and transformation to model training, prediction, and deployment. The core objective is to predict the potential profit or loss for individual loans, enabling financial institutions to make more informed decisions.
<img width="570" height="230" alt="image" src="https://github.com/user-attachments/assets/ddf7a396-4bac-4eed-b6ea-9f6e18bd26f1" />

## Detailed Component Analysis

**`dataset.ipynb`**: A Jupyter Notebook that outlines the process of fetching the raw loan data from Bondora's public API, performing initial data loading, and saving it into a more efficient Parquet format. This notebook is crucial for understandingHxunderstanding the data source and its initial preparation.

- **`dbt_transform_data.py`**: A Python script responsible for executing the dbt transformations. It orchestrates the running of dbt commands, applying the defined data models to the data in the PostgreSQL database.
- **`fetching_and_writing_data.py`**: This script handles the initial data ingestion. It fetches external data (likely from the Bondora API or a similar source) and writes it into the PostgreSQL database, serving as the starting point for the data pipeline.
- **`main.py`**: The primary entry point for running the entire profit scoring pipeline. This script sequentially calls `fetch_and_write_data.py`, `dbt_transform_data.py`, and `run_scoring.py` to execute the complete workflow from data fetching to profit scoring. It provides a unified interface for automating the process.
- **`scoring.py`**: This script is responsible for loading the pre-trained profit scoring model, performing predictions on the transformed data, and saving the results (e.g., predicted profit/loss and profit scores) back into the PostgreSQL database. It integrates the machine learning model into the data pipeline.
- **`README.md`**: A dedicated README file within this directory provides specific instructions on how to use the `main.py` script and outlines the prerequisites, such as setting up the PostgreSQL database and dbt project, and installing necessary Python packages.
- **`lgb_reg.pkl`**: This file contains the serialized (pickled) LightGBM Regressor model. This model is pre-trained on historical loan data to predict the `ProfitLoss` for new loan applications. Storing the model binary here allows for its easy loading and use in the `scoring.py` script without needing to retrain it every time the pipeline runs.

  <img width="1098" height="108" alt="image" src="https://github.com/user-attachments/assets/38ec514b-f55f-40cc-8fc2-580d79d096b6" />


### `prof_scoring.ipynb`

This Jupyter Notebook is the central piece for understanding the profit scoring methodology. It walks through the entire process of building and evaluating the profit scoring model. The key sections and their outputs are as follows:

#### 1. Data Import and Initial Exploration

The notebook begins by importing necessary libraries and loading the `dataset.parquet` file. Initial data exploration steps are performed to understand the dataset's structure, data types, and summary statistics.

```python
# Data head
# Output from data.head()
# (Example output, actual output will vary based on data)
#    Amount  InterestRate  ...  TotalIncome  ProfitLoss
# 0  1000.0         15.00  ...      1200.00       50.00
# 1  2000.0         12.50  ...      2500.00      100.00
# ...

# Data info
# Output from data.info()
# (Example output, actual output will vary based on data)
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10000 entries, 0 to 9999
# Data columns (total 20 columns):
#  #   Column        Non-Null Count  Dtype  
# ---  ------        --------------  -----  
#  0   Amount        10000 non-null  float64
#  1   InterestRate  10000 non-null  float64
# ...

# Data describe
# Output from data.describe()
# (Example output, actual output will vary based on data)
#             Amount  InterestRate  ...  TotalIncome  ProfitLoss
# count  10000.000000  10000.000000  ...  10000.000000  10000.000000
# mean    2500.000000     18.000000  ...   3000.000000    150.000000
# ...
```

#### 2. Data Preprocessing

This section focuses on cleaning and preparing the data for model training. It involves dropping irrelevant columns, handling missing values, and converting categorical features into numerical representations using one-hot encoding.

#### 3. Feature Engineering

New features are engineered to enhance the model's predictive power. Examples include `Age` (derived from `DateOfBirth`), `IncomePerPerson` (calculated from `TotalIncome` and `NrOfDependants`), and `LoanToIncomeRatio` (from `Amount` and `TotalIncome`).

#### 4. Model Training

A LightGBM Regressor model is initialized and trained on the preprocessed and engineered dataset. The data is split into training and testing sets to evaluate the model’s performance.

#### 5. Profit Scoring

The trained model is then used to predict the `ProfitLoss` for all loans, and a `ProfitScore` is assigned based on these predictions. The `ProfitScore` categorizes loans into 'High Profit', 'Neutral', or 'Low Profit'.

```python
# Output from data[['Amount', 'ProfitLoss', 'PredictedProfitLoss', 'ProfitScore']].head()
# (Example output, actual output will vary based on data)
#    Amount  ProfitLoss  PredictedProfitLoss ProfitScore
# 0  1000.0       50.00              48.50  High Profit
# 1  2000.0      100.00             102.10  High Profit
# ...
```

#### 6. Visualization

Visualizations are generated to understand the distribution of profit scores and the relationship between loan amount, profit/loss, and profit score. These plots help in interpreting the model’s output and its implications.

<img width="1233" height="726" alt="image" src="https://github.com/user-attachments/assets/1f1e6eb2-4ef2-4a19-9d53-51dbbfae1002" />


#### 7. Conclusion

The notebook concludes with a summary of the profit scoring model and suggestions for future improvements, such as more advanced feature engineering, hyperparameter tuning, exploring other machine learning models, incorporating more data sources, and implementing a more sophisticated profit scoring logic.

## Results

This section summarizes the key results and findings from the execution of the `prof_scoring.ipynb` notebook . The analysis uses the Bondora dataset to evaluate profit scoring models, comparing linear and tree-based approaches, and employing SHAP for interpretability, particularly in downturn vs. normal economic periods.

### Model Evaluation Metrics

- **Linear Model (Ridge Regression)**: Somers D statistic of 52.94%, indicating moderate concordance between predictions and actual values.
- **Tree-based Model (LightGBM Regressor)**: Spearman rho correlation of 57.58% between predicted and observed Annualized Return Rate (ARR). Gini coefficient of 0.49 for both ARR predictions and the Bondora Probability of Default (PD) model, suggesting moderate discriminatory power.

### Key Findings

- **Profitability Impact of Credit Score**: The credit score (likely `CreditScoreEsMicroL`) drives down profitability by 3 percentage points in downturn periods compared to normal times, as revealed by SHAP value differences. This highlights the limitations of traditional credit scoring in economic stress scenarios.
- **ARR Ratings and Trends**: ARR ratings (binned into 10 categories) show a trend where higher ratings correspond to better observed ARR (e.g., rating 1: 0.263522 observed ARR; rating 10: -0.883354). The Spearman rho of 57.58% indicates a moderate correlation between predicted and observed ARR.
- **Comparison with Credit Scoring**: Pivot tables and 3D surface plots compare ARR and PD ratings, showing mean risk-adjusted ARR as percentages. The Gini of 0.49 for both models suggests similar performance in distinguishing defaults, but profit scoring provides additional insights into returns.
- **SHAP Analysis**: Average SHAP values were computed for downturn (`dt`) and normal (`out`) samples. Differences were calculated for 42 features (e.g., `BidsPortfolioManager`, `BidsApi`, `BidsManual`, `NewCreditCustomer`, `CreditScoreEsMicroL`, etc.), with the top 13 by absolute difference visualized. Positive differences indicate features boosting predictions more in downturns; negative ones (like credit score) reduce them. Specific SHAP values per feature were not fully detailed.

### Visualizations and Implications

- **ARR and Default Rate Trend Plot**: Dual-axis line plot showing average observed ARR (blue) and default rate (red) over time, indicating potential correlations for risk assessment.
- **Feature Importance (LightGBM)**: Bar chart of top 10 features by gain, helping identify key predictors for ARR.
- **3D Surface Plots**: Two plots visualizing PD vs. ARR ratings with viridis colormap, supporting comparative analysis of scoring systems.
- **ROC Curve**: Shows Gini=0.49, implying moderate model performance in default prediction.
- **SHAP Summary Plots**: Bar plots of SHAP differences, highlighting feature impacts across scenarios.
- **SHAP Difference Bar Chart**: Horizontal bars for top 13 features, colored by sign (red for negative, blue for positive), emphasizing how features like credit score affect profitability differently in downturns.

These results underscore the value of profit scoring over traditional risk models, particularly in identifying features that disproportionately impact returns during economic downturns. Future work could include more granular SHAP value reporting and additional metrics like MSE/R2 for comprehensive evaluation.
