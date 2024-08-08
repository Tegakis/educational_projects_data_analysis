## Project Description: Customer Churn Prediction

### Project Overview
This project aims to develop a machine learning model to predict whether a customer will leave "Beta Bank" in the near future. The ability to retain existing customers is more cost-effective than acquiring new ones. Therefore, accurately predicting customer churn is crucial. The goal is to build a model with the highest possible F1-score, targeting a minimum of 0.59. Additionally, the model's performance will be evaluated using the AUC-ROC metric.

### Steps and Methodology

#### Data Preparation
1. **Data Loading:** Load the dataset using pandas.
   ```python
   data = pd.read_csv('/content/Churn_Modelling.csv')
   ```
2. **Feature Encoding:** Convert categorical features to numerical values using `OrdinalEncoder`.
3. **Data Scaling:** Normalize numerical features using `StandardScaler`.
4. **Handling Class Imbalance:** Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes.

#### Model Training and Evaluation
1. **Data Splitting:** Split the dataset into training and testing sets.
2. **Model Selection:** Test multiple models including:
   - Decision Tree Classifier
   - Random Forest Classifier
   - Dummy Classifier (for baseline comparison)
3. **Hyperparameter Tuning:** Use cross-validation to select the best hyperparameters for each model.
4. **Evaluation Metrics:** Evaluate model performance using F1-score and AUC-ROC.
   ```python
   from sklearn.metrics import f1_score, roc_auc_score
   ```

### Final Model and Performance
- **Best Model:** The Random Forest Classifier is selected based on its performance.
- **Model Configuration:**
  ```python
  RandomForestClassifier(n_estimators=100, random_state=42)
  ```
- **Performance Metrics:** The final F1-score and AUC-ROC are calculated on the test set to ensure the model meets the required threshold.

### Conclusion
This project successfully builds a model to predict customer churn, helping "Beta Bank" retain customers and optimize their marketing strategies. The model's performance metrics indicate its reliability and robustness in predicting churn, achieving the target F1-score and demonstrating good performance in AUC-ROC.

### Libraries Used
- **pandas:** Data manipulation and analysis.
- **matplotlib:** Data visualization.
- **scikit-learn:** Machine learning algorithms and metrics.
- **imblearn:** Techniques for handling imbalanced datasets.

### Tags
- Machine Learning
- Customer Churn
- Data Science
- Classification
- Imbalanced Data
- Random Forest
- SMOTE
- F1-score
- AUC-ROC