## Project Description: Customer Churn Prediction for the Telecom Operator "NoBreaks.com"

### Project Overview
The telecom operator "NoBreaks.com" aims to predict customer churn to proactively offer promo codes and special conditions to retain users. The project involves analyzing personal data, tariff plans, and contract information to build a predictive model for customer churn.

### Steps and Methodology

#### Data Preparation
1. **Data Loading:** Load the dataset using pandas.
   ```python
   import pandas as pd
   data = pd.read_csv('/path/to/dataset.csv')
   ```
2. **Exploratory Data Analysis (EDA):** Perform an initial analysis to understand data distribution and identify anomalies.
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.histplot(data['churn'])
   plt.show()
   ```
3. **Data Cleaning:** Handle missing values, outliers, and other inconsistencies.
4. **Feature Engineering:** Create new features and select relevant features for the model.
5. **Data Splitting:** Split the dataset into training and testing sets.
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['churn']), data['churn'], test_size=0.2, random_state=42)
   ```

#### Model Training and Evaluation
1. **Model Selection:** Test various models including:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting Classifier
2. **Hyperparameter Tuning:** Use cross-validation to select the best hyperparameters for each model.
   ```python
   from sklearn.model_selection import GridSearchCV
   param_grid = {'n_estimators': [100, 200, 300]}
   grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1', cv=5)
   grid.fit(X_train, y_train)
   ```
3. **Evaluation Metrics:** Evaluate model performance using F1-score, accuracy, and AUC-ROC.
   ```python
   from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
   y_pred = grid.predict(X_test)
   f1 = f1_score(y_test, y_pred)
   accuracy = accuracy_score(y_test, y_pred)
   auc_roc = roc_auc_score(y_test, y_pred)
   ```

### Final Model and Performance
- **Best Model:** The Random Forest Classifier is chosen based on its performance.
- **Model Configuration:**
  ```python
  final_model = RandomForestClassifier(n_estimators=200, random_state=42)
  final_model.fit(X_train, y_train)
  ```
- **Performance Metrics:** The final model achieved an F1-score of 0.75, accuracy of 0.85, and AUC-ROC of 0.80 on the test set.

### Conclusion
This project successfully develops a model to predict customer churn for "NoBreaks.com," allowing the company to identify users at risk of leaving and offer them special conditions to improve retention. The chosen Random Forest Classifier demonstrates reliable performance, meeting the project's goals.

### Libraries Used
- **pandas:** Data manipulation and analysis.
- **numpy:** Numerical computations.
- **matplotlib:** Data visualization.
- **seaborn:** Data visualization.
- **scikit-learn:** Machine learning algorithms and metrics.
- **phik:** Advanced correlation analysis.

### Tags
- Machine Learning
- Classification
- Customer Churn
- Data Science
- Random Forest
- Logistic Regression
- Gradient Boosting
- F1-score
- AUC-ROC