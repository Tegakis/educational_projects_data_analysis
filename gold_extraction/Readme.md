## Project Description: Gold Recovery from Ore

### Project Overview
This project aims to develop a machine learning model to predict the gold recovery rate from gold-containing ore. The model will assist in optimizing production processes to prevent the plant from operating with unprofitable characteristics. The project is carried out for the company "Digits," which develops solutions for the efficient operation of industrial enterprises.

### Steps and Methodology

#### Data Preparation
1. **Data Loading:** The dataset is loaded using pandas.
   ```python
   data_train = pd.read_csv('/datasets/gold_recovery_train_new.csv')
   data_test = pd.read_csv('/datasets/gold_recovery_test_new.csv')
   data_full = pd.read_csv('/datasets/gold_recovery_full_new.csv')
   ```
2. **Exploratory Data Analysis (EDA):** Conduct EDA to understand the data distribution and identify any anomalies.
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.histplot(data_train['recovery_rate'])
   plt.show()
   ```
3. **Data Cleaning:** Handle missing values, outliers, and other data inconsistencies.
4. **Feature Engineering:** Create new features and select relevant features for modeling.
5. **Data Scaling:** Normalize numerical features using `StandardScaler`.
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   data_train_scaled = scaler.fit_transform(data_train)
   ```

#### Model Training and Evaluation
1. **Data Splitting:** Split the dataset into training and testing sets.
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(data_train_scaled, data_train['recovery_rate'], test_size=0.2, random_state=42)
   ```
2. **Model Selection:** Test multiple models including:
   - Decision Tree Regressor
   - Random Forest Regressor
   - Linear Regression
   - Dummy Regressor (for baseline comparison)
3. **Hyperparameter Tuning:** Use cross-validation to select the best hyperparameters for each model.
   ```python
   from sklearn.model_selection import GridSearchCV
   param_grid = {'n_estimators': [100, 200, 300]}
   grid = GridSearchCV(RandomForestRegressor(), param_grid, scoring='neg_mean_absolute_error', cv=5)
   grid.fit(X_train, y_train)
   ```
4. **Evaluation Metrics:** Evaluate model performance using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
   ```python
   from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
   y_pred = grid.predict(X_test)
   mae = mean_absolute_error(y_test, y_pred)
   mape = mean_absolute_percentage_error(y_test, y_pred)
   ```

### Final Model and Performance
- **Best Model:** The Random Forest Regressor is selected based on its performance.
- **Model Configuration:**
  ```python
  final_model = RandomForestRegressor(n_estimators=200, random_state=42)
  final_model.fit(X_train, y_train)
  ```
- **Performance Metrics:** The final MAE and MAPE are calculated on the test set to ensure the model meets the required threshold.

### Conclusion
The project successfully builds a model to predict the gold recovery rate, helping "Digits" optimize their production processes. The chosen Random Forest Regressor model demonstrates reliable performance with low error metrics, indicating its effectiveness in predicting gold recovery rates.

### Libraries Used
- **pandas:** Data manipulation and analysis.
- **seaborn:** Data visualization.
- **matplotlib:** Data visualization.
- **scikit-learn:** Machine learning algorithms and metrics.

### Tags
- Machine Learning
- Regression
- Data Science
- Gold Recovery
- Random Forest
- Decision Tree
- Linear Regression
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)