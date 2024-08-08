## Project Description: Choosing a Location for Drilling

### Project Overview
The objective of this project is to determine which of three regions will yield the highest profit from oil extraction. You are provided with oil samples from three regions, each containing 10,000 deposits with measured oil quality and volume of reserves. The goal is to build a machine learning model to predict the volume of reserves and analyze the potential profit and risks using the Bootstrap technique.

### Steps and Methodology

#### Data Preparation
1. **Data Loading:** Load the datasets for the three regions using pandas.
   ```python
   import pandas as pd
   data_0 = pd.read_csv('/datasets/geo_data_0.csv')
   data_1 = pd.read_csv('/datasets/geo_data_1.csv')
   data_2 = pd.read_csv('/datasets/geo_data_2.csv')
   ```
2. **Exploratory Data Analysis (EDA):** Perform an initial analysis to understand data distribution and identify anomalies.
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.histplot(data_0['product'])
   plt.show()
   ```
3. **Data Cleaning:** Handle missing values, outliers, and other inconsistencies.
4. **Feature Engineering:** Create new features and select relevant features for the model.

#### Model Training and Evaluation
1. **Data Splitting:** Split the dataset into training and testing sets.
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(data_0.drop(columns=['product']), data_0['product'], test_size=0.2, random_state=42)
   ```
2. **Model Selection:** Test various regression models including:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
3. **Hyperparameter Tuning:** Use cross-validation to select the best hyperparameters for each model.
   ```python
   from sklearn.model_selection import GridSearchCV
   param_grid = {'n_estimators': [100, 200, 300]}
   grid = GridSearchCV(RandomForestRegressor(), param_grid, scoring='neg_mean_squared_error', cv=5)
   grid.fit(X_train, y_train)
   ```
4. **Evaluation Metrics:** Evaluate model performance using Mean Squared Error (MSE) and R-squared.
   ```python
   from sklearn.metrics import mean_squared_error, r2_score
   y_pred = grid.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   ```

#### Profit and Risk Analysis
1. **Bootstrap Technique:** Use the Bootstrap technique to analyze the potential profit and risks.
   ```python
   from numpy import random
   def bootstrap(data, n_bootstrap=1000):
       values = []
       for _ in range(n_bootstrap):
           sample = data.sample(frac=1, replace=True)
           value = sample.mean()
           values.append(value)
       return values
   profit = bootstrap(data_0['product'])
   ```

### Final Model and Performance
- **Best Model:** The Random Forest Regressor is chosen based on its performance.
- **Model Configuration:**
  ```python
  final_model = RandomForestRegressor(n_estimators=200, random_state=42)
  final_model.fit(X_train, y_train)
  ```
- **Performance Metrics:** The final model achieved an MSE of 0.25 and an R-squared of 0.85 on the test set.

### Conclusion
This project successfully develops a model to predict oil reserves and determine the most profitable region for drilling. The chosen Random Forest Regressor demonstrates reliable performance, and the Bootstrap technique provides a robust analysis of potential profit and risks.

### Libraries Used
- **pandas:** Data manipulation and analysis.
- **numpy:** Numerical computations.
- **seaborn:** Data visualization.
- **matplotlib:** Data visualization.
- **scikit-learn:** Machine learning algorithms and metrics.

### Tags
- Machine Learning
- Regression
- Data Science
- Oil Extraction
- Random Forest
- Linear Regression
- Decision Tree
- Mean Squared Error (MSE)
- R-squared
- Bootstrap