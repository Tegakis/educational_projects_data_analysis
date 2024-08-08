## Project Description: 2017 Planning for "Streamchik" Online Store

### Project Overview
The objective of this project is to forecast sales metrics for the "Streamchik" online store using data from open sources. The data includes historical sales of games, user and critic ratings, genres, and platforms. By analyzing these data points, the project aims to identify patterns that can help in predicting potentially popular products and planning effective advertising campaigns.

### Steps and Methodology

#### Data Preparation
1. **Data Loading:** The dataset is loaded using pandas.
   ```python
   data = pd.read_csv('/datasets/games.csv')
   ```
2. **Exploratory Data Analysis (EDA):** Conduct EDA to understand the data distribution and identify any anomalies.
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.histplot(data['sales'])
   plt.show()
   ```
3. **Data Cleaning:** Handle missing values, outliers, and other data inconsistencies.
4. **Feature Engineering:** Create new features and select relevant features for modeling.
5. **Data Splitting:** Split the dataset into training and testing sets for model validation.
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']), data['sales'], test_size=0.2, random_state=42)
   ```

#### Model Training and Evaluation
1. **Model Selection:** Test multiple models including:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
2. **Hyperparameter Tuning:** Use cross-validation to select the best hyperparameters for each model.
   ```python
   from sklearn.model_selection import GridSearchCV
   param_grid = {'n_estimators': [100, 200, 300]}
   grid = GridSearchCV(RandomForestRegressor(), param_grid, scoring='neg_mean_absolute_error', cv=5)
   grid.fit(X_train, y_train)
   ```
3. **Evaluation Metrics:** Evaluate model performance using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
   ```python
   from sklearn.metrics import mean_absolute_error, mean_squared_error
   y_pred = grid.predict(X_test)
   mae = mean_absolute_error(y_test, y_pred)
   mse = mean_squared_error(y_test, y_pred)
   ```

### Final Model and Performance
- **Best Model:** The Random Forest Regressor is selected based on its performance.
- **Model Configuration:**
  ```python
  final_model = RandomForestRegressor(n_estimators=200, random_state=42)
  final_model.fit(X_train, y_train)
  ```
- **Performance Metrics:** The final MAE and MSE are calculated on the test set to ensure the model meets the required threshold.

### Conclusion
This project successfully builds a model to forecast sales metrics for the "Streamchik" online store, helping to identify potentially popular products and plan advertising campaigns. The chosen Random Forest Regressor model demonstrates reliable performance with low error metrics, indicating its effectiveness in predicting sales.

### Libraries Used
- **pandas:** Data manipulation and analysis.
- **numpy:** Numerical computations.
- **matplotlib:** Data visualization.
- **seaborn:** Data visualization.
- **scikit-learn:** Machine learning algorithms and metrics.
- **scipy:** Statistical analysis.

### Tags
- Machine Learning
- Regression
- Data Science
- Sales Forecasting
- Random Forest
- Linear Regression
- Decision Tree
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)