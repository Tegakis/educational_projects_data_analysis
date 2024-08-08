## Project Description: Tariff Recommendation

### Project Overview
The objective of this project is to build a classification model that recommends the most appropriate tariff for customers based on their behavior. The data has already been preprocessed, and the goal is to achieve the highest possible accuracy, with a target accuracy rate of at least 0.75.

### Steps and Methodology

#### Data Exploration and Preparation
1. **Data Loading:** Load the preprocessed dataset using pandas.
   ```python
   import pandas as pd
   data = pd.read_csv('/path/to/dataset.csv')
   ```
2. **Exploratory Data Analysis (EDA):** Perform an initial analysis to understand the data distribution and identify any anomalies.
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.histplot(data['tariff'])
   plt.show()
   ```
3. **Feature Selection:** Identify and select relevant features for the model.
4. **Data Splitting:** Split the dataset into training and testing sets.
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['tariff']), data['tariff'], test_size=0.2, random_state=42)
   ```

#### Model Training and Evaluation
1. **Model Selection:** Test various classification models including:
   - Decision Tree Classifier
   - Random Forest Classifier
   - Dummy Classifier (for baseline comparison)
2. **Hyperparameter Tuning:** Use cross-validation to select the best hyperparameters for each model.
   ```python
   from sklearn.model_selection import GridSearchCV
   param_grid = {'n_estimators': [100, 200, 300]}
   grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring='accuracy', cv=5)
   grid.fit(X_train, y_train)
   ```
3. **Evaluation Metrics:** Evaluate model performance using accuracy and confusion matrix.
   ```python
   from sklearn.metrics import accuracy_score, confusion_matrix
   y_pred = grid.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   conf_matrix = confusion_matrix(y_test, y_pred)
   ```

### Final Model and Performance
- **Best Model:** The Random Forest Classifier is chosen based on its performance.
- **Model Configuration:**
  ```python
  final_model = RandomForestClassifier(n_estimators=200, random_state=42)
  final_model.fit(X_train, y_train)
  ```
- **Performance Metrics:** The final model achieved an accuracy rate of 0.78, meeting the project goal.

### Conclusion
This project successfully develops a model to recommend the most suitable tariff for customers based on their behavior. The chosen Random Forest Classifier demonstrates reliable performance, exceeding the target accuracy rate of 0.75.

### Libraries Used
- **pandas:** Data manipulation and analysis.
- **seaborn:** Data visualization.
- **matplotlib:** Data visualization.
- **scikit-learn:** Machine learning algorithms and metrics.

### Tags
- Machine Learning
- Classification
- Data Science
- Tariff Recommendation
- Random Forest
- Decision Tree
- Accuracy