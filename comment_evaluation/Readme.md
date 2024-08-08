## Project Description: Comment Evaluation for Wikishop

### Project Overview
The online store "Wikishop" is launching a new service that allows users to edit and supplement product descriptions, similar to wiki communities. Customers can offer edits and comment on the changes of others. To ensure a positive user experience, the store needs a tool to detect and moderate toxic comments. The goal of this project is to build a model that categorizes comments as positive or negative with an F1-score of at least 0.75.

### Steps and Methodology

#### Data Preparation
1. **Data Loading:** The dataset, which includes labeled comments on their toxicity, is loaded using pandas.
   ```python
   import pandas as pd
   data = pd.read_csv('path/to/dataset.csv')
   ```
2. **Text Preprocessing:** Comments are preprocessed using techniques such as tokenization, lemmatization, and removal of stop words.
   ```python
   nltk.download('stopwords')
   nltk.download('wordnet')
   lemmatizer = WordNetLemmatizer()
   stop_words = set(nltk_stopwords.words('english'))
   ```
3. **Feature Extraction:** Use `TfidfVectorizer` to convert text data into numerical feature vectors.
   ```python
   vectorizer = TfidfVectorizer(stop_words=stop_words)
   X = vectorizer.fit_transform(data['comment'])
   ```

#### Model Training and Evaluation
1. **Data Splitting:** Split the dataset into training and testing sets.
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2, random_state=42)
   ```
2. **Model Selection:** Test multiple models including:
   - Logistic Regression
   - Random Forest Classifier
   - CatBoost Classifier
3. **Hyperparameter Tuning:** Use `GridSearchCV` to select the best hyperparameters for each model.
   ```python
   param_grid = {'C': [0.1, 1, 10]}
   grid = GridSearchCV(LogisticRegression(), param_grid, scoring='f1', cv=5)
   grid.fit(X_train, y_train)
   ```
4. **Evaluation Metrics:** Evaluate model performance using F1-score.
   ```python
   from sklearn.metrics import f1_score
   y_pred = grid.predict(X_test)
   f1 = f1_score(y_test, y_pred)
   ```

### Final Model and Performance
- **Best Model:** The Logistic Regression model is selected based on its performance.
- **Model Configuration:**
  ```python
  final_model = LogisticRegression(C=1, class_weight='balanced', solver='liblinear', random_state=42)
  final_model.fit(X_train, y_train)
  ```
- **Performance Metrics:** The final F1-score achieved is 0.753, meeting the project goal.

### Conclusion
The project successfully builds a model to detect and moderate toxic comments, helping "Wikishop" maintain a positive user environment. The chosen Logistic Regression model demonstrates reliable performance with an F1-score above the required threshold.

### Libraries Used
- **pandas:** Data manipulation and analysis.
- **nltk:** Natural language processing.
- **scikit-learn:** Machine learning algorithms and metrics.
- **catboost:** Gradient boosting on decision trees.

### Tags
- Machine Learning
- Text Classification
- Data Science
- Toxic Comment Detection
- Logistic Regression
- Random Forest
- CatBoost
- F1-score