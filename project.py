import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('spam_email_dataset_500.csv')  # Update with the actual file path

# Combine 'subject' and 'body' into one text feature
data['text'] = data['subject'] + ' ' + data['body']

# Extract features and labels
X = data['text']  # Text data (subject + body)
y = data['label']  # Labels (spam or ham)

# Convert the text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_transformed = vectorizer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search to find the best parameters
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters from grid search
best_params = grid_search.best_params_

# Train the best model
best_rf_clf = grid_search.best_estimator_
best_rf_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_rf_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Feature importance analysis
feature_importances = best_rf_clf.feature_importances_
top_n = 20  # Show top 20 most important features
indices = feature_importances.argsort()[::-1][:top_n]
top_features = [vectorizer.get_feature_names_out()[i] for i in indices]

# Display results
print(f"Best Hyperparameters: {best_params}")

print(f"Accuracy on test set: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_rep)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Display top N important features
print(f"Top {top_n} important features and their importance scores:")
for i, feature in enumerate(top_features):
    print(f"{i + 1}. {feature} (Importance: {feature_importances[indices[i]]:.4f})")