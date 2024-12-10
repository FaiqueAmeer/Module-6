import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'U:\Inst 414\police_data.csv'
df = pd.read_csv(file_path)

# Data cleaning missing values in features
df['driver_gender'].fillna(df['driver_gender'].mode()[0], inplace=True)
df['driver_age'].fillna(df['driver_age'].median(), inplace=True)
df['stop_duration'].fillna('Unknown', inplace=True)

# Ensuring the target column (is_arrested) is binary
df['is_arrested'] = df['is_arrested'].fillna(0).astype(int)  # Convert to binary integers

# Converting categorical columns into numeric format
categorical_columns = ['driver_gender', 'driver_race', 'violation', 'stop_outcome', 'stop_duration']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define features (X) and target (y)
X = df.drop(columns=['is_arrested', 'stop_date', 'stop_time', 'search_conducted', 'drugs_related_stop'])
y = df['is_arrested']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Arrested', 'Arrested'], yticklabels=['Not Arrested', 'Arrested'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Featuring importance bar chart
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# To analyze misclassified samples
X_test['Actual'] = y_test
X_test['Predicted'] = y_pred
errors = X_test[X_test['Actual'] != X_test['Predicted']]
print("Top 5 Misclassified Samples:")
print(errors.head(5))
