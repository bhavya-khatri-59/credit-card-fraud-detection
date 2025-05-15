import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv('creditcard.csv')
<<<<<<< HEAD

# Separate features and target
data = data.groupby('Class', group_keys=False).sample(frac=0.2, random_state=42)
data = data.dropna(subset=["Class"])
X = data.drop(columns=['Class'])
y = data['Class']
=======
# Sample a smaller portion of the data to reduce runtime
# Ensure we keep the class imbalance
data_sampled = data.groupby('Class').apply(lambda x: x.sample(frac=0.25, random_state=42)).reset_index(drop=True)

# Separate features and target
X = data_sampled.drop(columns=['Class'])
y = data_sampled['Class']
>>>>>>> cf7aa6040be7f4d5edadf8b35fbb33acbe340006
print(X.value_counts())
print(y.value_counts())

print(data.hist(bins=30, figsize = (30, 30)))
print(X.describe())

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled_df.describe())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to handle class imbalance in the training data
smote = SMOTE(sampling_strategy = 0.2, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

<<<<<<< HEAD
# Logistic Regression model
from sklearn.linear_model import LogisticRegression
logistic_regression_model = LogisticRegression(solver='liblinear')
logistic_regression_model.fit(X_train_resampled, y_train_resampled)

y_pred = logistic_regression_model.predict(X_test)

# Evaluation for Logistic Regression
=======
# Train a RandomForest Classifier with class weights
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
>>>>>>> cf7aa6040be7f4d5edadf8b35fbb33acbe340006
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
<<<<<<< HEAD
roc_auc = roc_auc_score(y_test, logistic_regression_model.predict_proba(X_test)[:, 1])
=======
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
>>>>>>> cf7aa6040be7f4d5edadf8b35fbb33acbe340006

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

<<<<<<< HEAD
# Visualization for logistic regression
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Class Distribution Plot
sns.countplot(x='Class', data=data, palette='Set2', ax=ax[0])
=======
# Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Class Distribution Plot
sns.countplot(x='Class', data=data_sampled, palette='Set2', ax=ax[0])
>>>>>>> cf7aa6040be7f4d5edadf8b35fbb33acbe340006
ax[0].set_title('Class Distribution (Sampled Dataset)')

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'], ax=ax[1])
ax[1].set_title('Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
<<<<<<< HEAD

# ROC curve for logistic regression
from sklearn.metrics import roc_curve
fpr, tpr, thresolds = roc_curve(y_test, logistic_regression_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Random Forest
random_forest_model = RandomForestClassifier(random_state=42, class_weight='balanced')
random_forest_model.fit(X_train_resampled, y_train_resampled)

# Evaluate performance for random forest
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, random_forest_model.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Visualization for random forest
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Class Distribution Plot
sns.countplot(x='Class', data=data, palette='Set2', ax=ax[0])
ax[0].set_title('Class Distribution (Sampled Dataset)')

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'], ax=ax[1])
ax[1].set_title('Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

#random forest roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresolds = roc_curve(y_test, random_forest_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# svm model
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', probability=True, random_state = 42)
svm_model.fit(X_train_resampled, y_train_resampled)

# Evaluate performance for svm model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Visualization for svm model
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Class Distribution Plot
sns.countplot(x='Class', data=data, palette='Set2', ax=ax[0])
ax[0].set_title('Class Distribution (Sampled Dataset)')

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'], ax=ax[1])
ax[1].set_title('Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

#svm model roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresolds = roc_curve(y_test, svm_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

#XG Boost model
import xgboost as xgb
xgb_model = xgb.XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

y_pred = xgb_model.predict(X_test)

# Evaluate performance for xgb model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Visualization for xgb boost model
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Class Distribution Plot
sns.countplot(x='Class', data=data, palette='Set2', ax=ax[0])
ax[0].set_title('Class Distribution (Sampled Dataset)')

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'], ax=ax[1])
ax[1].set_title('Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

#xgb model roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresolds = roc_curve(y_test, svm_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
=======
>>>>>>> cf7aa6040be7f4d5edadf8b35fbb33acbe340006
