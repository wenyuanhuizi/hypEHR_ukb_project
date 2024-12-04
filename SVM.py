import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully")

# Step 1: Load the data
labels = pd.read_csv('./data/edge-labels-ukb.txt', header=None, names=['label'])
with open('./data/hyperedges-ukb.txt', 'r') as f:
    hyperedges = f.readlines()

# Step 2: Create the feature matrix
hyperedges = [line.strip() for line in hyperedges]
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','), binary=True)
X = vectorizer.fit_transform(hyperedges)
features = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Step 3: Assign labels
data = features.copy()
data['label'] = labels['label']

# Check data distribution
print("Label distribution:")
print(data['label'].value_counts())

# Plot data distribution
sns.countplot(x='label', data=data)
plt.title('Label Distribution')
plt.show()

# Step 4: Train-test split
X = data.drop(columns=['label'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 5: Normalize/standardize features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Step 6: Train an SVM model with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, 
                           scoring='f1_macro', cv=5, verbose=1, n_jobs=-1)

grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Step 7: Evaluate the model
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_pred_proba)
aupr = average_precision_score(y_test, y_pred_proba)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f'Accuracy: {accuracy:.2f}')
print(f'AUROC: {auroc:.2f}')
print(f'AUPR: {aupr:.2f}')
print(f'Macro-F1: {macro_f1:.2f}')
