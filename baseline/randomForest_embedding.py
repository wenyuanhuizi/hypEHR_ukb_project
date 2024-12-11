import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE


embedding_dict = {}
with open('../data/node-embeddings-ukb.txt', 'r') as f:
    next(f)  # Skip the first line (metadata)
    for line in f:
        parts = line.strip().split()
        code = parts[0]  # First part is the code
        embedding = np.array(parts[1:], dtype=float)  # Remaining parts are the embedding
        embedding_dict[code] = embedding

# Step 1: Load the data
labels = pd.read_csv('../data/edge-labels-ukb.txt', header=None, names=['label'])
with open('../data/hyperedges-ukb.txt', 'r') as f:
    hyperedges = f.readlines()

# Step 2: Create the feature matrix
hyperedges = [line.strip() for line in hyperedges]

person_embeddings = []
unmapped_codes = set()

for line in hyperedges:
    codes = line.strip().split(',')
    embeddings = [embedding_dict[code] for code in codes if code in embedding_dict]
    if embeddings:
        aggregated_embedding = np.mean(embeddings, axis=0)
        person_embeddings.append(aggregated_embedding)
    else:
        person_embeddings.append(np.zeros(len(next(iter(embedding_dict.values())))))
        unmapped_codes.update(set(codes) - embedding_dict.keys())

# Create feature matrix and assign labels
X = pd.DataFrame(person_embeddings)
y = labels['label']

assert len(labels) == len(person_embeddings), "Mismatch between labels and embeddings."

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 5: Normalize/standardize features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Step 6: Train a Random Forest model with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid, 
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
macro_f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'AUROC: {auroc:.4f}')
print(f'AUPR: {aupr:.4f}')
print(f'Macro-F1: {macro_f1:.4f}')
