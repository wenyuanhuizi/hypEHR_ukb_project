import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


print("Libraries imported successfully")

# Step 1: Load the data
labels = pd.read_csv('../data/edge-labels-ukb.txt', header=None, names=['label'])
with open('../data/hyperedges-ukb.txt', 'r') as f:
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

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Step 6: Define the FeedForward Neural Network
class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output layer for binary classification
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

# Initialize model, loss, and optimizer
input_size = X_train_tensor.shape[1]
model = FeedForwardNN(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 7: Train the model
epochs = 20
batch_size = 64
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Step 8: Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor).numpy()
    y_pred = y_pred_probs.argmax(axis=1)

accuracy = accuracy_score(y_test_tensor, y_pred)
auroc = roc_auc_score(y_test_tensor, y_pred_probs[:, 1])
aupr = average_precision_score(y_test_tensor, y_pred_probs[:, 1])
macro_f1 = f1_score(y_test_tensor, y_pred, average='macro')

print(f'Accuracy: {accuracy:.2f}')
print(f'AUROC: {auroc:.2f}')
print(f'AUPR: {aupr:.2f}')
print(f'Macro-F1: {macro_f1:.2f}')
