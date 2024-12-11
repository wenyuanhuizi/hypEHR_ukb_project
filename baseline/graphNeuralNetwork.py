import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import numpy as np

print("Libraries imported successfully")

# Step 1: Load embeddings
embedding_dict = {}
with open('../data/node-embeddings-ukb.txt', 'r') as f:
    next(f)  # Skip the first line (metadata)
    for line in f:
        parts = line.strip().split()
        code = parts[0]  # First part is the code
        embedding = np.array(parts[1:], dtype=float)  # Remaining parts are the embedding
        embedding_dict[code] = embedding

# Step 2: Load hyperedges and labels
labels = pd.read_csv('../data/edge-labels-ukb.txt', header=None, names=['label'])
with open('../data/hyperedges-ukb.txt', 'r') as f:
    hyperedges = f.readlines()

# Step 3: Create the feature matrix
hyperedges = [line.strip() for line in hyperedges]

person_embeddings = []
unmapped_codes = set()

for line in hyperedges:
    codes = line.split(',')
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

# Step 4: Train-test split
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

# Step 6: Create edge_index (fully connected graph)
num_nodes_train = X_train_tensor.shape[0]
train_edge_index = torch.combinations(torch.arange(num_nodes_train), r=2).T  # Fully connected graph for training

# Step 7: Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize the model
input_dim = X_train_tensor.shape[1]
hidden_dim = 128
output_dim = len(set(y_train_tensor.numpy()))
model = GNN(input_dim, hidden_dim, output_dim)

# Define optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Step 8: Train the model
data = Data(x=X_train_tensor, edge_index=train_edge_index, y=y_train_tensor)

epochs = 200
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Step 9: Evaluate the model on test data
model.eval()
with torch.no_grad():
    num_nodes_test = X_test_tensor.shape[0]
    test_edge_index = torch.combinations(torch.arange(num_nodes_test), r=2).T  # Fully connected graph for testing
    test_data = Data(x=X_test_tensor, edge_index=test_edge_index)
    pred_logits = model(test_data.x, test_data.edge_index)
    y_pred = pred_logits.argmax(dim=1).numpy()
    y_pred_proba = F.softmax(pred_logits, dim=1).numpy()[:, 1]  # Probabilities for positive class

# Compute metrics
accuracy = accuracy_score(y_test_tensor, y_pred)
auroc = roc_auc_score(y_test_tensor, y_pred_proba)
aupr = average_precision_score(y_test_tensor, y_pred_proba)
f1 = f1_score(y_test_tensor, y_pred, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"AUROC: {auroc:.4f}")
print(f"AUPR: {aupr:.4f}")
print(f"Weighted F1: {f1:.4f}")
