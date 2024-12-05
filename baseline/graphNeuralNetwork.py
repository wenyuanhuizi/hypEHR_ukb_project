import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer


print("Libraries imported successfully")

# Step 1: Load the data
labels = pd.read_csv('../data/edge-labels-ukb.txt', header=None, names=['label'])
with open('../data/hyperedges-ukb.txt', 'r') as f:
    hyperedges = f.readlines()

# Step 2: Create the feature matrix
hyperedges = [line.strip() for line in hyperedges]
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','), binary=True)
X = vectorizer.fit_transform(hyperedges).toarray()

# Step 3: Assign labels
data = pd.DataFrame(X)
data['label'] = labels['label']

# Encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Step 4: Train-test split
X = data.drop(columns=['label']).values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Step 5: Prepare the graph data
# Assume a fully connected graph for simplicity
num_nodes = X_train_tensor.shape[0]
edge_index = torch.combinations(torch.arange(num_nodes), r=2).T  # Fully connected edges

data = Data(x=X_train_tensor, edge_index=edge_index, y=y_train_tensor)

# Step 6: Define the Graph Neural Network
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

# Step 7: Train the model
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

# Step 8: Evaluate the model
model.eval()
with torch.no_grad():
    # Test data converted to PyTorch tensors
    test_data = Data(x=X_test_tensor, edge_index=edge_index)
    pred_logits = model(test_data.x, test_data.edge_index)
    y_pred = pred_logits.argmax(dim=1).numpy()

# Compute metrics
accuracy = accuracy_score(y_test_tensor, y_pred)
f1 = f1_score(y_test_tensor, y_pred, average="macro")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {f1:.4f}")
