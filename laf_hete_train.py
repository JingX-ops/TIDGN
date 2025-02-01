import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, average_precision_score
from sklearn.metrics import precision_recall_curve
import glob
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATv2Conv
from torch.nn import Parameter
import os.path as osp
import random
from typing import Sequence, Union
from torch_geometric.nn.inits import glorot, zeros
from tqdm import *
from torch.nn import Linear,CrossEntropyLoss
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class GConvLSTM(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Long Short Term Memory
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int=4,
        bias: bool = True,
    ):
        super(GConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):
        self.conv_x_i = torch.nn.ModuleList()
        self.conv_x_i.append(SAGEConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_x_i.append(GATv2Conv(
            in_channels=self.heads * self.out_channels,
            out_channels=self.out_channels,
            heads=self.heads,
            concat=False,
            edge_dim=1,
            bias=self.bias,
        ))

        self.conv_h_i = torch.nn.ModuleList()
        self.conv_h_i.append(SAGEConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))

        self.conv_h_i.append(GATv2Conv(
            in_channels=self.heads*self.out_channels,
            out_channels=self.out_channels,
            heads=self.heads,
            concat=False,
            edge_dim=1,
            bias=self.bias,
        ))

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_x_f = torch.nn.ModuleList()
        self.conv_x_f.append(SAGEConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_x_f.append(GATv2Conv(
            in_channels=self.heads * self.out_channels,
            out_channels=self.out_channels,
            heads=self.heads,
            concat=False,
            edge_dim=1,
            bias=self.bias,
        ))
        self.conv_h_f = torch.nn.ModuleList()
        self.conv_h_f.append(SAGEConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))

        self.conv_h_f.append(GATv2Conv(
            in_channels=self.heads * self.out_channels,
            out_channels=self.out_channels,
            heads=self.heads,
            concat=False,
            edge_dim=1,
            bias=self.bias,
        ))
        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):
        self.conv_x_c = torch.nn.ModuleList()
        self.conv_x_c.append(SAGEConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_x_c.append(GATv2Conv(
            in_channels=self.heads * self.out_channels,
            out_channels=self.out_channels,
            heads=self.heads,
            concat=False,
            edge_dim=1,
            bias=self.bias,
        ))
        self.conv_h_c = torch.nn.ModuleList()
        self.conv_h_c.append(SAGEConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))

        self.conv_h_c.append(GATv2Conv(
            in_channels=self.heads * self.out_channels,
            out_channels=self.out_channels,
            heads=self.heads,
            concat=False,
            edge_dim=1,
            bias=self.bias,
        ))
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):
        self.conv_x_o = torch.nn.ModuleList()
        self.conv_x_o.append(SAGEConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_x_o.append(GATv2Conv(
            in_channels=self.heads * self.out_channels,
            out_channels=self.out_channels,
            heads=self.heads,
            concat=False,
            edge_dim=1,
            bias=self.bias,
        ))
        self.conv_h_o = torch.nn.ModuleList()
        self.conv_h_o.append(SAGEConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))

        self.conv_h_o.append(GATv2Conv(
            in_channels=self.heads * self.out_channels,
            out_channels=self.out_channels,
            heads=self.heads,
            concat=False,
            edge_dim=1,
            bias=self.bias,
        ))
        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, H, C):
#        X = self.conv_x_i[0](X, edge_index=edge_index, edge_attr=edge_weight)
        I = self.conv_x_i[0](X, edge_index)
#        H = self.conv_h_i[0](H, edge_index, edge_weight)
        I = I + self.conv_h_i[0](H, edge_index)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, H, C):
#        X = self.conv_x_f[0](X, edge_index, edge_weight)
        F = self.conv_x_f[0](X, edge_index)
#        H = self.conv_h_f[0](H, edge_index, edge_weight)
        F = F + self.conv_h_f[0](H, edge_index)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, H, C, I, F):
#        X = self.conv_x_c[0](X, edge_index, edge_weight)
        T = self.conv_x_c[0](X, edge_index)
#        H = self.conv_h_c[0](H, edge_index, edge_weight)
        T = T + self.conv_h_c[0](H, edge_index)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, H, C):
#        X = self.conv_x_o[0](X, edge_index, edge_weight)
        O = self.conv_x_o[0](X, edge_index)
#        H = self.conv_h_o[0](H, edge_index, edge_weight)
        O = O + self.conv_h_o[0](H, edge_index)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, H, C)
        F = self._calculate_forget_gate(X, edge_index, H, C)
        C = self._calculate_cell_state(X, edge_index, H, C, I, F)
        O = self._calculate_output_gate(X, edge_index, H, C)
        H = self._calculate_hidden_state(O, C)
        return H, C

class GCNLSTM(nn.Module):
    def __init__(self, node_features, hidden_dim, num_classes=168, dropout_rate=0.5):
        super(GCNLSTM, self).__init__()
        self.recurrent = GConvLSTM(in_channels=node_features, out_channels=hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim * 168, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes * num_classes)
        )

    def forward(self, temporal_data):
        h, c = None, None
        for data in temporal_data:  # temporal_data is a list of Data objects
            x, edge_index = data.x, data.edge_index
            h, c = self.recurrent(x, edge_index, h, c)
        h = self.dropout(F.relu(h))

        # Flatten the hidden state before passing to MLP
        h = h.view(-1, h.size(1) * h.size(0))  # Ensure h is flattened correctly

        # Pass through MLP
        x = self.MLP(h)

        # Reshape the output to (num_classes, num_classes)
        x = x.view(-1, 168, 168)
        return x

class GCNLSTM(nn.Module):
    def __init__(self, node_features, hidden_dim, num_classes=168, dropout_rate=0.5):
        super(GCNLSTM, self).__init__()
        self.recurrent = GConvLSTM(in_channels=node_features, out_channels=hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim * 168, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes * num_classes)
        )

    def forward(self, temporal_data):
        h, c = None, None
        for data in temporal_data:  # temporal_data is a list of Data objects
            x, edge_index = data.x, data.edge_index
            h, c = self.recurrent(x, edge_index, h, c)
        h = self.dropout(F.relu(h))

        # Flatten the hidden state before passing to MLP
        h = h.view(-1, h.size(1) * h.size(0))  # Ensure h is flattened correctly

        # Pass through MLP
        x = self.MLP(h)

        # Reshape the output to (num_classes, num_classes)
        x = x.view(-1, 168, 168)
        return x

train_features_dir = './Feature/LAF in FUS-LAF/feature_train'
train_edges_dir = './Contactmap/LAF in FUS-LAF/distance_train'
test_features_dir = './Feature/LAF in FUS-LAF/feature_test'
test_edges_dir = './Contactmap/LAF in FUS-LAF/distance_test'
label_file = './Dataset/LAF in FUS-LAF/test_laf_hete.txt'

model_path = './Model/LAF in FUS-LAF/Pretrain_laf_c.pth'
gcn_lstm_model = GCNLSTM(node_features=25, hidden_dim=168)
gcn_lstm_model.load_state_dict(torch.load(model_path))
gcn_lstm_model.eval()  

class GCN_MLP(nn.Module):
    def __init__(self, gcn_lstm_model):
        super(GCN_MLP, self).__init__()
        self.gcn_lstm = gcn_lstm_model
        self.mlp = nn.Sequential(
            nn.Flatten(),  # Flatten the output to match the target shape
            nn.Linear(168 * 168, 512),
            nn.ReLU(),
            nn.Linear(512, 168)  # Make sure the output is [batch_size, 168]
        )
    
    def forward(self, temporal_data):
        output = self.gcn_lstm(temporal_data)
        output = self.mlp(output)
        return output  # Output shape should be [batch_size, 168]

def load_real_data(features_dir, edges_dir, i, labels):
    features = []
    edges = []
    for j in range(1, 11):
        feature_file_path = os.path.join(features_dir, f"{i}.{j}.txt")
        edge_file_path = os.path.join(edges_dir, f"{i}.{j}.txt")

        feature_data = np.loadtxt(feature_file_path, delimiter=',', usecols=range(2, 27))
        feature_tensor = torch.tensor(feature_data, dtype=torch.float)

        edge_data = np.loadtxt(edge_file_path, delimiter=',', usecols=(0, 1)).astype(int) - 1
        edge_tensor = torch.tensor(edge_data, dtype=torch.long).t()

        features.append(feature_tensor)
        edges.append(edge_tensor)

    temporal_data = [Data(x=f, edge_index=e) for f, e in zip(features, edges)]
    return temporal_data, labels

def load_labels(label_file):

    labels = np.loadtxt(label_file, delimiter=' ')    

    labels = torch.tensor(labels, dtype=torch.float)
    
    return labels

def get_existing_indices(features_dir):
    pattern = os.path.join(features_dir, "*.1.txt")
    file_paths = glob.glob(pattern)
    existing_indices = [int(os.path.basename(fp).split('.')[0]) for fp in file_paths]
    return sorted(existing_indices)

def create_data_loader(features_dir, edges_dir, label_file, batch_size=1):
    labels = load_labels(label_file)
    existing_indices = get_existing_indices(features_dir) 
    data_list = []
    for i in existing_indices:
        label = labels[i - 1]  
        temporal_data, label = load_real_data(features_dir, edges_dir, i, label)
        data_list.append((temporal_data, label))  
    
    return DataLoader(data_list, batch_size=batch_size, shuffle=True)

def create_fold_data_loader(features_dir, edges_dir, label_file, indices, batch_size=1):
    labels = load_labels(label_file)
    data_list = []
    for i in indices:
        label = labels[i - 1]  
        temporal_data, label = load_real_data(features_dir, edges_dir, i, label)
        data_list.append((temporal_data, label))
    
    return DataLoader(data_list, batch_size=batch_size, shuffle=True)

def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        for temporal_data, labels in train_loader:
            optimizer.zero_grad()
            output = model(temporal_data)  
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

from sklearn.model_selection import KFold

def cross_validation_train(model_class, train_features_dir, train_edges_dir, label_file, num_epochs=100, k_folds=5):

    labels = load_labels(label_file)
    existing_indices = get_existing_indices(train_features_dir)
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    best_epoch = 0
    best_val_loss = float('inf')

    for fold, (train_idx, val_idx) in enumerate(kf.split(existing_indices)):
        print(f"Fold {fold+1}/{k_folds}")
        
        train_indices = [existing_indices[i] for i in train_idx]
        val_indices = [existing_indices[i] for i in val_idx]
        
        train_loader = create_fold_data_loader(train_features_dir, train_edges_dir, label_file, train_indices)
        val_loader = create_fold_data_loader(train_features_dir, train_edges_dir, label_file, val_indices)
        
        model = model_class(gcn_lstm_model)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, epochs=1)  
            val_loss = evaluate_loss(model, val_loader, criterion)  

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

    print(f"Best Epoch: {best_epoch} with Validation Loss: {best_val_loss:.4f}")
    return best_epoch

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for temporal_data, labels in data_loader:
            output = model(temporal_data)
            probs = torch.sigmoid(output).cpu().numpy()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

    y_true = np.concatenate(all_labels).flatten()
    y_pred_probs = np.concatenate(all_probs).flatten()

    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_true, y_pred_probs >= thr, zero_division=1) for thr in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]

    y_pred = (y_pred_probs >= best_threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    mcc = matthews_corrcoef(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred_probs) if len(np.unique(y_true)) > 1 else float('nan')
    auprc = average_precision_score(y_true, y_pred_probs)
 
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

def evaluate_loss(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for temporal_data, labels in data_loader:
            output = model(temporal_data)
            loss = criterion(output, labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

model = GCN_MLP(gcn_lstm_model)
criterion = nn.BCEWithLogitsLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = create_data_loader(train_features_dir, train_edges_dir, label_file)
test_loader = create_data_loader(test_features_dir, test_edges_dir, label_file)

best_epoch = cross_validation_train(GCN_MLP, train_features_dir, train_edges_dir, label_file, num_epochs=100, k_folds=5)

train_model(model, train_loader, criterion, optimizer, epochs=best_epoch)

torch.save(model.state_dict(), './model.pth')

evaluate_model(model, test_loader)


