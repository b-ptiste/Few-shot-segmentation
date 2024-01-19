import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MFConv, GATv2Conv, SuperGATConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from transformers import AutoModel

class MLPModel(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super(MLPModel, self).__init__()
        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )
        self.mol_hidden1 = nn.Linear(num_node_features, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        x = self.relu(self.mol_hidden1(x))
        x = self.relu(self.mol_hidden2(x))
        x = self.mol_hidden3(x)
        x = self.ln(x)
        x = x * torch.exp(self.temp)
        x = global_max_pool(x, batch)
        return x
    
class GCNModel(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GCNModel, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )
        self.ln1 = nn.LayerNorm((nout))
        self.relu = nn.ReLU()
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x).relu()
        x = self.mol_hidden3(x)
        x = self.ln1(x)
        x = x * torch.exp(self.temp)
        return x

class GatConv(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GatConv, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GATConv(num_node_features, graph_hidden_channels, heads=heads)
        self.skip_1 = nn.Linear(num_node_features, graph_hidden_channels * heads)
        self.conv2 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_2 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)
        self.conv3 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_3 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)
        self.conv4 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_4 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels * heads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x1 = self.conv1(x, edge_index)
        skip_x = self.skip_1(x)  # Prepare skip connection
        x = skip_x + x1  # Apply skip connection
        x = self.relu(x)
        
        x2 = self.conv2(x, edge_index)
        skip_x = self.skip_2(x)  # Prepare skip connection
        x = skip_x + x2  # Apply skip connection
        x = self.relu(x)
        
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x)  # Prepare skip connection
        x = skip_x + x3  # Apply skip connection
        x = self.relu(x)
        
        x4 = self.conv4(x, edge_index)
        skip_x = self.skip_4(x)  # Prepare skip connection
        x = skip_x + x4  # Apply skip connection
        x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x(nn.Module)

class MoMuGNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, nout, JK = "last", drop_ratio = 0, num_node_features=0):
        super(MoMuGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.output_dim = nout
        self.num_node_features = num_node_features

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnns = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(torch.nn.Sequential(torch.nn.Linear(num_node_features, 2*num_node_features), torch.nn.ReLU(), torch.nn.Linear(2*num_node_features, num_node_features)), aggr = "add"))
            self.batch_norms.append(torch.nn.BatchNorm1d(num_node_features))

    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio)
            h_list.append(h)

        node_representation = h_list[-1]    
        node_counts = batch.bincount()
        node_representation_list = []

        for graph_idx in range(len(node_counts)):
            node_representation_list.append(node_representation[batch == graph_idx])
        node_representation_padded = torch.nn.utils.rnn.pad_sequence(node_representation_list, batch_first=True)
        node_representation_padded = global_max_pool(node_representation, batch)
        return node_representation_padded

class GINConModel(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super().__init__()
        
        self.graph2d_encoder = self.graph_encoder = MoMuGNN(
            num_layer=2, #TODO: can change nb of GIN layers
            nout=nout,
            drop_ratio=0,
            JK='last', 
            num_node_features=num_node_features
        )
    
        self.num_features = num_node_features
        self.nout = nout
        self.fc_hidden = nn.Linear(self.num_features, self.nout)
    
    def forward(self, graph_batch):
        node_feats = self.graph2d_encoder(graph_batch)
        node_feats = self.fc_hidden(node_feats)
        return node_feats

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        # Define a linear layer to learn the attention weights
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, encoded_states):
        # Compute attention scores
        # encoded_states shape: (batch_size, sequence_length, hidden_dim)
        attention_scores = self.attention_weights(encoded_states)
        
        # Apply softmax to get probabilities (shape: batch_size, sequence_length, 1)
        attention_probs = F.softmax(attention_scores, dim=1)

        # Multiply each hidden state with the attention weights and sum them
        # Use torch.bmm for batch matrix multiplication
        pooled_output = torch.bmm(torch.transpose(encoded_states, 1, 2), attention_probs).squeeze(2)
        return pooled_output
    
class TextEncoder(nn.Module):
    def __init__(self, model_name, hidden_dim):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        # self.attentionpooling = AttentionPooling(hidden_dim)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        # pooled_output = self.attentionpooling(encoded_text.last_hidden_state) 
        # return pooled_output   
        return encoded_text.last_hidden_state[:,0,:]
    
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(Model, self).__init__()
        #self.graph_encoder = GINConModel(num_node_features, nout, nhid)
        #self.graph_encoder = MLPModel(num_node_features, nout, nhid)
        self.graph_encoder = GatConv(num_node_features, nout, nhid, graph_hidden_channels, heads)
        self.text_encoder = TextEncoder(model_name, nout)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
