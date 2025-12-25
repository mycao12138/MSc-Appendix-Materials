import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

def get_time_decayed_subgraph(data, node_idx, num_hops=2, decay_rate=0.1):
    """
    Generates a Time-Decayed Induced Subgraph for a target node.
    
    Implements the innovation described in Methodology Sec 3.3:
    1. Extract r-hop ego-network.
    2. Compute edge weights via exponential temporal decay.
    
    Args:
        data (Data): The entire DGraphFin graph object.
        node_idx (int): Index of the target user node.
        num_hops (int): Radius of the subgraph (r). Default: 2.
        decay_rate (float): Lambda parameter for time decay. Default: 0.1.
        
    Returns:
        sub_data (Data): Induced subgraph with 'edge_weight' attribute.
    """
    
    # 1. Extract k-hop neighbors and edges
    # subset: global indices of nodes in subgraph
    # edge_index: re-indexed edges for the subgraph
    # mapping: index of the target node inside the subgraph
    # edge_mask: boolean mask indicating which global edges are included
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, 
        num_hops, 
        data.edge_index, 
        relabel_nodes=True, 
        num_nodes=data.num_nodes
    )
    
    # 2. Retrieve node features and labels for the subset
    x = data.x[subset]
    y = data.y[subset]
    
    # 3. Retrieve temporal attributes (Relative Time Delta)
    # data.edge_attr was pre-processed to be relative time (0 to 1) in dataset.py
    if data.edge_attr is not None:
        edge_time_delta = data.edge_attr[edge_mask] # Shape: [E_sub, 1]
        
        # 4. Apply Temporal Decay Formula (The Innovation)
        # Weight = exp(-lambda * delta_t)
        # Flatten to [E_sub] for compatibility with GNN conv layers
        edge_weight = torch.exp(-decay_rate * edge_time_delta).squeeze()
    else:
        # Fallback for static graphs (all weights = 1.0)
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    # 5. Construct the subgraph Data object
    sub_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight, # Pass weights as edge_attr for GAT/GCN
        y=y
    )
    
    # Track which node is the target (useful for pooling/readout)
    # Create a mask where only the target node is True
    target_mask = torch.zeros(subset.size(0), dtype=torch.bool)
    target_mask[mapping] = True
    sub_data.target_mask = target_mask
    
    return sub_data

# ==================================================================
# Sanity Check
# ==================================================================
if __name__ == "__main__":
    # Mock data to test logic
    print("Testing induction logic...")
    
    # Create a dummy graph: 0->1, 1->2, with timestamps
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(3, 16) # 3 nodes, 16 feats
    y = torch.tensor([0, 0, 1])
    
    # edge_attr represents relative time delta (0.0 = recent, 1.0 = old)
    # Edge 0-1 is recent (0.1), Edge 1-2 is old (0.9)
    edge_attr = torch.tensor([[0.1], [0.1], [0.9], [0.9]]) 
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=3)
    
    # Generate subgraph for Node 0
    lambda_val = 1.0
    subgraph = get_time_decayed_subgraph(data, node_idx=0, num_hops=2, decay_rate=lambda_val)
    
    print(f"Subgraph Nodes: {subgraph.num_nodes}")
    print(f"Subgraph Edges: {subgraph.num_edges}")
    
    # Check weights: exp(-1.0 * 0.1) approx 0.904, exp(-1.0 * 0.9) approx 0.406
    print(f"Calculated Weights: {subgraph.edge_attr}")
    
    expected_recent = torch.exp(torch.tensor(-1.0 * 0.1))
    print(f"Expected Recent Weight: {expected_recent:.4f}")