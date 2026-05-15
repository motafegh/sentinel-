"""
Graph Builder for SENTINEL Dual-Path Architecture
Converts AST structure into PyTorch Geometric graphs for GNN training
"""

import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Handle both relative and absolute imports
try:
    from .ast_extractor import ContractAST, ASTNode, ASTEdge
except ImportError:
    from ast_extractor import ContractAST, ASTNode, ASTEdge


@dataclass
class GraphBuildConfig:
    """Configuration for graph building"""
    
    # Node type vocabulary
    node_types: List[str] = None
    
    # Visibility vocabulary
    visibility_types: List[str] = None
    
    # Mutability vocabulary
    mutability_types: List[str] = None
    
    # Edge type vocabulary
    edge_types: List[str] = None
    
    def __post_init__(self):
        if self.node_types is None:
            self.node_types = ['function', 'state_variable', 'modifier', 'event']
        
        if self.visibility_types is None:
            self.visibility_types = ['public', 'private', 'internal', 'external']
        
        if self.mutability_types is None:
            self.mutability_types = ['view', 'pure', 'payable', 'nonpayable', 'mutable', 'constant']
        
        if self.edge_types is None:
            self.edge_types = ['CALLS', 'READS', 'WRITES', 'INHERITS', 'MODIFIES']


class GraphBuilder:
    """
    Convert AST structure into PyTorch Geometric Data objects
    
    Node features: [type_onehot(4), visibility_onehot(4), mutability_onehot(6), special_flags(3)]
    Total: 17-dim node features
    
    Edge features: [edge_type_onehot(5)]
    """
    
    def __init__(self, config: Optional[GraphBuildConfig] = None, verbose: bool = False):
        self.config = config or GraphBuildConfig()
        self.verbose = verbose
    
    def build(self, ast: ContractAST) -> Optional[Data]:
        """
        Convert ContractAST to PyTorch Geometric Data object
        
        Args:
            ast: ContractAST object from ASTExtractor
        
        Returns:
            torch_geometric.data.Data object or None if failed
        """
        if not ast.success or len(ast.nodes) == 0:
            if self.verbose:
                print(f"⚠️  Cannot build graph: {ast.error_message or 'No nodes'}")
            return None
        
        try:
            # Build node features
            node_features = self._build_node_features(ast.nodes)
            
            # Build edge index and edge attributes
            edge_index, edge_attr = self._build_edges(ast.edges, len(ast.nodes))
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(ast.nodes),
                contract_name=ast.contract_name,
                contract_path=ast.contract_path,
            )
            
            if self.verbose:
                print(f"✅ Built graph: {data.num_nodes} nodes, {data.num_edges} edges")
            
            return data
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Graph building failed: {e}")
            return None
    
    def _build_node_features(self, nodes: List[ASTNode]) -> torch.Tensor:
        """
        Build node feature matrix
        
        Features per node:
        - Node type one-hot (4 dims)
        - Visibility one-hot (4 dims)
        - Mutability one-hot (6 dims)
        - Special flags: is_constructor, is_fallback, is_receive (3 dims)
        
        Total: 17 dims
        
        Returns:
            [num_nodes, 17] tensor
        """
        num_nodes = len(nodes)
        features = []
        
        for node in nodes:
            # Node type one-hot
            type_onehot = self._one_hot(node.node_type, self.config.node_types)
            
            # Visibility one-hot
            vis_onehot = self._one_hot(node.visibility, self.config.visibility_types)
            
            # Mutability one-hot
            mut_onehot = self._one_hot(node.mutability, self.config.mutability_types)
            
            # Special flags
            special = [
                float(node.is_constructor),
                float(node.is_fallback),
                float(node.is_receive),
            ]
            
            # Concatenate all features
            node_feat = type_onehot + vis_onehot + mut_onehot + special
            features.append(node_feat)
        
        return torch.tensor(features, dtype=torch.float)
    
    def _build_edges(self, edges: List[ASTEdge], num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edge_index and edge_attr tensors
        
        Args:
            edges: List of ASTEdge objects
            num_nodes: Number of nodes in graph
        
        Returns:
            edge_index: [2, num_edges] tensor
            edge_attr: [num_edges, 5] tensor (edge type one-hot)
        """
        if len(edges) == 0:
            # Empty graph - return empty tensors
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(self.config.edge_types)), dtype=torch.float)
            return edge_index, edge_attr
        
        # Build edge lists
        source_nodes = []
        target_nodes = []
        edge_features = []
        
        for edge in edges:
            try:
                source_idx = self._node_id_to_index(edge.source)
                target_idx = self._node_id_to_index(edge.target)
                
                # Validate indices
                if source_idx >= num_nodes or target_idx >= num_nodes:
                    continue
                
                source_nodes.append(source_idx)
                target_nodes.append(target_idx)
                
                # Edge type one-hot
                edge_type_onehot = self._one_hot(edge.edge_type, self.config.edge_types)
                edge_features.append(edge_type_onehot)
                
            except (ValueError, IndexError):
                # Skip edges with invalid node IDs
                continue
        
        if len(source_nodes) == 0:
            # No valid edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(self.config.edge_types)), dtype=torch.float)
            return edge_index, edge_attr
        
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return edge_index, edge_attr
    
    def _node_id_to_index(self, node_id: str) -> int:
        """
        Convert node_id like "func_3" or "var_1" to numeric index
        """
        # Extract the numeric part
        parts = node_id.split('_')
        if len(parts) == 2:
            return int(parts[1])
        else:
            raise ValueError(f"Invalid node_id format: {node_id}")
    
    def _one_hot(self, value: str, vocabulary: List[str]) -> List[float]:
        """
        Create one-hot encoding for a categorical value
        
        Args:
            value: Category value
            vocabulary: List of all possible categories
        
        Returns:
            One-hot encoded list
        """
        one_hot = [0.0] * len(vocabulary)
        
        # Handle case-insensitive matching
        value_lower = value.lower()
        vocab_lower = [v.lower() for v in vocabulary]
        
        if value_lower in vocab_lower:
            idx = vocab_lower.index(value_lower)
            one_hot[idx] = 1.0
        else:
            # Unknown value - all zeros (or could use a special "unknown" category)
            pass
        
        return one_hot


if __name__ == "__main__":
    # Test the graph builder
    from ast_extractor import ASTExtractor
    
    print("="*70)
    print("🧪 TESTING GRAPH BUILDER")
    print("="*70)
    
    # Extract AST
    extractor = ASTExtractor(verbose=True)
    test_contract = "BCCC-SCsVul-2024/SourceCodes/Reentrancy/00001c839d754c4d89b3433aa51e4d6266226a9d907aff96dc019549e86f8289.sol"
    
    print("\n📝 Step 1: Extract AST")
    print("-"*70)
    ast = extractor.extract(test_contract)
    
    if ast.success:
        print(f"✅ AST extracted: {len(ast.nodes)} nodes, {len(ast.edges)} edges")
        print(f"   Node IDs: {[n.node_id for n in ast.nodes]}")
        print(f"   Edge details:")
        for edge in ast.edges:
            print(f"      {edge.source} --{edge.edge_type}--> {edge.target}")
    else:
        print(f"❌ AST extraction failed")
        import sys
        sys.exit(1)
    
    # Build graph
    print("\n🔨 Step 2: Build PyTorch Geometric Graph")
    print("-"*70)
    builder = GraphBuilder(verbose=True)
    graph = builder.build(ast)
    
    if graph is not None:
        print(f"\n✅ GRAPH BUILT SUCCESSFULLY!")
        print(f"   Contract: {graph.contract_name}")
        print(f"   Nodes: {graph.num_nodes}")
        print(f"   Edges: {graph.num_edges}")
        print(f"   Node features shape: {graph.x.shape}")
        print(f"   Edge index shape: {graph.edge_index.shape}")
        print(f"   Edge attr shape: {graph.edge_attr.shape}")
        print(f"\n   Node feature sample (first node):")
        print(f"   {graph.x[0]}")
        print(f"\n   Sample edges: {graph.edge_index[:, :5] if graph.num_edges > 0 else 'No edges'}")
    else:
        print(f"\n❌ GRAPH BUILDING FAILED")
    
    print("="*70)
