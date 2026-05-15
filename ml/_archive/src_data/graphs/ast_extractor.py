"""
ml/_archive/src_data/graphs/ast_extractor.py  
AST Extractor for SENTINEL Dual-Path Architecture
Extracts AST structure from Slither for GNN path
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from slither import Slither
from slither.core.declarations import Contract, Function
import time


@dataclass
class ASTNode:
    """Represents a node in the contract AST graph"""
    node_id: str
    node_type: str  # 'function', 'state_variable', 'modifier', 'event'
    name: str
    visibility: str  # 'public', 'private', 'internal', 'external'
    mutability: str  # 'view', 'pure', 'payable', 'nonpayable'
    is_constructor: bool = False
    is_fallback: bool = False
    is_receive: bool = False
    
    # For embedding later
    code_snippet: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'name': self.name,
            'visibility': self.visibility,
            'mutability': self.mutability,
            'is_constructor': self.is_constructor,
            'is_fallback': self.is_fallback,
            'is_receive': self.is_receive,
        }


@dataclass
class ASTEdge:
    """Represents an edge in the contract AST graph"""
    source: str
    target: str
    edge_type: str  # 'CALLS', 'READS', 'WRITES', 'INHERITS', 'MODIFIES'
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
        }


@dataclass
class ContractAST:
    """Complete AST representation for a contract"""
    contract_path: str
    contract_name: str
    nodes: List[ASTNode] = field(default_factory=list)
    edges: List[ASTEdge] = field(default_factory=list)
    extraction_time: float = 0.0
    solidity_version: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'contract_path': self.contract_path,
            'contract_name': self.contract_name,
            'nodes': [n.to_dict() for n in self.nodes],
            'edges': [e.to_dict() for e in self.edges],
            'extraction_time': self.extraction_time,
            'solidity_version': self.solidity_version,
            'success': self.success,
            'error_message': self.error_message,
        }


class ASTExtractor:
    """
    Extract AST structure from Solidity contracts using Slither
    
    This creates graph data for the GNN path of the dual-path model.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def extract(self, contract_path: str, solidity_version: Optional[str] = None) -> ContractAST:
        """
        Extract AST structure from a Solidity contract
        
        Args:
            contract_path: Path to .sol file
            solidity_version: Optional Solidity version (auto-detected if None)
        
        Returns:
            ContractAST object with nodes and edges
        """
        start_time = time.time()
        contract_path = Path(contract_path)
        
        try:
            # Run Slither to get AST
            if self.verbose:
                print(f"🔍 Extracting AST from {contract_path.name}...")
            
            slither = Slither(str(contract_path))
            
            # Get the main contract (skip dependencies)
            contracts = [c for c in slither.contracts if not c.is_from_dependency()]
            
            if not contracts:
                return ContractAST(
                    contract_path=str(contract_path),
                    contract_name="unknown",
                    success=False,
                    error_message="No contracts found (all from dependencies)",
                    extraction_time=time.time() - start_time
                )
            
            # Use the first non-dependency contract
            contract = contracts[0]
            
            nodes, edges = self._extract_graph_data(contract)
            
            extraction_time = time.time() - start_time
            
            if self.verbose:
                print(f"✅ Extracted {len(nodes)} nodes, {len(edges)} edges in {extraction_time:.2f}s")
            
            return ContractAST(
                contract_path=str(contract_path),
                contract_name=contract.name,
                nodes=nodes,
                edges=edges,
                extraction_time=extraction_time,
                solidity_version=solidity_version,
                success=True
            )
            
        except Exception as e:
            extraction_time = time.time() - start_time
            if self.verbose:
                print(f"❌ AST extraction failed: {e}")
            
            return ContractAST(
                contract_path=str(contract_path),
                contract_name="unknown",
                success=False,
                error_message=str(e),
                extraction_time=extraction_time
            )
    
    def _extract_graph_data(self, contract: Contract) -> Tuple[List[ASTNode], List[ASTEdge]]:
        """
        Extract nodes and edges from Slither Contract object
        
        Returns:
            (nodes, edges) tuple
        """
        nodes = []
        edges = []
        
        # Create node ID mappings
        func_to_id = {}
        var_to_id = {}
        
        # Extract state variables as nodes
        for i, var in enumerate(contract.state_variables):
            node_id = f"var_{i}"
            var_to_id[var.name] = node_id
            
            nodes.append(ASTNode(
                node_id=node_id,
                node_type='state_variable',
                name=var.name,
                visibility=str(var.visibility),
                mutability='constant' if var.is_constant else 'mutable',
            ))
        
        # Extract functions as nodes
        for i, func in enumerate(contract.functions):
            node_id = f"func_{i}"
            func_to_id[func.canonical_name] = node_id
            
            nodes.append(ASTNode(
                node_id=node_id,
                node_type='function',
                name=func.name,
                visibility=str(func.visibility),
                mutability=str(func.view) if hasattr(func, 'view') else 'nonpayable',
                is_constructor=func.is_constructor,
                is_fallback=func.is_fallback,
                is_receive=func.is_receive if hasattr(func, 'is_receive') else False,
            ))
            
            # Extract edges: function CALLS function
            for called_func in func.internal_calls:
                if hasattr(called_func, 'canonical_name'):
                    target_id = func_to_id.get(called_func.canonical_name)
                    if target_id:
                        edges.append(ASTEdge(
                            source=node_id,
                            target=target_id,
                            edge_type='CALLS'
                        ))
            
            # Extract edges: function READS state variable
            for var in func.state_variables_read:
                target_id = var_to_id.get(var.name)
                if target_id:
                    edges.append(ASTEdge(
                        source=node_id,
                        target=target_id,
                        edge_type='READS'
                    ))
            
            # Extract edges: function WRITES state variable
            for var in func.state_variables_written:
                target_id = var_to_id.get(var.name)
                if target_id:
                    edges.append(ASTEdge(
                        source=node_id,
                        target=target_id,
                        edge_type='WRITES'
                    ))
        
        return nodes, edges


if __name__ == "__main__":
    # Quick test
    extractor = ASTExtractor(verbose=True)
    
    test_contract = "BCCC-SCsVul-2024/SourceCodes/Reentrancy/00001c839d754c4d89b3433aa51e4d6266226a9d907aff96dc019549e86f8289.sol"
    
    ast = extractor.extract(test_contract)
    
    if ast.success:
        print(f"\n✅ Contract: {ast.contract_name}")
        print(f"   Nodes: {len(ast.nodes)}")
        print(f"   Edges: {len(ast.edges)}")
        print(f"\n   Node types: {[n.node_type for n in ast.nodes[:5]]}")
        print(f"   Edge types: {[e.edge_type for e in ast.edges[:5]]}")
    else:
        print(f"\n❌ Failed: {ast.error_message}")
