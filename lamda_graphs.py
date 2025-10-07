#!/usr/bin/env python3
"""
Lambda Calculus Graph Transformation System
===========================================

A unified system that combines lambda calculus with graph rewriting,
enabling functional programming approaches to graph transformations.

Author: Lambda Graph Transformation Team
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass
from copy import deepcopy
from collections import defaultdict


# ============================================================================
# LAMBDA CALCULUS CORE
# ============================================================================

class Expression(ABC):
    """Base class for all lambda calculus expressions."""

    @abstractmethod
    def free_vars(self) -> Set[str]:
        """Return free variables in this expression."""
        pass

    @abstractmethod
    def substitute(self, var: str, value: 'Expression') -> 'Expression':
        """Substitute variable with value."""
        pass

    @abstractmethod
    def alpha_equiv(self, other: 'Expression') -> bool:
        """Check alpha equivalence."""
        pass


class Variable(Expression):
    """Variable in lambda calculus."""

    def __init__(self, name: str):
        self.name = name

    def free_vars(self) -> Set[str]:
        return {self.name}

    def substitute(self, var: str, value: Expression) -> Expression:
        if self.name == var:
            return deepcopy(value)
        return self

    def alpha_equiv(self, other: Expression) -> bool:
        return isinstance(other, Variable) and self.name == other.name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, Variable) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class Abstraction(Expression):
    """Lambda abstraction: λvar. body"""

    def __init__(self, var: str, body: Expression):
        self.var = var
        self.body = body

    def free_vars(self) -> Set[str]:
        return self.body.free_vars() - {self.var}

    def substitute(self, var: str, value: Expression) -> Expression:
        if var == self.var:
            return self
        elif var in value.free_vars() and self.var in value.free_vars():
            fresh_var = self._fresh_var(value.free_vars() | {var})
            new_body = self.body.substitute(self.var, Variable(fresh_var))
            return Abstraction(fresh_var, new_body.substitute(var, value))
        else:
            return Abstraction(self.var, self.body.substitute(var, value))

    def _fresh_var(self, avoid: Set[str]) -> str:
        """Generate fresh variable name."""
        base = self.var
        counter = 1
        while f"{base}_{counter}" in avoid:
            counter += 1
        return f"{base}_{counter}"

    def alpha_equiv(self, other: Expression) -> bool:
        if not isinstance(other, Abstraction):
            return False
        fresh = self._fresh_var(self.free_vars() | other.free_vars())
        self_body = self.body.substitute(self.var, Variable(fresh))
        other_body = other.body.substitute(other.var, Variable(fresh))
        return self_body.alpha_equiv(other_body)

    def __repr__(self) -> str:
        return f"(λ{self.var}.{self.body})"


class Application(Expression):
    """Function application: (func arg)"""

    def __init__(self, func: Expression, arg: Expression):
        self.func = func
        self.arg = arg

    def free_vars(self) -> Set[str]:
        return self.func.free_vars() | self.arg.free_vars()

    def substitute(self, var: str, value: Expression) -> Expression:
        return Application(
            self.func.substitute(var, value),
            self.arg.substitute(var, value)
        )

    def alpha_equiv(self, other: Expression) -> bool:
        return (isinstance(other, Application) and
                self.func.alpha_equiv(other.func) and
                self.arg.alpha_equiv(other.arg))

    def __repr__(self) -> str:
        return f"({self.func} {self.arg})"


# ============================================================================
# LAMBDA CALCULUS EVALUATION ENGINE
# ============================================================================

class LambdaEngine:
    """Enhanced lambda calculus evaluation engine."""

    def __init__(self, max_steps: int = 1000):
        self.max_steps = max_steps

    def beta_reduce(self, expr: Expression) -> Optional[Expression]:
        """Single step beta reduction."""
        if isinstance(expr, Application):
            func, arg = expr.func, expr.arg

            if isinstance(func, Abstraction):
                return func.body.substitute(func.var, arg)
            else:
                reduced_func = self.beta_reduce(func)
                if reduced_func:
                    return Application(reduced_func, arg)

                reduced_arg = self.beta_reduce(arg)
                if reduced_arg:
                    return Application(func, reduced_arg)

        elif isinstance(expr, Abstraction):
            reduced_body = self.beta_reduce(expr.body)
            if reduced_body:
                return Abstraction(expr.var, reduced_body)

        return None

    def evaluate(self, expr: Expression) -> Expression:
        """Evaluate to normal form."""
        current = expr
        for _ in range(self.max_steps):
            next_expr = self.beta_reduce(current)
            if next_expr is None:
                break
            current = next_expr
        return current

    def apply(self, func: Expression, arg: Expression) -> Expression:
        """Apply function to argument and evaluate."""
        return self.evaluate(Application(func, arg))


# ============================================================================
# ENHANCED GRAPH REPRESENTATION
# ============================================================================

@dataclass
class NodeData:
    """Rich node data structure."""
    id: str
    type: str
    attributes: Dict[str, Any]

    def matches(self, pattern: Dict[str, Any]) -> bool:
        """Check if node matches pattern."""
        for key, value in pattern.items():
            if key == "type" and self.type != value:
                return False
            elif key in self.attributes and self.attributes[key] != value:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "attributes": self.attributes
        }


@dataclass
class EdgeData:
    """Rich edge data structure."""
    source: str
    target: str
    label: str
    attributes: Dict[str, Any]

    def matches(self, pattern: Dict[str, Any]) -> bool:
        """Check if edge matches pattern."""
        if "label" in pattern and self.label != pattern["label"]:
            return False
        for key, value in pattern.items():
            if key != "label" and key in self.attributes and self.attributes[key] != value:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "attributes": self.attributes
        }


class EnhancedGraph:
    """Enhanced graph with rich metadata and operations."""

    def __init__(self):
        self.nodes: Dict[str, NodeData] = {}
        self.edges: List[EdgeData] = []
        self._adjacency: Dict[str, List[EdgeData]] = defaultdict(list)

    def add_node(self, node_id: str, node_type: str = "default", **attrs):
        """Add node with metadata."""
        self.nodes[node_id] = NodeData(node_id, node_type, attrs)

    def add_edge(self, source: str, target: str, label: str = "default", **attrs):
        """Add edge with metadata."""
        edge = EdgeData(source, target, label, attrs)
        self.edges.append(edge)
        self._adjacency[source].append(edge)

    def get_node(self, node_id: str) -> Optional[NodeData]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_edges_from(self, node_id: str) -> List[EdgeData]:
        """Get all edges from a node."""
        return self._adjacency.get(node_id, [])

    def find_nodes(self, pattern: Dict[str, Any]) -> List[NodeData]:
        """Find nodes matching pattern."""
        return [node for node in self.nodes.values() if node.matches(pattern)]

    def find_edges(self, pattern: Dict[str, Any]) -> List[EdgeData]:
        """Find edges matching pattern."""
        return [edge for edge in self.edges if edge.matches(pattern)]

    def remove_node(self, node_id: str):
        """Remove node and all its edges."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]
            if node_id in self._adjacency:
                del self._adjacency[node_id]
            for adj_list in self._adjacency.values():
                adj_list[:] = [e for e in adj_list if e.target != node_id]

    def remove_edge(self, source: str, target: str, label: str = None):
        """Remove specific edge."""
        def should_remove(edge):
            return (edge.source == source and edge.target == target and
                   (label is None or edge.label == label))

        self.edges[:] = [e for e in self.edges if not should_remove(e)]
        if source in self._adjacency:
            self._adjacency[source][:] = [e for e in self._adjacency[source] if not should_remove(e)]

    def copy(self) -> 'EnhancedGraph':
        """Create deep copy of graph."""
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges]
        }

    def to_lambda_expr(self) -> Expression:
        """Convert graph structure to lambda calculus expression."""
        if not self.nodes:
            return Variable("EmptyGraph")
        
        expr = Variable("Graph")
        for node_id in sorted(self.nodes.keys()):
            expr = Application(expr, Variable(f"Node_{node_id}"))
        
        return expr

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedGraph':
        """Create graph from dictionary representation."""
        graph = cls()
        for node_data in data.get("nodes", []):
            graph.add_node(
                node_data["id"],
                node_data.get("type", "default"),
                **node_data.get("attributes", {})
            )
        for edge_data in data.get("edges", []):
            graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                edge_data.get("label", "default"),
                **edge_data.get("attributes", {})
            )
        return graph

    def __repr__(self) -> str:
        return f"Graph(nodes={len(self.nodes)}, edges={len(self.edges)})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, EnhancedGraph):
            return False
        return (set(self.nodes.keys()) == set(other.nodes.keys()) and
                len(self.edges) == len(other.edges))


# ============================================================================
# PATTERN MATCHING AND UNIFICATION
# ============================================================================

@dataclass
class GraphPattern:
    """Graph pattern for matching."""
    node_patterns: Dict[str, Dict[str, Any]]
    edge_patterns: List[Tuple[str, str, Dict[str, Any]]]
    constraints: List[Callable[[Dict[str, str]], bool]]

    def to_lambda_expr(self) -> Expression:
        """Convert pattern to lambda calculus expression."""
        body = Variable("PATTERN")
        for var in sorted(self.node_patterns.keys()):
            body = Abstraction(var, body)
        return body

    def __repr__(self) -> str:
        return f"Pattern(nodes={self.node_patterns}, edges={self.edge_patterns})"


class PatternMatcher:
    """Advanced pattern matching with unification."""

    def __init__(self):
        self.engine = LambdaEngine()

    def find_matches(self, graph: EnhancedGraph, pattern: GraphPattern) -> List[Dict[str, str]]:
        """Find all matches of pattern in graph."""
        matches = []

        candidates = {}
        for var, node_pattern in pattern.node_patterns.items():
            candidates[var] = [n.id for n in graph.find_nodes(node_pattern)]

        for assignment in self._generate_assignments(candidates):
            if self._check_assignment(graph, pattern, assignment):
                matches.append(assignment)

        return matches

    def _generate_assignments(self, candidates: Dict[str, List[str]]):
        """Generate all possible variable assignments."""
        if not candidates:
            yield {}
            return

        var, values = next(iter(candidates.items()))
        rest = {k: v for k, v in candidates.items() if k != var}

        for value in values:
            for rest_assignment in self._generate_assignments(rest):
                if value not in rest_assignment.values():
                    yield {var: value, **rest_assignment}

    def _check_assignment(self, graph: EnhancedGraph, pattern: GraphPattern, assignment: Dict[str, str]) -> bool:
        """Check if assignment satisfies all pattern constraints."""
        for src_var, tgt_var, edge_pattern in pattern.edge_patterns:
            src_id = assignment[src_var]
            tgt_id = assignment[tgt_var]

            found_edge = False
            for edge in graph.get_edges_from(src_id):
                if edge.target == tgt_id and edge.matches(edge_pattern):
                    found_edge = True
                    break

            if not found_edge:
                return False

        for constraint in pattern.constraints:
            if not constraint(assignment):
                return False

        return True


# ============================================================================
# LAMBDA CALCULUS GRAPH TRANSFORMATION RULES
# ============================================================================

class GraphTransformation(Expression):
    """Graph transformation as lambda calculus expression."""

    def __init__(self, pattern: GraphPattern, transform_fn: Callable[[EnhancedGraph, Dict[str, str]], EnhancedGraph]):
        self.pattern = pattern
        self.transform_fn = transform_fn

    def free_vars(self) -> Set[str]:
        return set(self.pattern.node_patterns.keys())

    def substitute(self, var: str, value: Expression) -> Expression:
        return self

    def alpha_equiv(self, other: Expression) -> bool:
        return isinstance(other, GraphTransformation) and str(self.pattern) == str(other.pattern)

    def apply_to(self, graph: EnhancedGraph) -> List[EnhancedGraph]:
        """Apply transformation to graph, returning all possible results."""
        matcher = PatternMatcher()
        matches = matcher.find_matches(graph, self.pattern)

        results = []
        for match in matches:
            new_graph = graph.copy()
            transformed = self.transform_fn(new_graph, match)
            results.append(transformed)

        return results

    def to_lambda_expr(self) -> Expression:
        """Convert to pure lambda calculus representation."""
        pattern_expr = self.pattern.to_lambda_expr()
        return Abstraction("graph", Application(pattern_expr, Variable("graph")))

    def __repr__(self) -> str:
        return f"GraphTransform({self.pattern})"


class TransformationRule:
    """High-level transformation rule combining pattern and action."""

    def __init__(self, name: str):
        self.name = name
        self.lambda_expr: Optional[Expression] = None

    def pattern(self, node_patterns: Dict[str, Dict[str, Any]],
               edge_patterns: List[Tuple[str, str, Dict[str, Any]]] = None,
               constraints: List[Callable] = None) -> 'TransformationRule':
        """Define the pattern to match."""
        self.graph_pattern = GraphPattern(
            node_patterns,
            edge_patterns or [],
            constraints or []
        )
        return self

    def transform(self, transform_fn: Callable[[EnhancedGraph, Dict[str, str]], EnhancedGraph]) -> 'TransformationRule':
        """Define the transformation function."""
        self.graph_transform = GraphTransformation(self.graph_pattern, transform_fn)
        self.lambda_expr = self._create_lambda_expr()
        return self

    def _create_lambda_expr(self) -> Expression:
        """Create lambda calculus representation of the rule."""
        body = Variable("TRANSFORM")
        for var in sorted(self.graph_pattern.node_patterns.keys()):
            body = Abstraction(var, body)
        return body

    def apply(self, graph: EnhancedGraph) -> List[EnhancedGraph]:
        """Apply rule to graph."""
        if hasattr(self, 'graph_transform'):
            return self.graph_transform.apply_to(graph)
        return [graph]

    def compose_with(self, other: 'TransformationRule') -> 'TransformationRule':
        """Compose two transformation rules."""
        def composed_transform(graph, match):
            intermediate_results = self.apply(graph)
            final_results = []
            for intermediate in intermediate_results:
                final_results.extend(other.apply(intermediate))
            return final_results[0] if final_results else graph

        new_rule = TransformationRule(f"{self.name}∘{other.name}")
        new_rule.graph_pattern = self.graph_pattern
        new_rule.graph_transform = GraphTransformation(new_rule.graph_pattern, composed_transform)
        new_rule.lambda_expr = self._create_composed_lambda_expr(other)
        return new_rule

    def _create_composed_lambda_expr(self, other: 'TransformationRule') -> Expression:
        """Create lambda expression for composed rule."""
        if self.lambda_expr and other.lambda_expr:
            return Abstraction("g", 
                Application(other.lambda_expr, 
                    Application(self.lambda_expr, Variable("g"))))
        return Variable("COMPOSED_TRANSFORM")

    def __repr__(self) -> str:
        return f"Rule({self.name}: {self.lambda_expr})"


# ============================================================================
# TRANSFORMATION SYSTEM
# ============================================================================

class GraphTransformationSystem:
    """System for managing and applying graph transformation rules."""

    def __init__(self):
        self.rules: List[TransformationRule] = []
        self.engine = LambdaEngine()

    def add_rule(self, rule: TransformationRule):
        """Add transformation rule to system."""
        self.rules.append(rule)

    def apply_rule(self, rule_name: str, graph: EnhancedGraph) -> List[EnhancedGraph]:
        """Apply specific rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule.apply(graph)
        raise ValueError(f"Rule '{rule_name}' not found")

    def apply_all_rules(self, graph: EnhancedGraph) -> List[EnhancedGraph]:
        """Apply all applicable rules to graph."""
        results = [graph]
        for rule in self.rules:
            new_results = []
            for g in results:
                new_results.extend(rule.apply(g))
            if new_results:
                results.extend(new_results)
        return results

    def transform_iteratively(self, graph: EnhancedGraph, max_iterations: int = 10) -> EnhancedGraph:
        """Apply rules iteratively until fixpoint."""
        current = graph
        for i in range(max_iterations):
            results = self.apply_all_rules(current)
            if not results or all(r == current for r in results):
                break
            current = results[0]
        return current

    def get_lambda_representation(self) -> Expression:
        """Get lambda calculus representation of the entire system."""
        if not self.rules:
            return Variable("EmptySystem")
        
        combined = self.rules[0].lambda_expr or Variable("Rule1")
        for i, rule in enumerate(self.rules[1:], 2):
            rule_expr = rule.lambda_expr or Variable(f"Rule{i}")
            combined = Application(combined, rule_expr)
        
        return combined


# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

def create_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> EnhancedGraph:
    """
    Create a graph from node and edge specifications.
    
    Args:
        nodes: List of dicts with 'id', 'type', and 'attributes' keys
        edges: List of dicts with 'source', 'target', 'label', and 'attributes' keys
    
    Returns:
        EnhancedGraph instance
    """
    graph = EnhancedGraph()
    
    for node in nodes:
        graph.add_node(
            node["id"],
            node.get("type", "default"),
            **node.get("attributes", {})
        )
    
    for edge in edges:
        graph.add_edge(
            edge["source"],
            edge["target"],
            edge.get("label", "default"),
            **edge.get("attributes", {})
        )
    
    return graph


def create_transformation_rule(
    name: str,
    node_patterns: Dict[str, Dict[str, Any]],
    edge_patterns: List[Tuple[str, str, Dict[str, Any]]],
    transform_fn: Callable[[EnhancedGraph, Dict[str, str]], EnhancedGraph],
    constraints: Optional[List[Callable]] = None
) -> TransformationRule:
    """
    Create a transformation rule.
    
    Args:
        name: Rule name
        node_patterns: Pattern for matching nodes
        edge_patterns: Pattern for matching edges
        transform_fn: Function to transform matched subgraph
        constraints: Additional matching constraints
    
    Returns:
        TransformationRule instance
    """
    return (TransformationRule(name)
            .pattern(node_patterns, edge_patterns, constraints or [])
            .transform(transform_fn))


def apply_transformations(
    graph: EnhancedGraph,
    rules: List[TransformationRule],
    mode: str = "all",
    max_iterations: int = 10
) -> Union[EnhancedGraph, List[EnhancedGraph]]:
    """
    Apply transformation rules to a graph.
    
    Args:
        graph: Input graph
        rules: List of transformation rules
        mode: 'all' (apply all rules once), 'iterative' (apply until fixpoint),
              'first' (apply first matching rule)
        max_iterations: Maximum iterations for iterative mode
    
    Returns:
        Transformed graph(s)
    """
    system = GraphTransformationSystem()
    for rule in rules:
        system.add_rule(rule)
    
    if mode == "iterative":
        return system.transform_iteratively(graph, max_iterations)
    elif mode == "all":
        results = system.apply_all_rules(graph)
        return results if results else [graph]
    elif mode == "first":
        for rule in rules:
            results = rule.apply(graph)
            if results:
                return results[0]
        return graph
    else:
        raise ValueError(f"Unknown mode: {mode}")


def evaluate_lambda_expression(expr: Expression, max_steps: int = 1000) -> Expression:
    """
    Evaluate a lambda calculus expression to normal form.
    
    Args:
        expr: Lambda expression
        max_steps: Maximum reduction steps
    
    Returns:
        Evaluated expression
    """
    engine = LambdaEngine(max_steps)
    return engine.evaluate(expr)


def match_pattern(graph: EnhancedGraph, pattern: GraphPattern) -> List[Dict[str, str]]:
    """
    Find all matches of a pattern in a graph.
    
    Args:
        graph: Graph to search
        pattern: Pattern to match
    
    Returns:
        List of variable assignments for each match
    """
    matcher = PatternMatcher()
    return matcher.find_matches(graph, pattern)


def graph_to_dict(graph: EnhancedGraph) -> Dict[str, Any]:
    """Convert graph to dictionary representation."""
    return graph.to_dict()


def dict_to_graph(data: Dict[str, Any]) -> EnhancedGraph:
    """Create graph from dictionary representation."""
    return EnhancedGraph.from_dict(data)


def compose_rules(rule1: TransformationRule, rule2: TransformationRule) -> TransformationRule:
    """Compose two transformation rules."""
    return rule1.compose_with(rule2)


def graph_to_lambda(graph: EnhancedGraph) -> Expression:
    """Convert graph to lambda calculus expression."""
    return graph.to_lambda_expr()


__all__ = [
    'Expression', 'Variable', 'Abstraction', 'Application',
    'LambdaEngine', 'EnhancedGraph', 'NodeData', 'EdgeData',
    'GraphPattern', 'PatternMatcher', 'GraphTransformation',
    'TransformationRule', 'GraphTransformationSystem',
    'create_graph', 'create_transformation_rule', 'apply_transformations',
    'evaluate_lambda_expression', 'match_pattern', 'graph_to_dict',
    'dict_to_graph', 'compose_rules', 'graph_to_lambda'
]
