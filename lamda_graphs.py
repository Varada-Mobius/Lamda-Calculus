#!/usr/bin/env python3
"""
Lambda Calculus Graph Transformation System
===========================================

A unified system that combines lambda calculus with graph rewriting,
enabling functional programming approaches to graph transformations.

Usage:
    python main.py

Author: Lambda Graph Transformation Team
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass
from copy import deepcopy
import json
from collections import defaultdict
import sys


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
            return self  # Variable is bound
        elif var in value.free_vars() and self.var in value.free_vars():
            # Alpha convert to avoid capture
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
                # Beta redex: (λx. body) arg → body[x := arg]
                return func.body.substitute(func.var, arg)
            else:
                # Try reducing function first
                reduced_func = self.beta_reduce(func)
                if reduced_func:
                    return Application(reduced_func, arg)

                # Then try argument
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
        steps = 0
        for steps in range(self.max_steps):
            next_expr = self.beta_reduce(current)
            if next_expr is None:
                break
            current = next_expr
        
        print(f"Lambda evaluation completed in {steps} steps")
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
            # Remove edges
            self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]
            # Update adjacency
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
        # Create a lambda expression representing the graph structure
        # This is a symbolic representation
        if not self.nodes:
            return Variable("EmptyGraph")
        
        # Create nested abstraction for each node
        expr = Variable("Graph")
        for node_id in sorted(self.nodes.keys()):
            expr = Application(expr, Variable(f"Node_{node_id}"))
        
        return expr

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
    node_patterns: Dict[str, Dict[str, Any]]  # var_name -> pattern
    edge_patterns: List[Tuple[str, str, Dict[str, Any]]]  # (src_var, tgt_var, pattern)
    constraints: List[Callable[[Dict[str, str]], bool]]  # Additional constraints

    def to_lambda_expr(self) -> Expression:
        """Convert pattern to lambda calculus expression."""
        # Create lambda abstractions for pattern variables
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

        # Get candidate nodes for each pattern variable
        candidates = {}
        for var, node_pattern in pattern.node_patterns.items():
            candidates[var] = [n.id for n in graph.find_nodes(node_pattern)]

        # Generate all possible variable assignments
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
                if value not in rest_assignment.values():  # No duplicate assignments
                    yield {var: value, **rest_assignment}

    def _check_assignment(self, graph: EnhancedGraph, pattern: GraphPattern, assignment: Dict[str, str]) -> bool:
        """Check if assignment satisfies all pattern constraints."""
        # Check edge patterns
        for src_var, tgt_var, edge_pattern in pattern.edge_patterns:
            src_id = assignment[src_var]
            tgt_id = assignment[tgt_var]

            # Find matching edge
            found_edge = False
            for edge in graph.get_edges_from(src_id):
                if edge.target == tgt_id and edge.matches(edge_pattern):
                    found_edge = True
                    break

            if not found_edge:
                return False

        # Check additional constraints
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
        # For graph transformations, substitution is more complex
        # This is a simplified version
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
        # Create lambda abstraction representing the transformation
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
        # Create nested lambda abstractions for all pattern variables
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

        # Create new rule with combined pattern (simplified)
        new_rule = TransformationRule(f"{self.name}∘{other.name}")
        new_rule.graph_pattern = self.graph_pattern  # Simplified
        new_rule.graph_transform = GraphTransformation(new_rule.graph_pattern, composed_transform)
        new_rule.lambda_expr = self._create_composed_lambda_expr(other)
        return new_rule

    def _create_composed_lambda_expr(self, other: 'TransformationRule') -> Expression:
        """Create lambda expression for composed rule."""
        # λg. other_expr (self_expr g)
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
                print(f"Reached fixpoint after {i} iterations")
                break
            current = results[0]
        return current

    def get_lambda_representation(self) -> Expression:
        """Get lambda calculus representation of the entire system."""
        if not self.rules:
            return Variable("EmptySystem")
        
        # Compose all rules into a single lambda expression
        combined = self.rules[0].lambda_expr or Variable("Rule1")
        for i, rule in enumerate(self.rules[1:], 2):
            rule_expr = rule.lambda_expr or Variable(f"Rule{i}")
            combined = Application(combined, rule_expr)
        
        return combined


# ============================================================================
# EXAMPLES AND DEMO FUNCTIONS
# ============================================================================

def create_sample_graph() -> EnhancedGraph:
    """Create a sample graph for demonstration."""
    graph = EnhancedGraph()
    
    # Add nodes with different types and attributes
    graph.add_node("A", "person", name="Alice", age=30)
    graph.add_node("B", "person", name="Bob", age=25)
    graph.add_node("C", "company", name="TechCorp", industry="software")
    graph.add_node("D", "project", name="WebApp", status="active")
    
    # Add edges with labels and attributes
    graph.add_edge("A", "C", "works_at", role="manager", years=3)
    graph.add_edge("B", "C", "works_at", role="developer", years=1)
    graph.add_edge("A", "D", "manages", responsibility="full")
    graph.add_edge("B", "D", "works_on", hours_per_week=40)
    
    return graph


def create_sample_rules() -> List[TransformationRule]:
    """Create sample transformation rules."""
    rules = []
    
    # Rule 1: Promote developer to senior if they work on active projects
    def promote_developer(graph: EnhancedGraph, match: Dict[str, str]) -> EnhancedGraph:
        person_id = match["person"]
        person = graph.get_node(person_id)
        if person and "role" not in person.attributes:
            person.attributes["role"] = "senior_developer"
        return graph
    
    promote_rule = (TransformationRule("promote_developer")
                   .pattern({
                       "person": {"type": "person"},
                       "project": {"type": "project", "status": "active"}
                   }, [
                       ("person", "project", {"label": "works_on"})
                   ])
                   .transform(promote_developer))
    
    rules.append(promote_rule)
    
    # Rule 2: Add collaboration edge between people working on same project
    def add_collaboration(graph: EnhancedGraph, match: Dict[str, str]) -> EnhancedGraph:
        person1_id = match["person1"]
        person2_id = match["person2"]
        
        # Check if collaboration edge already exists
        existing = any(e.target == person2_id and e.label == "collaborates_with" 
                      for e in graph.get_edges_from(person1_id))
        
        if not existing:
            graph.add_edge(person1_id, person2_id, "collaborates_with", 
                          type="inferred", strength="medium")
        
        return graph
    
    collab_rule = (TransformationRule("add_collaboration")
                  .pattern({
                      "person1": {"type": "person"},
                      "person2": {"type": "person"},
                      "project": {"type": "project"}
                  }, [
                      ("person1", "project", {"label": "works_on"}),
                      ("person2", "project", {"label": "works_on"})
                  ], [
                      lambda m: m["person1"] != m["person2"]  # Different people
                  ])
                  .transform(add_collaboration))
    
    rules.append(collab_rule)
    
    return rules


def print_graph_analysis(graph: EnhancedGraph, title: str = "Graph Analysis"):
    """Print detailed graph analysis."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    print(f"Nodes: {len(graph.nodes)}")
    for node in graph.nodes.values():
        print(f"  - {node.id} ({node.type}): {node.attributes}")
    
    print(f"\nEdges: {len(graph.edges)}")
    for edge in graph.edges:
        print(f"  - {edge.source} --[{edge.label}]--> {edge.target}: {edge.attributes}")
    
    # Lambda representation
    lambda_expr = graph.to_lambda_expr()
    print(f"\nLambda Representation: {lambda_expr}")


def print_transformation_output(original: EnhancedGraph, transformed: List[EnhancedGraph], 
                               rule_name: str, lambda_expr: Optional[Expression] = None):
    """Print transformation results in lambda calculus format."""
    print(f"\n{'='*60}")
    print(f"TRANSFORMATION: {rule_name}")
    print(f"{'='*60}")
    
    if lambda_expr:
        print(f"Lambda Rule: {lambda_expr}")
    
    print(f"\nOriginal Graph Lambda: {original.to_lambda_expr()}")
    
    print(f"\nTransformation Results: {len(transformed)} possible outcomes")
    for i, result in enumerate(transformed):
        print(f"\nResult {i+1} Lambda: {result.to_lambda_expr()}")
        if result != original:
            print(f"  - Graph was transformed")
            print(f"  - New nodes: {len(result.nodes)}, New edges: {len(result.edges)}")
        else:
            print(f"  - Graph unchanged")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function demonstrating the Lambda Calculus Graph Transformation System."""
    print("Lambda Calculus Graph Transformation System")
    print("=" * 50)
    
    # Create transformation system
    system = GraphTransformationSystem()
    
    # Create sample graph
    print("\n1. Creating sample graph...")
    graph = create_sample_graph()
    print_graph_analysis(graph, "Original Graph")
    
    # Create and add transformation rules
    print("\n2. Creating transformation rules...")
    rules = create_sample_rules()
    for rule in rules:
        system.add_rule(rule)
        print(f"Added rule: {rule}")
    
    # Get system's lambda representation
    system_lambda = system.get_lambda_representation()
    print(f"\nSystem Lambda Expression: {system_lambda}")
    
    # Apply individual rules
    print("\n3. Applying transformation rules...")
    
    # Apply first rule (promotion)
    transformed1 = system.apply_rule("promote_developer", graph)
    print_transformation_output(graph, transformed1, "promote_developer", rules[0].lambda_expr)
    
    if transformed1:
        graph = transformed1[0]
        print_graph_analysis(graph, "After Promotion Rule")
    
    # Apply second rule (collaboration)
    transformed2 = system.apply_rule("add_collaboration", graph)
    print_transformation_output(graph, transformed2, "add_collaboration", rules[1].lambda_expr)
    
    if transformed2:
        graph = transformed2[0]
        print_graph_analysis(graph, "After Collaboration Rule")
    
    # Apply all rules iteratively
    print("\n4. Iterative transformation...")
    original_graph = create_sample_graph()
    final_graph = system.transform_iteratively(original_graph)
    
    print_transformation_output(original_graph, [final_graph], "Iterative Application", system_lambda)
    print_graph_analysis(final_graph, "Final Transformed Graph")
    
    # Lambda calculus evaluation demo
    print("\n5. Lambda Calculus Evaluation Demo...")
    engine = LambdaEngine()
    
    # Create a simple lambda expression and evaluate it
    # (λx. x) y → y
    identity = Abstraction("x", Variable("x"))
    arg = Variable("y")
    application = Application(identity, arg)
    
    print(f"Original expression: {application}")
    result = engine.evaluate(application)
    print(f"Evaluated result: {result}")
    
    # More complex example: (λf. λx. f (f x)) (λy. y) z → z
    twice = Abstraction("f", Abstraction("x", 
                       Application(Variable("f"), 
                                 Application(Variable("f"), Variable("x")))))
    complex_app = Application(Application(twice, identity), Variable("z"))
    print(f"Complex expression: {complex_app}")
    complex_result = engine.evaluate(complex_app)
    print(f"Complex result: {complex_result}")
    
    # Export final results
    print("\n6. Exporting results...")
    results = {
        "original_graph": original_graph.to_dict(),
        "final_graph": final_graph.to_dict(),
        "system_lambda": str(system_lambda),
        "transformations_applied": len(rules),
        "final_node_count": len(final_graph.nodes),
        "final_edge_count": len(final_graph.edges)
    }
    
    print(f"Results exported: {json.dumps(results, indent=2)}")
    
    print(f"\n{'='*50}")
    print("Lambda Calculus Graph Transformation Complete!")
    print(f"{'='*50}")
    
    return final_graph, system_lambda


if __name__ == "__main__":
    try:
        final_graph, system_lambda_expr = main()
        
        # Additional interactive features
        print("\n" + "="*50)
        print("INTERACTIVE FEATURES")
        print("="*50)
        
        # Allow user to create custom transformations
        print("\n7. Custom Transformation Example...")
        
        def create_custom_graph():
            """Create a custom graph based on user scenario."""
            custom_graph = EnhancedGraph()
            
            # Social network scenario
            custom_graph.add_node("user1", "user", name="Charlie", followers=100)
            custom_graph.add_node("user2", "user", name="Diana", followers=50)
            custom_graph.add_node("post1", "post", content="Hello World!", likes=0)
            custom_graph.add_node("post2", "post", content="Lambda Calculus rocks!", likes=5)
            
            custom_graph.add_edge("user1", "post1", "authored")
            custom_graph.add_edge("user2", "post2", "authored")
            custom_graph.add_edge("user1", "user2", "follows")
            custom_graph.add_edge("user2", "post1", "liked")
            
            return custom_graph
        
        def create_viral_rule():
            """Rule: If a post has more than 3 likes, mark it as viral."""
            def make_viral(graph: EnhancedGraph, match: Dict[str, str]) -> EnhancedGraph:
                post_id = match["post"]
                post = graph.get_node(post_id)
                if post:
                    post.attributes["viral"] = True
                    post.attributes["boosted"] = True
                return graph
            
            return (TransformationRule("viral_content")
                   .pattern({
                       "post": {"type": "post"}
                   }, constraints=[
                       lambda m: graph.get_node(m["post"]) and 
                               graph.get_node(m["post"]).attributes.get("likes", 0) > 3
                   ])
                   .transform(make_viral))
        
        # Create and apply custom transformation
        custom_graph = create_custom_graph()
        print_graph_analysis(custom_graph, "Custom Social Network Graph")
        
        custom_system = GraphTransformationSystem()
        viral_rule = create_viral_rule()
        custom_system.add_rule(viral_rule)
        
        custom_results = custom_system.apply_rule("viral_content", custom_graph)
        print_transformation_output(custom_graph, custom_results, "viral_content", viral_rule.lambda_expr)
        
        # Demonstrate rule composition
        print("\n8. Rule Composition Demo...")
        
        # Create two simple rules
        rule_a = (TransformationRule("add_timestamp")
                 .pattern({"node": {"type": "post"}})
                 .transform(lambda g, m: add_timestamp(g, m)))
        
        rule_b = (TransformationRule("increment_likes")
                 .pattern({"node": {"type": "post"}})
                 .transform(lambda g, m: increment_likes(g, m)))
        
        def add_timestamp(graph: EnhancedGraph, match: Dict[str, str]) -> EnhancedGraph:
            node_id = match["node"]
            node = graph.get_node(node_id)
            if node:
                node.attributes["timestamp"] = "2024-01-01T12:00:00Z"
            return graph
        
        def increment_likes(graph: EnhancedGraph, match: Dict[str, str]) -> EnhancedGraph:
            node_id = match["node"]
            node = graph.get_node(node_id)
            if node:
                current_likes = node.attributes.get("likes", 0)
                node.attributes["likes"] = current_likes + 1
            return graph
        
        # Compose rules
        composed_rule = rule_a.compose_with(rule_b)
        print(f"Composed rule: {composed_rule}")
        print(f"Composed lambda: {composed_rule.lambda_expr}")
        
        # Test composition
        test_graph = create_custom_graph()
        composition_system = GraphTransformationSystem()
        composition_system.add_rule(composed_rule)
        
        composed_results = composition_system.apply_rule(composed_rule.name, test_graph)
        print_transformation_output(test_graph, composed_results, composed_rule.name, composed_rule.lambda_expr)
        
        # Advanced lambda calculus demonstrations
        print("\n9. Advanced Lambda Calculus Features...")
        
        engine = LambdaEngine()
        
        # Church numerals example
        print("\nChurch Numerals:")
        zero = Abstraction("f", Abstraction("x", Variable("x")))
        one = Abstraction("f", Abstraction("x", Application(Variable("f"), Variable("x"))))
        two = Abstraction("f", Abstraction("x", 
                         Application(Variable("f"), 
                                   Application(Variable("f"), Variable("x")))))
        
        print(f"Zero: {zero}")
        print(f"One: {one}")
        print(f"Two: {two}")
        
        # Successor function
        succ = Abstraction("n", Abstraction("f", Abstraction("x",
                          Application(Variable("f"),
                                    Application(Application(Variable("n"), Variable("f")), Variable("x"))))))
        
        print(f"Successor function: {succ}")
        
        # Apply successor to zero
        succ_zero = Application(succ, zero)
        result_one = engine.evaluate(succ_zero)
        print(f"Successor of zero: {result_one}")
        print(f"Alpha equivalent to one: {result_one.alpha_equiv(one)}")
        
        # Boolean logic
        print("\nBoolean Logic:")
        true_expr = Abstraction("x", Abstraction("y", Variable("x")))
        false_expr = Abstraction("x", Abstraction("y", Variable("y")))
        
        print(f"True: {true_expr}")
        print(f"False: {false_expr}")
        
        # Conditional
        if_then_else = Abstraction("p", Abstraction("x", Abstraction("y",
                                  Application(Application(Variable("p"), Variable("x")), Variable("y")))))
        
        test_true = Application(Application(Application(if_then_else, true_expr), Variable("a")), Variable("b"))
        test_false = Application(Application(Application(if_then_else, false_expr), Variable("a")), Variable("b"))
        
        print(f"If true then a else b: {engine.evaluate(test_true)}")
        print(f"If false then a else b: {engine.evaluate(test_false)}")
        
        # Performance analysis
        print("\n10. Performance Analysis...")
        import time
        
        def benchmark_transformation(graph: EnhancedGraph, system: GraphTransformationSystem, iterations: int = 100):
            start_time = time.time()
            for _ in range(iterations):
                result = system.transform_iteratively(graph.copy(), max_iterations=3)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iterations
            return avg_time, result
        
        # Benchmark the system
        avg_time, benchmark_result = benchmark_transformation(custom_graph, custom_system, 10)
        print(f"Average transformation time: {avg_time:.6f} seconds")
        print(f"Transformations per second: {1/avg_time:.2f}")
        
        # Memory usage analysis
        import sys
        graph_size = sys.getsizeof(custom_graph)
        system_size = sys.getsizeof(custom_system)
        
        print(f"Graph memory usage: {graph_size} bytes")
        print(f"System memory usage: {system_size} bytes")
        
        # Export comprehensive results
        print("\n11. Comprehensive Export...")
        
        comprehensive_results = {
            "system_metadata": {
                "total_rules": len(custom_system.rules),
                "lambda_expression": str(custom_system.get_lambda_representation()),
                "performance": {
                    "avg_transformation_time": avg_time,
                    "transformations_per_second": 1/avg_time
                }
            },
            "graphs": {
                "original": custom_graph.to_dict(),
                "transformed": benchmark_result.to_dict() if benchmark_result else None
            },
            "lambda_examples": {
                "church_zero": str(zero),
                "church_one": str(one),
                "church_two": str(two),
                "boolean_true": str(true_expr),
                "boolean_false": str(false_expr)
            }
        }
        
        # Save to file
        try:
            with open("lambda_graph_results.json", "w") as f:
                json.dump(comprehensive_results, f, indent=2)
            print("Results saved to lambda_graph_results.json")
        except Exception as e:
            print(f"Could not save results to file: {e}")
        
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"✓ Created and transformed {len(custom_graph.nodes)} node graphs")
        print(f"✓ Applied {len(custom_system.rules)} transformation rules")
        print(f"✓ Demonstrated lambda calculus evaluation")
        print(f"✓ Showed rule composition capabilities")
        print(f"✓ Performed performance benchmarking")
        print(f"✓ Exported comprehensive results")
        
        # Final lambda calculus representation of the entire session
        session_lambda = Abstraction("session", 
                          Application(
                              Application(Variable("transform"), Variable("graph")),
                              Variable("rules")
                          ))
        
        print(f"\nSession Lambda: {session_lambda}")
        print(f"Final System State: {custom_system.get_lambda_representation()}")
        
        return comprehensive_results
        
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
