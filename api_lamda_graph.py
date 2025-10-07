#!/usr/bin/env python3
"""
FastAPI REST API for Lambda Calculus Graph Transformation System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

from lambda_graph_transform import (
    create_graph,
    create_transformation_rule,
    apply_transformations,
    EnhancedGraph,
    graph_to_dict,
    dict_to_graph,
    Variable,
    Abstraction,
    Application,
    evaluate_lambda_expression,
    match_pattern,
    GraphPattern,
    graph_to_lambda,
    compose_rules,
    TransformationRule
)

app = FastAPI(
    title="Lambda Graph Transform API",
    description="REST API for lambda calculus graph transformations",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class NodeSpec(BaseModel):
    id: str
    type: str = "default"
    attributes: Dict[str, Any] = Field(default_factory=dict)


class EdgeSpec(BaseModel):
    source: str
    target: str
    label: str = "default"
    attributes: Dict[str, Any] = Field(default_factory=dict)


class GraphInput(BaseModel):
    nodes: List[NodeSpec]
    edges: List[EdgeSpec]


class GraphOutput(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class TransformationRuleSpec(BaseModel):
    name: str
    node_patterns: Dict[str, Dict[str, Any]]
    edge_patterns: List[List[Any]] = Field(default_factory=list)
    transformation_type: str  # "add_attribute", "remove_node", "add_edge", "custom"
    transformation_params: Dict[str, Any] = Field(default_factory=dict)


class TransformationRequest(BaseModel):
    graph: GraphInput
    rules: List[TransformationRuleSpec]
    mode: str = "all"  # "all", "iterative", "first"
    max_iterations: int = 10


class TransformationResponse(BaseModel):
    success: bool
    original_graph: GraphOutput
    transformed_graphs: List[GraphOutput]
    lambda_expression: Optional[str] = None
    message: str = ""


class PatternMatchRequest(BaseModel):
    graph: GraphInput
    node_patterns: Dict[str, Dict[str, Any]]
    edge_patterns: List[List[Any]] = Field(default_factory=list)


class PatternMatchResponse(BaseModel):
    success: bool
    matches: List[Dict[str, str]]
    count: int


class LambdaExpressionRequest(BaseModel):
    expression_type: str  # "variable", "abstraction", "application"
    params: Dict[str, Any]


class LambdaEvaluationRequest(BaseModel):
    expression: str  # String representation
    max_steps: int = 1000


class LambdaEvaluationResponse(BaseModel):
    success: bool
    original: str
    result: str
    steps: int


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_graph_input(graph_input: GraphInput) -> EnhancedGraph:
    """Convert GraphInput to EnhancedGraph"""
    nodes = [node.dict() for node in graph_input.nodes]
    edges = [edge.dict() for edge in graph_input.edges]
    return create_graph(nodes, edges)


def create_transform_function(transform_type: str, params: Dict[str, Any]):
    """Create transformation function based on type and parameters"""
    
    if transform_type == "add_attribute":
        attr_name = params.get("attribute_name")
        attr_value = params.get("attribute_value")
        target_var = params.get("target_variable", "node")
        
        def add_attr(graph: EnhancedGraph, match: Dict[str, str]) -> EnhancedGraph:
            node_id = match.get(target_var)
            if node_id:
                node = graph.get_node(node_id)
                if node:
                    node.attributes[attr_name] = attr_value
            return graph
        return add_attr
    
    elif transform_type == "remove_node":
        target_var = params.get("target_variable", "node")
        
        def remove_node(graph: EnhancedGraph, match: Dict[str, str]) -> EnhancedGraph:
            node_id = match.get(target_var)
            if node_id:
                graph.remove_node(node_id)
            return graph
        return remove_node
    
    elif transform_type == "add_edge":
        source_var = params.get("source_variable")
        target_var = params.get("target_variable")
        edge_label = params.get("edge_label", "default")
        edge_attrs = params.get("edge_attributes", {})
        
        def add_edge(graph: EnhancedGraph, match: Dict[str, str]) -> EnhancedGraph:
            source_id = match.get(source_var)
            target_id = match.get(target_var)
            if source_id and target_id:
                graph.add_edge(source_id, target_id, edge_label, **edge_attrs)
            return graph
        return add_edge
    
    elif transform_type == "update_attribute":
        attr_name = params.get("attribute_name")
        operation = params.get("operation", "set")  # "set", "increment", "append"
        value = params.get("value")
        target_var = params.get("target_variable", "node")
        
        def update_attr(graph: EnhancedGraph, match: Dict[str, str]) -> EnhancedGraph:
            node_id = match.get(target_var)
            if node_id:
                node = graph.get_node(node_id)
                if node:
                    if operation == "set":
                        node.attributes[attr_name] = value
                    elif operation == "increment":
                        current = node.attributes.get(attr_name, 0)
                        node.attributes[attr_name] = current + value
                    elif operation == "append":
                        current = node.attributes.get(attr_name, [])
                        if isinstance(current, list):
                            current.append(value)
                            node.attributes[attr_name] = current
            return graph
        return update_attr
    
    else:
        # Default: no-op transformation
        def noop(graph: EnhancedGraph, match: Dict[str, str]) -> EnhancedGraph:
            return graph
        return noop


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Lambda Graph Transform API",
        "version": "1.0.0",
        "endpoints": {
            "create_graph": "/graph/create",
            "transform": "/graph/transform",
            "pattern_match": "/graph/match",
            "lambda_evaluate": "/lambda/evaluate",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "lambda-graph-transform"}


@app.post("/graph/create", response_model=GraphOutput)
async def create_graph_endpoint(graph_input: GraphInput):
    """
    Create a graph from nodes and edges specification
    
    Example:
    ```json
    {
        "nodes": [
            {"id": "A", "type": "person", "attributes": {"name": "Alice"}},
            {"id": "B", "type": "person", "attributes": {"name": "Bob"}}
        ],
        "edges": [
            {"source": "A", "target": "B", "label": "knows"}
        ]
    }
    ```
    """
    try:
        graph = parse_graph_input(graph_input)
        return graph_to_dict(graph)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/graph/transform", response_model=TransformationResponse)
async def transform_graph(request: TransformationRequest):
    """
    Apply transformation rules to a graph
    
    Example:
    ```json
    {
        "graph": {
            "nodes": [{"id": "A", "type": "person", "attributes": {"age": 25}}],
            "edges": []
        },
        "rules": [
            {
                "name": "promote",
                "node_patterns": {"person": {"type": "person"}},
                "edge_patterns": [],
                "transformation_type": "add_attribute",
                "transformation_params": {
                    "attribute_name": "promoted",
                    "attribute_value": true,
                    "target_variable": "person"
                }
            }
        ],
        "mode": "all"
    }
    ```
    """
    try:
        # Parse input graph
        graph = parse_graph_input(request.graph)
        original_dict = graph_to_dict(graph)
        
        # Create transformation rules
        rules = []
        for rule_spec in request.rules:
            # Convert edge patterns from list to tuples
            edge_patterns = [
                (ep[0], ep[1], ep[2]) if len(ep) == 3 else (ep[0], ep[1], {})
                for ep in rule_spec.edge_patterns
            ]
            
            # Create transformation function
            transform_fn = create_transform_function(
                rule_spec.transformation_type,
                rule_spec.transformation_params
            )
            
            # Create rule
            rule = create_transformation_rule(
                name=rule_spec.name,
                node_patterns=rule_spec.node_patterns,
                edge_patterns=edge_patterns,
                transform_fn=transform_fn
            )
            rules.append(rule)
        
        # Apply transformations
        result = apply_transformations(graph, rules, request.mode, request.max_iterations)
        
        # Handle different return types
        if isinstance(result, list):
            transformed_graphs = [graph_to_dict(g) for g in result]
        else:
            transformed_graphs = [graph_to_dict(result)]
        
        # Get lambda representation
        lambda_expr = str(graph_to_lambda(graph))
        
        return TransformationResponse(
            success=True,
            original_graph=original_dict,
            transformed_graphs=transformed_graphs,
            lambda_expression=lambda_expr,
            message=f"Applied {len(rules)} rules in {request.mode} mode"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/graph/match", response_model=PatternMatchResponse)
async def match_pattern_endpoint(request: PatternMatchRequest):
    """
    Find pattern matches in a graph
    
    Example:
    ```json
    {
        "graph": {
            "nodes": [
                {"id": "A", "type": "person"},
                {"id": "B", "type": "person"}
            ],
            "edges": [
                {"source": "A", "target": "B", "label": "knows"}
            ]
        },
        "node_patterns": {
            "p1": {"type": "person"},
            "p2": {"type": "person"}
        },
        "edge_patterns": [
            ["p1", "p2", {"label": "knows"}]
        ]
    }
    ```
    """
    try:
        # Parse input graph
        graph = parse_graph_input(request.graph)
        
        # Convert edge patterns
        edge_patterns = [
            (ep[0], ep[1], ep[2]) if len(ep) == 3 else (ep[0], ep[1], {})
            for ep in request.edge_patterns
        ]
        
        # Create pattern
        pattern = GraphPattern(
            node_patterns=request.node_patterns,
            edge_patterns=edge_patterns,
            constraints=[]
        )
        
        # Find matches
        matches = match_pattern(graph, pattern)
        
        return PatternMatchResponse(
            success=True,
            matches=matches,
            count=len(matches)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/graph/to-lambda")
async def graph_to_lambda_endpoint(graph_input: GraphInput):
    """
    Convert a graph to its lambda calculus representation
    
    Example:
    ```json
    {
        "nodes": [{"id": "A", "type": "node"}],
        "edges": []
    }
    ```
    """
    try:
        graph = parse_graph_input(graph_input)
        lambda_expr = graph_to_lambda(graph)
        
        return {
            "success": True,
            "lambda_expression": str(lambda_expr),
            "graph": graph_to_dict(graph)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/lambda/evaluate", response_model=LambdaEvaluationResponse)
async def evaluate_lambda(request: LambdaEvaluationRequest):
    """
    Evaluate a lambda calculus expression (simplified - uses predefined expressions)
    
    Example:
    ```json
    {
        "expression": "identity",
        "max_steps": 1000
    }
    ```
    
    Supported expressions: "identity", "const", "compose"
    """
    try:
        # Predefined expressions for demo
        expressions = {
            "identity": Abstraction("x", Variable("x")),
            "const": Abstraction("x", Abstraction("y", Variable("x"))),
            "compose": Abstraction("f", Abstraction("g", Abstraction("x",
                Application(Variable("f"), Application(Variable("g"), Variable("x")))))),
        }
        
        if request.expression not in expressions:
            raise ValueError(f"Unknown expression. Available: {list(expressions.keys())}")
        
        expr = expressions[request.expression]
        original = str(expr)
        
        # Evaluate
        result = evaluate_lambda_expression(expr, request.max_steps)
        
        return LambdaEvaluationResponse(
            success=True,
            original=original,
            result=str(result),
            steps=0  # Would need to track this in the engine
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/graph/compose-rules")
async def compose_rules_endpoint(rule_names: List[str]):
    """
    Compose multiple transformation rules
    (This is a simplified version - in practice, you'd need to store rules)
    """
    try:
        return {
            "success": True,
            "message": f"Would compose rules: {rule_names}",
            "composed_rule_name": "∘".join(rule_names)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/examples/basic-graph")
async def example_basic_graph():
    """Get an example basic graph"""
    return {
        "nodes": [
            {"id": "A", "type": "person", "attributes": {"name": "Alice", "age": 30}},
            {"id": "B", "type": "person", "attributes": {"name": "Bob", "age": 25}},
            {"id": "C", "type": "company", "attributes": {"name": "TechCorp"}},
        ],
        "edges": [
            {"source": "A", "target": "C", "label": "works_at", "attributes": {"role": "manager"}},
            {"source": "B", "target": "C", "label": "works_at", "attributes": {"role": "developer"}},
        ]
    }


@app.get("/examples/transformation-rule")
async def example_transformation_rule():
    """Get an example transformation rule"""
    return {
        "name": "promote_to_senior",
        "node_patterns": {
            "person": {"type": "person"},
            "company": {"type": "company"}
        },
        "edge_patterns": [
            ["person", "company", {"label": "works_at"}]
        ],
        "transformation_type": "add_attribute",
        "transformation_params": {
            "attribute_name": "seniority",
            "attribute_value": "senior",
            "target_variable": "person"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
