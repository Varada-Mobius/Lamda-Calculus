"""
Lambda Calculus Mathematical Expression Playbook
=================================================

A comprehensive system for converting mathematical expressions to lambda calculus,
supporting a wide range of operators, functions, and mathematical constructs.

Features:
- Extensive operator library (arithmetic, trigonometric, logarithmic, etc.)
- Custom function definitions
- Matrix and vector operations
- Statistical functions
- Calculus operations (derivatives, integrals)
- Advanced mathematical functions
- Export to multiple formats
- Interactive exploration tools
"""

import math
import numpy as np
from sympy import (
    sympify, symbols, Pow, sin, cos, tan, exp, log, sqrt, 
    factorial, gamma, beta, erf, Abs as SymAbs,
    diff, integrate, limit, series, Matrix,
    I, pi, E, oo, zoo, nan
)
from lambda_calculus.terms import Variable, Abstraction, Application
from typing import Dict, List, Set, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import copy

# ============================================================================
# COMPREHENSIVE OPERATOR LIBRARY
# ============================================================================

class OperatorLibrary:
    """Comprehensive library of mathematical operators as lambda calculus variables."""
    
    # Basic Arithmetic
    ADD = Variable("+")
    SUB = Variable("-") 
    MUL = Variable("*")
    DIV = Variable("/")
    MOD = Variable("%")
    POW = Variable("^")
    
    # Comparison Operators
    EQ = Variable("==")
    NE = Variable("!=")
    LT = Variable("<")
    LE = Variable("<=")
    GT = Variable(">")
    GE = Variable(">=")
    
    # Logical Operators
    AND = Variable("&&")
    OR = Variable("||")
    NOT = Variable("!")
    
    # Trigonometric Functions
    SIN = Variable("sin")
    COS = Variable("cos")
    TAN = Variable("tan")
    ASIN = Variable("arcsin")
    ACOS = Variable("arccos")
    ATAN = Variable("arctan")
    SINH = Variable("sinh")
    COSH = Variable("cosh")
    TANH = Variable("tanh")
    
    # Exponential and Logarithmic
    EXP = Variable("exp")
    LOG = Variable("log")
    LOG10 = Variable("log10")
    LOG2 = Variable("log2")
    LN = Variable("ln")
    
    # Root Functions
    SQRT = Variable("sqrt")
    CBRT = Variable("cbrt")
    
    # Rounding and Absolute
    ABS = Variable("abs")
    FLOOR = Variable("floor")
    CEIL = Variable("ceil")
    ROUND = Variable("round")
    
    # Special Functions
    FACTORIAL = Variable("factorial")
    GAMMA = Variable("gamma")
    BETA = Variable("beta")
    ERF = Variable("erf")
    ERFC = Variable("erfc")
    
    # Statistics
    MEAN = Variable("mean")
    MEDIAN = Variable("median")
    STD = Variable("std")
    VAR = Variable("var")
    MAX = Variable("max")
    MIN = Variable("min")
    
    # Linear Algebra
    DOT = Variable("dot")
    CROSS = Variable("cross")
    NORM = Variable("norm")
    DET = Variable("det")
    INV = Variable("inv")
    TRANSPOSE = Variable("transpose")
    
    # Calculus
    DIFF = Variable("diff")
    INTEGRATE = Variable("integrate")
    LIMIT = Variable("limit")
    SERIES = Variable("series")
    
    # Constants
    PI = Variable("π")
    E = Variable("e")
    I = Variable("i")  # imaginary unit
    INF = Variable("∞")
    
    @classmethod
    def get_operator(cls, name: str) -> Optional[Variable]:
        """Get operator by name."""
        return getattr(cls, name.upper(), None)
    
    @classmethod
    def list_operators(cls) -> List[str]:
        """List all available operators."""
        return [attr for attr in dir(cls) 
                if not attr.startswith('_') and isinstance(getattr(cls, attr), Variable)]


class ExpressionType(Enum):
    """Types of mathematical expressions."""
    POLYNOMIAL = "polynomial"
    TRIGONOMETRIC = "trigonometric" 
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    RATIONAL = "rational"
    RADICAL = "radical"
    PIECEWISE = "piecewise"
    VECTOR = "vector"
    MATRIX = "matrix"
    DIFFERENTIAL = "differential"
    INTEGRAL = "integral"
    STATISTICAL = "statistical"


# ============================================================================
# ENHANCED SYMPY TO LAMBDA CONVERTER
# ============================================================================

class EnhancedLambdaConverter:
    """Enhanced converter with comprehensive mathematical function support."""
    
    def __init__(self):
        self.ops = OperatorLibrary()
        self.custom_functions = {}
        self.conversion_stats = {"converted": 0, "failed": 0}
    
    def register_custom_function(self, name: str, lambda_var: Variable):
        """Register custom function mapping."""
        self.custom_functions[name] = lambda_var
    
    def sympy_to_lambda(self, expr) -> Variable:
        """Enhanced conversion supporting comprehensive mathematical functions."""
        try:
            result = self._convert_expr(expr)
            self.conversion_stats["converted"] += 1
            return result
        except Exception as e:
            self.conversion_stats["failed"] += 1
            raise NotImplementedError(f"Conversion failed for {expr}: {e}")
    
    def _convert_expr(self, expr):
        """Internal conversion logic."""
        # Basic types
        if expr.is_Symbol:
            return Variable(str(expr))
        
        elif expr.is_Number:
            if expr == pi:
                return self.ops.PI
            elif expr == E:
                return self.ops.E
            elif expr == I:
                return self.ops.I
            else:
                return Variable(str(expr))
        
        # Arithmetic operations
        elif expr.is_Add:
            return self._handle_addition(expr)
        
        elif expr.is_Mul:
            return self._handle_multiplication(expr)
        
        elif expr.is_Pow:
            base, exp_val = expr.args
            return Application(
                Application(self.ops.POW, self.sympy_to_lambda(base)),
                self.sympy_to_lambda(exp_val)
            )
        
        # Trigonometric functions
        elif expr.func == sin:
            return Application(self.ops.SIN, self.sympy_to_lambda(expr.args[0]))
        elif expr.func == cos:
            return Application(self.ops.COS, self.sympy_to_lambda(expr.args[0]))
        elif expr.func == tan:
            return Application(self.ops.TAN, self.sympy_to_lambda(expr.args[0]))
        
        # Exponential and logarithmic
        elif expr.func == exp:
            return Application(self.ops.EXP, self.sympy_to_lambda(expr.args[0]))
        elif expr.func == log:
            if len(expr.args) == 1:
                return Application(self.ops.LN, self.sympy_to_lambda(expr.args[0]))
            else:
                # log(x, base)
                x, base = expr.args
                return Application(
                    Application(self.ops.LOG, self.sympy_to_lambda(x)),
                    self.sympy_to_lambda(base)
                )
        
        # Square root
        elif expr.func == sqrt:
            return Application(self.ops.SQRT, self.sympy_to_lambda(expr.args[0]))
        
        # Absolute value
        elif expr.func == SymAbs:
            return Application(self.ops.ABS, self.sympy_to_lambda(expr.args[0]))
        
        # Factorial
        elif expr.func == factorial:
            return Application(self.ops.FACTORIAL, self.sympy_to_lambda(expr.args[0]))
        
        # Special functions
        elif expr.func == gamma:
            return Application(self.ops.GAMMA, self.sympy_to_lambda(expr.args[0]))
        elif expr.func == beta:
            return Application(
                Application(self.ops.BETA, self.sympy_to_lambda(expr.args[0])),
                self.sympy_to_lambda(expr.args[1])
            )
        elif expr.func == erf:
            return Application(self.ops.ERF, self.sympy_to_lambda(expr.args[0]))
        
        # Derivatives
        elif hasattr(expr, 'func') and str(expr.func) == 'Derivative':
            return self._handle_derivative(expr)
        
        # Integrals  
        elif hasattr(expr, 'func') and str(expr.func) == 'Integral':
            return self._handle_integral(expr)
        
        # Generic function handling
        elif hasattr(expr, 'func'):
            return self._handle_generic_function(expr)
        
        else:
            raise NotImplementedError(f"Unsupported expression: {expr}")
    
    def _handle_addition(self, expr):
        """Handle addition with proper subtraction for negative terms."""
        terms = list(expr.args)
        if len(terms) == 1:
            return self.sympy_to_lambda(terms[0])
        
        result = self.sympy_to_lambda(terms[0])
        for term in terms[1:]:
            if term.is_Number and term < 0:
                result = Application(
                    Application(self.ops.SUB, result),
                    self.sympy_to_lambda(-term)
                )
            else:
                result = Application(
                    Application(self.ops.ADD, result),
                    self.sympy_to_lambda(term)
                )
        return result
    
    def _handle_multiplication(self, expr):
        """Handle multiplication operations."""
        terms = list(expr.args)
        if len(terms) == 1:
            return self.sympy_to_lambda(terms[0])
        
        result = self.sympy_to_lambda(terms[0])
        for term in terms[1:]:
            result = Application(
                Application(self.ops.MUL, result),
                self.sympy_to_lambda(term)
            )
        return result
    
    def _handle_derivative(self, expr):
        """Handle derivative expressions."""
        # Simplified: just represent as diff(f, x)
        func = expr.args[0]
        var = expr.args[1]
        return Application(
            Application(self.ops.DIFF, self.sympy_to_lambda(func)),
            self.sympy_to_lambda(var)
        )
    
    def _handle_integral(self, expr):
        """Handle integral expressions."""
        # Simplified: represent as integrate(f, x)
        func = expr.args[0]
        var = expr.args[1]
        return Application(
            Application(self.ops.INTEGRATE, self.sympy_to_lambda(func)),
            self.sympy_to_lambda(var)
        )
    
    def _handle_generic_function(self, expr):
        """Handle generic functions."""
        func_name = str(expr.func)
        
        # Check custom functions first
        if func_name in self.custom_functions:
            func_var = self.custom_functions[func_name]
        else:
            func_var = Variable(func_name)
        
        if expr.args:
            result = func_var
            for arg in expr.args:
                result = Application(result, self.sympy_to_lambda(arg))
            return result
        else:
            return func_var


# ============================================================================
# ENHANCED FUNCTION PARSER
# ============================================================================

class MathematicalExpressionParser:
    """Enhanced parser for mathematical expressions."""
    
    def __init__(self):
        self.converter = EnhancedLambdaConverter()
        self.expression_cache = {}
    
    def parse_function(self, expr_str: str, variables: Optional[List[str]] = None,
                      expression_type: Optional[ExpressionType] = None) -> Abstraction:
        """
        Parse mathematical expression with enhanced capabilities.
        
        Args:
            expr_str: Mathematical expression as string
            variables: Variable order for lambda abstraction
            expression_type: Type hint for optimization
            
        Returns:
            Lambda calculus abstraction
        """
        # Check cache first
        cache_key = (expr_str, tuple(variables) if variables else None)
        if cache_key in self.expression_cache:
            return self.expression_cache[cache_key]
        
        try:
            expr = sympify(expr_str)
        except Exception as e:
            raise ValueError(f"Failed to parse expression '{expr_str}': {e}")
        
        # Auto-detect variables if not provided
        if variables is None:
            variables = sorted([str(s) for s in expr.free_symbols])
            if not variables:
                result = self.converter.sympy_to_lambda(expr)
                self.expression_cache[cache_key] = result
                return result
        
        # Convert to lambda calculus
        lam_expr = self.converter.sympy_to_lambda(expr)
        
        # Wrap in abstractions (currying)
        for v in reversed(variables):
            lam_expr = Abstraction(Variable(v), lam_expr)
        
        self.expression_cache[cache_key] = lam_expr
        return lam_expr
    
    def parse_multiple(self, expressions: Dict[str, str], 
                      shared_variables: Optional[List[str]] = None) -> Dict[str, Abstraction]:
        """Parse multiple related expressions."""
        results = {}
        for name, expr_str in expressions.items():
            results[name] = self.parse_function(expr_str, shared_variables)
        return results
    
    def analyze_expression(self, expr_str: str) -> Dict[str, Any]:
        """Analyze mathematical expression properties."""
        expr = sympify(expr_str)
        
        analysis = {
            "original": expr_str,
            "sympy_form": str(expr),
            "free_symbols": [str(s) for s in expr.free_symbols],
            "is_polynomial": expr.is_polynomial(),
            "complexity": self._measure_complexity(expr),
            "has_trig": any(func in str(expr) for func in ['sin', 'cos', 'tan']),
            "has_exp": 'exp' in str(expr) or 'log' in str(expr),
            "has_special": any(func in str(expr) for func in ['gamma', 'beta', 'erf']),
        }
        
        return analysis
    
    def _measure_complexity(self, expr) -> int:
        """Measure expression complexity."""
        return len(str(expr).replace(' ', ''))
