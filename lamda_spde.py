#!/usr/bin/env python3
"""
SPDE Lambda Calculus System
============================

A comprehensive system for symbolic manipulation of Stochastic Partial Differential Equations
using lambda calculus, with export capabilities to UFL/FEniCS format.


"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Set, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import copy


# ============================================================================
# LAMBDA CALCULUS CORE
# ============================================================================

class Expr(ABC):
    """Base class for all lambda calculus expressions."""

    @abstractmethod
    def free_vars(self) -> Set[str]:
        """Return set of free variables."""
        pass

    @abstractmethod
    def substitute(self, var: str, val: 'Expr') -> 'Expr':
        """Substitute variable with value."""
        pass

    @abstractmethod
    def to_latex(self) -> str:
        """Convert to LaTeX representation."""
        pass

    def __eq__(self, other) -> bool:
        return str(self) == str(other)


class Var(Expr):
    """Variable: u, v, x, t, etc."""

    def __init__(self, name: str, domain: str = "real"):
        self.name = name
        self.domain = domain

    def free_vars(self) -> Set[str]:
        return {self.name}

    def substitute(self, var: str, val: Expr) -> Expr:
        if self.name == var:
            return copy.deepcopy(val)
        return self

    def to_latex(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class Abs(Expr):
    """Lambda abstraction: λx.body"""

    def __init__(self, var: str, body: Expr):
        self.var = var
        self.body = body

    def free_vars(self) -> Set[str]:
        return self.body.free_vars() - {self.var}

    def substitute(self, var: str, val: Expr) -> Expr:
        if var == self.var:
            return self
        elif var in val.free_vars() and self.var in val.free_vars():
            fresh_var = self._fresh_var(val.free_vars() | {var})
            new_body = self.body.substitute(self.var, Var(fresh_var))
            return Abs(fresh_var, new_body.substitute(var, val))
        else:
            return Abs(self.var, self.body.substitute(var, val))

    def _fresh_var(self, avoid: Set[str]) -> str:
        """Generate fresh variable name."""
        base = self.var
        counter = 1
        while f"{base}_{counter}" in avoid:
            counter += 1
        return f"{base}_{counter}"

    def to_latex(self) -> str:
        return f"\\lambda {self.var}.{self.body.to_latex()}"

    def __repr__(self) -> str:
        return f"(λ{self.var}.{self.body})"


class App(Expr):
    """Function application: (f arg)"""

    def __init__(self, func: Expr, arg: Expr):
        self.func = func
        self.arg = arg

    def free_vars(self) -> Set[str]:
        return self.func.free_vars() | self.arg.free_vars()

    def substitute(self, var: str, val: Expr) -> Expr:
        return App(
            self.func.substitute(var, val),
            self.arg.substitute(var, val)
        )

    def to_latex(self) -> str:
        return f"{self.func.to_latex()}({self.arg.to_latex()})"

    def __repr__(self) -> str:
        return f"({self.func} {self.arg})"


# ============================================================================
# DIFFERENTIAL OPERATOR EXTENSIONS
# ============================================================================

class DerivativeType(Enum):
    """Types of derivatives."""
    TIME = "time"
    SPATIAL = "spatial"
    GRADIENT = "gradient"
    DIVERGENCE = "divergence"
    LAPLACIAN = "laplacian"


class Derivative(Expr):
    """Differential operator: ∂f/∂x, ∇f, Δf, etc."""

    def __init__(self, func: Expr, wrt: str, deriv_type: DerivativeType = DerivativeType.SPATIAL, order: int = 1):
        self.func = func
        self.wrt = wrt
        self.deriv_type = deriv_type
        self.order = order

    def free_vars(self) -> Set[str]:
        return self.func.free_vars()

    def substitute(self, var: str, val: Expr) -> Expr:
        return Derivative(
            self.func.substitute(var, val),
            self.wrt,
            self.deriv_type,
            self.order
        )

    def to_latex(self) -> str:
        if self.deriv_type == DerivativeType.TIME:
            if self.order == 1:
                return f"\\frac{{\\partial {self.func.to_latex()}}}{{\\partial {self.wrt}}}"
            else:
                return f"\\frac{{\\partial^{self.order} {self.func.to_latex()}}}{{\\partial {self.wrt}^{self.order}}}"
        elif self.deriv_type == DerivativeType.GRADIENT:
            return f"\\nabla {self.func.to_latex()}"
        elif self.deriv_type == DerivativeType.DIVERGENCE:
            return f"\\nabla \\cdot {self.func.to_latex()}"
        elif self.deriv_type == DerivativeType.LAPLACIAN:
            return f"\\Delta {self.func.to_latex()}"
        else:
            return f"\\frac{{\\partial {self.func.to_latex()}}}{{\\partial {self.wrt}}}"

    def __repr__(self) -> str:
        if self.deriv_type == DerivativeType.TIME:
            return f"∂{self.func}/∂{self.wrt}"
        elif self.deriv_type == DerivativeType.GRADIENT:
            return f"∇{self.func}"
        elif self.deriv_type == DerivativeType.LAPLACIAN:
            return f"Δ{self.func}"
        else:
            return f"∂{self.func}/∂{self.wrt}"


class BinaryOp(Expr):
    """Binary operations: +, -, *, /, etc."""

    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, var: str, val: Expr) -> Expr:
        return BinaryOp(
            self.op,
            self.left.substitute(var, val),
            self.right.substitute(var, val)
        )

    def to_latex(self) -> str:
        if self.op == "*":
            return f"{self.left.to_latex()} \\cdot {self.right.to_latex()}"
        elif self.op == "/":
            return f"\\frac{{{self.left.to_latex()}}}{{{self.right.to_latex()}}}"
        else:
            return f"({self.left.to_latex()} {self.op} {self.right.to_latex()})"

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class Noise(Expr):
    """Stochastic forcing term."""

    def __init__(self, label: str = "ξ", noise_type: str = "white", intensity: float = 1.0):
        self.label = label
        self.noise_type = noise_type
        self.intensity = intensity

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, var: str, val: Expr) -> Expr:
        return self

    def to_latex(self) -> str:
        if self.intensity != 1.0:
            return f"{self.intensity}\\{self.label}"
        return f"\\{self.label}"

    def __repr__(self) -> str:
        return self.label


class StochasticProcess(Expr):
    """Probabilistic lambda term: Ξ = λω.ξ(ω)"""

    def __init__(self, sample_var: str = "ω", noise: Noise = None):
        self.sample_var = sample_var
        self.noise = noise or Noise()

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, var: str, val: Expr) -> Expr:
        return self

    def to_latex(self) -> str:
        return f"\\lambda {self.sample_var}.{self.noise.to_latex()}({self.sample_var})"

    def sample(self) -> Noise:
        """Sample from the stochastic process."""
        return self.noise

    def __repr__(self) -> str:
        return f"(λ{self.sample_var}.{self.noise}({self.sample_var}))"


# ============================================================================
# REDUCTION ENGINE
# ============================================================================

class SPDEReductionEngine:
    """Advanced reduction engine for SPDE expressions."""

    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.reduction_count = 0

    def beta_step(self, expr: Expr) -> Optional[Expr]:
        """Single beta reduction step."""
        if isinstance(expr, App) and isinstance(expr.func, Abs):
            return expr.func.body.substitute(expr.func.var, expr.arg)

        if isinstance(expr, App):
            red_func = self.beta_step(expr.func)
            if red_func:
                return App(red_func, expr.arg)

            red_arg = self.beta_step(expr.arg)
            if red_arg:
                return App(expr.func, red_arg)

        if isinstance(expr, Abs):
            red_body = self.beta_step(expr.body)
            if red_body:
                return Abs(expr.var, red_body)

        if isinstance(expr, BinaryOp):
            red_left = self.beta_step(expr.left)
            if red_left:
                return BinaryOp(expr.op, red_left, expr.right)
            red_right = self.beta_step(expr.right)
            if red_right:
                return BinaryOp(expr.op, expr.left, red_right)

        if isinstance(expr, Derivative):
            red_func = self.beta_step(expr.func)
            if red_func:
                return Derivative(red_func, expr.wrt, expr.deriv_type, expr.order)

        return None

    def normalize(self, expr: Expr) -> Expr:
        """Normalize expression to beta-normal form."""
        self.reduction_count = 0
        current = expr

        for _ in range(self.max_steps):
            nxt = self.beta_step(current)
            if not nxt:
                break
            current = nxt
            self.reduction_count += 1

        return current

    def apply_operator(self, operator: Abs, operand: Expr) -> Expr:
        """Apply differential operator to operand."""
        return self.normalize(App(operator, operand))


# ============================================================================
# SYMBOLIC DIFFERENTIAL OPERATORS
# ============================================================================

class SPDEOperators:
    """Library of common SPDE operators as lambda abstractions."""

    @staticmethod
    def time_derivative() -> Abs:
        """∂/∂t operator: λf.∂f/∂t"""
        return Abs("f", Derivative(Var("f"), "t", DerivativeType.TIME))

    @staticmethod
    def gradient() -> Abs:
        """∇ operator: λf.∇f"""
        return Abs("f", Derivative(Var("f"), "x", DerivativeType.GRADIENT))

    @staticmethod
    def divergence() -> Abs:
        """∇· operator: λf.∇·f"""
        return Abs("f", Derivative(Var("f"), "x", DerivativeType.DIVERGENCE))

    @staticmethod
    def laplacian() -> Abs:
        """Δ operator: λf.Δf"""
        return Abs("f", Derivative(Var("f"), "x", DerivativeType.LAPLACIAN))

    @staticmethod
    def heat_operator(diffusivity: float = 1.0) -> Abs:
        """Heat operator: λu.(∂u/∂t - α·Δu)"""
        dt = SPDEOperators.time_derivative()
        laplace = SPDEOperators.laplacian()

        def heat_body(u_var: Var) -> BinaryOp:
            u_t = App(dt, u_var)
            alpha_delta_u = BinaryOp("*", Var(str(diffusivity)), App(laplace, u_var))
            return BinaryOp("-", u_t, alpha_delta_u)

        return Abs("u", heat_body(Var("u")))

    @staticmethod
    def stochastic_heat_operator(diffusivity: float = 1.0, noise_intensity: float = 1.0) -> Abs:
        """Stochastic heat operator: λu.(∂u/∂t - α·Δu + σ·ξ)"""
        heat_op = SPDEOperators.heat_operator(diffusivity)
        noise = Noise("ξ", "white", noise_intensity)

        def stochastic_heat_body(u_var: Var) -> BinaryOp:
            heat_part = App(heat_op, u_var)
            return BinaryOp("+", heat_part, noise)

        return Abs("u", stochastic_heat_body(Var("u")))

    @staticmethod
    def wave_operator(speed: float = 1.0) -> Abs:
        """Wave operator: λu.(∂²u/∂t² - c²·Δu)"""
        dt2 = Abs("f", Derivative(Var("f"), "t", DerivativeType.TIME, order=2))
        laplace = SPDEOperators.laplacian()

        def wave_body(u_var: Var) -> BinaryOp:
            u_tt = App(dt2, u_var)
            c2_delta_u = BinaryOp("*", Var(str(speed**2)), App(laplace, u_var))
            return BinaryOp("-", u_tt, c2_delta_u)

        return Abs("u", wave_body(Var("u")))

    @staticmethod
    def advection_diffusion_operator(velocity: List[float] = [1.0], diffusivity: float = 0.1) -> Abs:
        """Advection-diffusion operator: λu.(∂u/∂t + v·∇u - D·Δu)"""
        dt = SPDEOperators.time_derivative()
        grad = SPDEOperators.gradient()
        laplace = SPDEOperators.laplacian()

        def advection_diffusion_body(u_var: Var) -> BinaryOp:
            u_t = App(dt, u_var)
            v_dot_grad_u = BinaryOp("*", Var("v"), App(grad, u_var))
            D_delta_u = BinaryOp("*", Var(str(diffusivity)), App(laplace, u_var))

            temp = BinaryOp("+", u_t, v_dot_grad_u)
            return BinaryOp("-", temp, D_delta_u)

        return Abs("u", advection_diffusion_body(Var("u")))


# ============================================================================
# UFL/FENICS EXPORT SYSTEM
# ============================================================================

class UFLExporter:
    """Export symbolic expressions to UFL format for FEniCS."""

    def __init__(self):
        self.function_space = "V"
        self.test_function = "v"
        self.trial_function = "u"

    def to_ufl_string(self, expr: Expr) -> str:
        """Convert expression to UFL string."""
        if isinstance(expr, Var):
            if expr.name == "u":
                return "u"
            elif expr.name == "v":
                return "v"
            elif expr.domain == "test_function":
                return "v"
            else:
                return expr.name

        elif isinstance(expr, Derivative):
            func_str = self.to_ufl_string(expr.func)

            if expr.deriv_type == DerivativeType.TIME:
                return f"u_t"
            elif expr.deriv_type == DerivativeType.GRADIENT:
                return f"grad({func_str})"
            elif expr.deriv_type == DerivativeType.DIVERGENCE:
                return f"div({func_str})"
            elif expr.deriv_type == DerivativeType.LAPLACIAN:
                return f"div(grad({func_str}))"
            else:
                return f"Dx({func_str}, {expr.order})"

        elif isinstance(expr, BinaryOp):
            left_str = self.to_ufl_string(expr.left)
            right_str = self.to_ufl_string(expr.right)

            if expr.op in ["+", "-"]:
                return f"({left_str} {expr.op} {right_str})"
            elif expr.op == "*":
                return f"{left_str}*{right_str}"
            elif expr.op == "/":
                return f"({left_str}/{right_str})"
            else:
                return f"({left_str} {expr.op} {right_str})"

        elif isinstance(expr, Noise):
            return f"xi"

        elif isinstance(expr, StochasticProcess):
            return f"xi"

        elif isinstance(expr, App):
            return f"{self.to_ufl_string(expr.func)}({self.to_ufl_string(expr.arg)})"

        elif isinstance(expr, Abs):
            return f"lambda_{expr.var}({self.to_ufl_string(expr.body)})"

        else:
            return str(expr)

    def to_variational_form(self, expr: Expr, test_func: str = "v") -> str:
        """Convert to weak form suitable for FEniCS."""
        expr_str = self.to_ufl_string(expr)
        return f"inner(({expr_str}), {test_func})*dx"

    def to_fenics_code(self, expr: Expr, equation_name: str = "pde") -> str:
        """Generate complete FEniCS code template."""
        ufl_str = self.to_ufl_string(expr)
        variational_form = self.to_variational_form(expr)

        return f"""# Generated FEniCS code for {equation_name}
from fenics import *

# Define function space
V = FunctionSpace(mesh, 'P', 1)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Stochastic forcing (if present)
xi = Function(V)

# Weak form: {ufl_str}
a = {variational_form}

# Solve (add boundary conditions and time-stepping as needed)
# solve(a == 0, u, bcs)
"""


# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

def create_operator(operator_type: str, **params) -> Abs:
    """
    Create a differential operator.
    
    Args:
        operator_type: One of 'time_derivative', 'gradient', 'divergence', 
                      'laplacian', 'heat', 'stochastic_heat', 'wave', 
                      'advection_diffusion'
        **params: Operator-specific parameters
    
    Returns:
        Lambda abstraction representing the operator
    """
    if operator_type == "time_derivative":
        return SPDEOperators.time_derivative()
    elif operator_type == "gradient":
        return SPDEOperators.gradient()
    elif operator_type == "divergence":
        return SPDEOperators.divergence()
    elif operator_type == "laplacian":
        return SPDEOperators.laplacian()
    elif operator_type == "heat":
        return SPDEOperators.heat_operator(params.get("diffusivity", 1.0))
    elif operator_type == "stochastic_heat":
        return SPDEOperators.stochastic_heat_operator(
            params.get("diffusivity", 1.0),
            params.get("noise_intensity", 1.0)
        )
    elif operator_type == "wave":
        return SPDEOperators.wave_operator(params.get("speed", 1.0))
    elif operator_type == "advection_diffusion":
        return SPDEOperators.advection_diffusion_operator(
            params.get("velocity", [1.0]),
            params.get("diffusivity", 0.1)
        )
    else:
        raise ValueError(f"Unknown operator type: {operator_type}")


def apply_operator(operator: Abs, operand: Expr, normalize: bool = True) -> Expr:
    """
    Apply a differential operator to an operand.
    
    Args:
        operator: Lambda abstraction representing the operator
        operand: Expression to apply operator to
        normalize: Whether to normalize the result
    
    Returns:
        Resulting expression
    """
    engine = SPDEReductionEngine()
    if normalize:
        return engine.apply_operator(operator, operand)
    else:
        return App(operator, operand)


def normalize_expression(expr: Expr, max_steps: int = 100) -> Expr:
    """
    Normalize an expression to beta-normal form.
    
    Args:
        expr: Expression to normalize
        max_steps: Maximum reduction steps
    
    Returns:
        Normalized expression
    """
    engine = SPDEReductionEngine(max_steps)
    return engine.normalize(expr)


def to_latex(expr: Expr) -> str:
    """Convert expression to LaTeX string."""
    return expr.to_latex()


def to_ufl(expr: Expr) -> str:
    """Convert expression to UFL string for FEniCS."""
    exporter = UFLExporter()
    return exporter.to_ufl_string(expr)


def to_variational_form(expr: Expr, test_func: str = "v") -> str:
    """Convert expression to variational form."""
    exporter = UFLExporter()
    return exporter.to_variational_form(expr, test_func)


def to_fenics_code(expr: Expr, equation_name: str = "pde") -> str:
    """Generate complete FEniCS code for the expression."""
    exporter = UFLExporter()
    return exporter.to_fenics_code(expr, equation_name)


def create_variable(name: str, domain: str = "real") -> Var:
    """Create a variable."""
    return Var(name, domain)


def create_noise(label: str = "ξ", noise_type: str = "white", intensity: float = 1.0) -> Noise:
    """Create a noise term."""
    return Noise(label, noise_type, intensity)


def create_stochastic_process(sample_var: str = "ω", noise: Optional[Noise] = None) -> StochasticProcess:
    """Create a stochastic process."""
    return StochasticProcess(sample_var, noise)


def create_derivative(func: Expr, wrt: str, deriv_type: str = "spatial", order: int = 1) -> Derivative:
    """
    Create a derivative.
    
    Args:
        func: Function to differentiate
        wrt: Variable to differentiate with respect to
        deriv_type: Type of derivative ('time', 'spatial', 'gradient', 'divergence', 'laplacian')
        order: Order of derivative
    
    Returns:
        Derivative expression
    """
    dtype = DerivativeType[deriv_type.upper()]
    return Derivative(func, wrt, dtype, order)


def create_binary_op(op: str, left: Expr, right: Expr) -> BinaryOp:
    """Create a binary operation."""
    return BinaryOp(op, left, right)


def substitute(expr: Expr, var: str, value: Expr) -> Expr:
    """Substitute variable in expression."""
    return expr.substitute(var, value)


def get_free_vars(expr: Expr) -> Set[str]:
    """Get free variables in expression."""
    return expr.free_vars()


__all__ = [
    'Expr', 'Var', 'Abs', 'App', 'Derivative', 'BinaryOp', 'Noise', 'StochasticProcess',
    'DerivativeType', 'SPDEReductionEngine', 'SPDEOperators', 'UFLExporter',
    'create_operator', 'apply_operator', 'normalize_expression',
    'to_latex', 'to_ufl', 'to_variational_form', 'to_fenics_code',
    'create_variable', 'create_noise', 'create_stochastic_process',
    'create_derivative', 'create_binary_op', 'substitute', 'get_free_vars'
]
