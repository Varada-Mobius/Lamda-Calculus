"""
Concurrent λ-Calculus with π-Calculus Channels
==============================================

A production system combining λ-calculus with π-calculus for concurrent computation.

Main API:
    process_concurrent_system(input_spec) -> output_result
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import time
import uuid
import copy


# ============================================================================
# CORE EXPRESSION TYPES
# ============================================================================

class Expression(ABC):
    """Base class for all λ+π expressions."""
    
    @abstractmethod
    def free_vars(self) -> Set[str]:
        pass
    
    @abstractmethod
    def free_channels(self) -> Set[str]:
        pass
    
    @abstractmethod
    def substitute(self, var: str, value: 'Expression') -> 'Expression':
        pass
    
    def __eq__(self, other) -> bool:
        return str(self) == str(other)


class Variable(Expression):
    def __init__(self, name: str):
        self.name = name
    
    def free_vars(self) -> Set[str]:
        return {self.name}
    
    def free_channels(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, value: Expression) -> Expression:
        return copy.deepcopy(value) if self.name == var else self
    
    def __repr__(self) -> str:
        return self.name


class Abstraction(Expression):
    def __init__(self, var: str, body: Expression):
        self.var = var
        self.body = body
    
    def free_vars(self) -> Set[str]:
        return self.body.free_vars() - {self.var}
    
    def free_channels(self) -> Set[str]:
        return self.body.free_channels()
    
    def substitute(self, var: str, value: Expression) -> Expression:
        if var == self.var:
            return self
        elif var in value.free_vars() and self.var in value.free_vars():
            fresh = self._fresh_var(value.free_vars() | {var})
            new_body = self.body.substitute(self.var, Variable(fresh))
            return Abstraction(fresh, new_body.substitute(var, value))
        else:
            return Abstraction(self.var, self.body.substitute(var, value))
    
    def _fresh_var(self, avoid: Set[str]) -> str:
        counter = 1
        while f"{self.var}_{counter}" in avoid:
            counter += 1
        return f"{self.var}_{counter}"
    
    def __repr__(self) -> str:
        return f"(λ{self.var}.{self.body})"


class Application(Expression):
    def __init__(self, func: Expression, arg: Expression):
        self.func = func
        self.arg = arg
    
    def free_vars(self) -> Set[str]:
        return self.func.free_vars() | self.arg.free_vars()
    
    def free_channels(self) -> Set[str]:
        return self.func.free_channels() | self.arg.free_channels()
    
    def substitute(self, var: str, value: Expression) -> Expression:
        return Application(
            self.func.substitute(var, value),
            self.arg.substitute(var, value)
        )
    
    def __repr__(self) -> str:
        return f"({self.func} {self.arg})"


class Channel(Expression):
    def __init__(self, name: str):
        self.name = name
    
    def free_vars(self) -> Set[str]:
        return set()
    
    def free_channels(self) -> Set[str]:
        return {self.name}
    
    def substitute(self, var: str, value: Expression) -> Expression:
        return self
    
    def __repr__(self) -> str:
        return f"#{self.name}"


class Send(Expression):
    def __init__(self, channel: Channel, value: Expression):
        self.channel = channel
        self.value = value
    
    def free_vars(self) -> Set[str]:
        return self.value.free_vars()
    
    def free_channels(self) -> Set[str]:
        return self.channel.free_channels() | self.value.free_channels()
    
    def substitute(self, var: str, value: Expression) -> Expression:
        return Send(self.channel, self.value.substitute(var, value))
    
    def __repr__(self) -> str:
        return f"send({self.channel}, {self.value})"


class Recv(Expression):
    def __init__(self, channel: Channel):
        self.channel = channel
    
    def free_vars(self) -> Set[str]:
        return set()
    
    def free_channels(self) -> Set[str]:
        return self.channel.free_channels()
    
    def substitute(self, var: str, value: Expression) -> Expression:
        return self
    
    def __repr__(self) -> str:
        return f"recv({self.channel})"


class Parallel(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right
    
    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()
    
    def free_channels(self) -> Set[str]:
        return self.left.free_channels() | self.right.free_channels()
    
    def substitute(self, var: str, value: Expression) -> Expression:
        return Parallel(
            self.left.substitute(var, value),
            self.right.substitute(var, value)
        )
    
    def __repr__(self) -> str:
        return f"({self.left} | {self.right})"


class NewChannel(Expression):
    def __init__(self, channel_name: str, body: Expression):
        self.channel_name = channel_name
        self.body = body
    
    def free_vars(self) -> Set[str]:
        return self.body.free_vars()
    
    def free_channels(self) -> Set[str]:
        return self.body.free_channels() - {self.channel_name}
    
    def substitute(self, var: str, value: Expression) -> Expression:
        return NewChannel(self.channel_name, self.body.substitute(var, value))
    
    def __repr__(self) -> str:
        return f"(new {self.channel_name} in {self.body})"


class Nil(Expression):
    def free_vars(self) -> Set[str]:
        return set()
    
    def free_channels(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, value: Expression) -> Expression:
        return self
    
    def __repr__(self) -> str:
        return "0"


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

@dataclass
class ExecutionResult:
    """Result of concurrent execution."""
    initial_expression: str
    final_expression: str
    steps_taken: int
    communications: List[Dict[str, str]]
    terminated: bool
    deadlocked: bool
    execution_time: float
    analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConcurrentEngine:
    """Reduction engine for concurrent λ+π calculus."""
    
    def __init__(self, max_steps: int = 1000, timeout: float = 10.0):
        self.max_steps = max_steps
        self.timeout = timeout
    
    def reduce(self, expr: Expression) -> ExecutionResult:
        """Execute concurrent expression and return results."""
        initial_str = str(expr)
        start_time = time.time()
        
        current = expr
        steps = 0
        communications = []
        
        for step in range(self.max_steps):
            if time.time() - start_time > self.timeout:
                break
            
            reduction = self._try_reduce(current)
            if reduction is None:
                break
            
            rule, new_expr, metadata = reduction
            current = new_expr
            steps += 1
            
            if 'communication' in metadata:
                communications.append(metadata['communication'])
        
        execution_time = time.time() - start_time
        terminated = self._is_terminated(current)
        deadlocked = not terminated and steps < self.max_steps
        
        analysis = self._analyze_expression(current)
        
        return ExecutionResult(
            initial_expression=initial_str,
            final_expression=str(current),
            steps_taken=steps,
            communications=communications,
            terminated=terminated,
            deadlocked=deadlocked,
            execution_time=execution_time,
            analysis=analysis
        )
    
    def _try_reduce(self, expr: Expression) -> Optional[Tuple[str, Expression, Dict[str, Any]]]:
        """Attempt single reduction step."""
        # β-reduction
        if isinstance(expr, Application) and isinstance(expr.func, Abstraction):
            new_expr = expr.func.body.substitute(expr.func.var, expr.arg)
            return "beta", new_expr, {}
        
        # Communication
        if isinstance(expr, Parallel):
            comm = self._try_communication(expr.left, expr.right)
            if comm:
                return comm
            
            # Reduce left
            left_red = self._try_reduce(expr.left)
            if left_red:
                rule, new_left, meta = left_red
                return f"par_left_{rule}", Parallel(new_left, expr.right), meta
            
            # Reduce right
            right_red = self._try_reduce(expr.right)
            if right_red:
                rule, new_right, meta = right_red
                return f"par_right_{rule}", Parallel(expr.left, new_right), meta
        
        # NewChannel
        if isinstance(expr, NewChannel):
            body_red = self._try_reduce(expr.body)
            if body_red:
                rule, new_body, meta = body_red
                return f"new_{rule}", NewChannel(expr.channel_name, new_body), meta
        
        # Application
        if isinstance(expr, Application):
            func_red = self._try_reduce(expr.func)
            if func_red:
                rule, new_func, meta = func_red
                return f"app_func_{rule}", Application(new_func, expr.arg), meta
            
            arg_red = self._try_reduce(expr.arg)
            if arg_red:
                rule, new_arg, meta = arg_red
                return f"app_arg_{rule}", Application(expr.func, new_arg), meta
        
        return None
    
    def _try_communication(self, left: Expression, right: Expression) -> Optional[Tuple[str, Expression, Dict]]:
        """Attempt communication synchronization."""
        if isinstance(left, Send) and isinstance(right, Application):
            if isinstance(right.func, Abstraction) and isinstance(right.arg, Recv):
                if left.channel.name == right.arg.channel.name:
                    new_expr = Application(right.func, left.value)
                    meta = {
                        'communication': {
                            'channel': left.channel.name,
                            'value': str(left.value)
                        }
                    }
                    return "comm", new_expr, meta
        
        if isinstance(right, Send) and isinstance(left, Application):
            if isinstance(left.func, Abstraction) and isinstance(left.arg, Recv):
                if right.channel.name == left.arg.channel.name:
                    new_expr = Application(left.func, right.value)
                    meta = {
                        'communication': {
                            'channel': right.channel.name,
                            'value': str(right.value)
                        }
                    }
                    return "comm", new_expr, meta
        
        return None
    
    def _is_terminated(self, expr: Expression) -> bool:
        """Check if expression is terminated."""
        if isinstance(expr, Nil) or isinstance(expr, Variable):
            return True
        if isinstance(expr, Abstraction):
            return self._is_terminated(expr.body)
        if isinstance(expr, Parallel):
            return self._is_terminated(expr.left) and self._is_terminated(expr.right)
        return False
    
    def _analyze_expression(self, expr: Expression) -> Dict[str, Any]:
        """Analyze expression properties."""
        sends, receives = self._count_communications(expr)
        
        return {
            'free_variables': list(expr.free_vars()),
            'free_channels': list(expr.free_channels()),
            'send_operations': sends,
            'receive_operations': receives,
            'balanced_communication': sends == receives,
            'expression_size': self._size_of(expr)
        }
    
    def _count_communications(self, expr: Expression) -> Tuple[int, int]:
        """Count send and receive operations."""
        sends, receives = 0, 0
        
        def count(e):
            nonlocal sends, receives
            if isinstance(e, Send):
                sends += 1
                count(e.value)
            elif isinstance(e, Recv):
                receives += 1
            elif isinstance(e, Parallel):
                count(e.left)
                count(e.right)
            elif isinstance(e, Application):
                count(e.func)
                count(e.arg)
            elif isinstance(e, Abstraction):
                count(e.body)
            elif isinstance(e, NewChannel):
                count(e.body)
        
        count(expr)
        return sends, receives
    
    def _size_of(self, expr: Expression) -> int:
        """Calculate expression size."""
        if isinstance(expr, (Variable, Channel, Nil)):
            return 1
        elif isinstance(expr, Abstraction):
            return 1 + self._size_of(expr.body)
        elif isinstance(expr, Application):
            return 1 + self._size_of(expr.func) + self._size_of(expr.arg)
        elif isinstance(expr, Parallel):
            return 1 + self._size_of(expr.left) + self._size_of(expr.right)
        elif isinstance(expr, Send):
            return 1 + self._size_of(expr.value)
        elif isinstance(expr, Recv):
            return 1
        elif isinstance(expr, NewChannel):
            return 1 + self._size_of(expr.body)
        return 1


# ============================================================================
# MAIN API
# ============================================================================

def process_concurrent_system(input_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main API function: Process concurrent λ+π system.
    
    Args:
        input_spec: Dictionary with system specification:
            {
                'expression': Expression or dict specification,
                'max_steps': int (optional, default 1000),
                'timeout': float (optional, default 10.0)
            }
    
    Returns:
        Dictionary with execution results:
            {
                'initial_expression': str,
                'final_expression': str,
                'steps_taken': int,
                'communications': List[Dict],
                'terminated': bool,
                'deadlocked': bool,
                'execution_time': float,
                'analysis': Dict
            }
    """
    # Parse input
    if isinstance(input_spec.get('expression'), dict):
        expr = parse_expression_dict(input_spec['expression'])
    elif isinstance(input_spec.get('expression'), Expression):
        expr = input_spec['expression']
    else:
        raise ValueError("Input must contain 'expression' field")
    
    # Configuration
    max_steps = input_spec.get('max_steps', 1000)
    timeout = input_spec.get('timeout', 10.0)
    
    # Execute
    engine = ConcurrentEngine(max_steps, timeout)
    result = engine.reduce(expr)
    
    return result.to_dict()


def parse_expression_dict(spec: Dict[str, Any]) -> Expression:
    """
    Parse expression from dictionary specification.
    
    Supported types:
        - {'type': 'variable', 'name': str}
        - {'type': 'abstraction', 'var': str, 'body': dict}
        - {'type': 'application', 'func': dict, 'arg': dict}
        - {'type': 'channel', 'name': str}
        - {'type': 'send', 'channel': dict, 'value': dict}
        - {'type': 'recv', 'channel': dict}
        - {'type': 'parallel', 'left': dict, 'right': dict}
        - {'type': 'new_channel', 'name': str, 'body': dict}
        - {'type': 'nil'}
    """
    expr_type = spec.get('type')
    
    if expr_type == 'variable':
        return Variable(spec['name'])
    
    elif expr_type == 'abstraction':
        return Abstraction(spec['var'], parse_expression_dict(spec['body']))
    
    elif expr_type == 'application':
        return Application(
            parse_expression_dict(spec['func']),
            parse_expression_dict(spec['arg'])
        )
    
    elif expr_type == 'channel':
        return Channel(spec['name'])
    
    elif expr_type == 'send':
        return Send(
            parse_expression_dict(spec['channel']),
            parse_expression_dict(spec['value'])
        )
    
    elif expr_type == 'recv':
        return Recv(parse_expression_dict(spec['channel']))
    
    elif expr_type == 'parallel':
        return Parallel(
            parse_expression_dict(spec['left']),
            parse_expression_dict(spec['right'])
        )
    
    elif expr_type == 'new_channel':
        return NewChannel(spec['name'], parse_expression_dict(spec['body']))
    
    elif expr_type == 'nil':
        return Nil()
    
    else:
        raise ValueError(f"Unknown expression type: {expr_type}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_simple_communication(channel_name: str, value_name: str) -> Expression:
    """Create simple send/recv pattern."""
    c = Channel(channel_name)
    v = Variable(value_name)
    sender = Send(c, v)
    receiver = Application(Abstraction("x", Variable("x")), Recv(c))
    return NewChannel(channel_name, Parallel(sender, receiver))


def create_producer_consumer(channel_name: str) -> Expression:
    """Create producer-consumer pattern."""
    ch = Channel(channel_name)
    producer = Abstraction("_", Send(ch, Variable("item")))
    consumer = Abstraction("x", Application(Variable("process"), Recv(ch)))
    return NewChannel(channel_name, Parallel(
        Application(producer, Nil()),
        Application(consumer, Nil())
    ))


def create_request_response(req_channel: str, resp_channel: str) -> Expression:
    """Create request-response pattern."""
    req_ch = Channel(req_channel)
    resp_ch = Channel(resp_channel)
    
    client = Parallel(
        Send(req_ch, Variable("query")),
        Application(Abstraction("result", Variable("result")), Recv(resp_ch))
    )
    
    server = Parallel(
        Application(Abstraction("q", Variable("q")), Recv(req_ch)),
        Send(resp_ch, Variable("answer"))
    )
    
    return NewChannel(req_channel, NewChannel(resp_channel, Parallel(client, server)))


# ============================================================================
# USAGE EXAMPLE (for testing only)
# ============================================================================

if __name__ == "__main__":
    # Example 1: Simple communication via direct API
    expr = create_simple_communication("c", "value")
    
    result = process_concurrent_system({
        'expression': expr,
        'max_steps': 100,
        'timeout': 5.0
    })
    
    print("Result:", result['final_expression'])
    print("Steps:", result['steps_taken'])
    print("Communications:", len(result['communications']))
    print("Terminated:", result['terminated'])
    
    # Example 2: Using dictionary specification
    input_spec = {
        'expression': {
            'type': 'new_channel',
            'name': 'c',
            'body': {
                'type': 'parallel',
                'left': {
                    'type': 'send',
                    'channel': {'type': 'channel', 'name': 'c'},
                    'value': {'type': 'variable', 'name': '42'}
                },
                'right': {
                    'type': 'application',
                    'func': {
                        'type': 'abstraction',
                        'var': 'x',
                        'body': {'type': 'variable', 'name': 'x'}
                    },
                    'arg': {
                        'type': 'recv',
                        'channel': {'type': 'channel', 'name': 'c'}
                    }
                }
            }
        },
        'max_steps': 100
    }
    
    result2 = process_concurrent_system(input_spec)
    print("\nResult 2:", result2['final_expression'])
    print("Deadlocked:", result2['deadlocked'])
