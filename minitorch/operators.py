"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1 - Elementary Mathematical Functions

def add(x: float, y: float) -> float:
    """Returns the sum of x and y."""
    return x + y

def mul(x: float, y: float) -> float:
    """Returns the product of x and y."""
    return x * y

def neg(x: float) -> float:
    """Returns the negation of x."""
    return -x

def lt(x: float, y: float) -> bool:
    """Returns True if x is less than y, otherwise False."""
    return x < y

def eq(x: float, y: float) -> bool:
    """Returns True if x is equal to y, otherwise False."""
    return x == y

def max(x: float, y: float) -> float:
    """Returns the maximum of x and y."""
    return x if x > y else y

def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    """Returns True if x and y are within the given tolerance."""
    return abs(x - y) < tol

def sigmoid(x: float) -> float:
    """Computes the sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    """Computes the ReLU (Rectified Linear Unit) function."""
    return max(0, x)

def log(x: float) -> float:
    """Computes the natural logarithm of x."""
    return math.log(x)

def exp(x: float) -> float:
    """Computes the exponential of x."""
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    """Computes the gradient of the log function."""
    return d / x

def inv(x: float) -> float:
    """Computes the inverse of x."""
    return 1 / x

def inv_back(x: float, d: float) -> float:
    """Computes the gradient of the inverse function."""
    return -d / (x ** 2)

def relu_back(x: float, d: float) -> float:
    """Computes the gradient of the ReLU function."""
    return d if x > 0 else 0

def id(x: float) -> float:
    """Returns the input as is."""
    return x

def prod(*args):
    """Computes the product of multiple numbers."""
    result = 1.0
    for x in args:
        result *= x
    return result


def sum(lst):
    """Computes the sum of a list."""
    result = 0.0
    for x in lst:
        result += x
    return result


# ## Task 0.3 - Higher Order Functions

def map(fn: Callable[[float], float], lst: Iterable[float]) -> Iterable[float]:
    """Applies a function to each element in a list."""
    return [fn(x) for x in lst]

def zipWith(fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Applies a function to pairs of elements from two lists."""
    return [fn(x, y) for x, y in zip(lst1, lst2)]

def reduce(fn: Callable[[float, float], float], lst: Iterable[float], start: float) -> float:
    """Applies a function cumulatively to a list, starting with an initial value."""
    result = start
    for x in lst:
        result = fn(result, x)
    return result

# Using the above functions to implement additional utilities

def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negates a list of numbers."""
    return map(neg, lst)

def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Adds two lists element-wise."""
    return zipWith(add, lst1, lst2)

def sumList(lst: Iterable[float]) -> float:
    """Computes the sum of a list."""
    return reduce(add, lst, 0.0)

def prodList(lst: Iterable[float]) -> float:
    """Computes the product of a list."""
    return reduce(mul, lst, 1.0)
