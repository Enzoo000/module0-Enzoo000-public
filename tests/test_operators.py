from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch import MathTest
import minitorch
from minitorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.

@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)

@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0

@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0

@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a

@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0

@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0

@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0

# ## Task 0.2 - Property Testing

@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    """Check properties of the sigmoid function."""
    assert 0.0 <= sigmoid(a) <= 1.0
    assert_close(1.0 - sigmoid(a), sigmoid(-a))
    assert_close(sigmoid(0), 0.5)
    
    # Ensure the test only runs for values where sigmoid changes significantly
    if abs(a) < 10:  
        assert sigmoid(a) > sigmoid(a - 0.1)  # Strictly increasing

@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    """Test transitivity: if a < b and b < c, then a < c."""
    if lt(a, b) and lt(b, c):
        assert lt(a, c) == 1.0

@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(x: float, y: float) -> None:
    """Ensure multiplication is symmetric."""
    assert_close(mul(x, y), mul(y, x))

@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x: float, y: float, z: float) -> None:
    """Test distributive property: z * (x + y) = z * x + z * y."""
    assert_close(mul(z, add(x, y)), add(mul(z, x), mul(z, y)))

@pytest.mark.task0_2
@given(small_floats)
def test_other(a: float) -> None:
    """Ensure sigmoid is within range and its derivative is non-negative."""
    assert 0.0 <= sigmoid(a) <= 1.0
    assert sigmoid(a) >= 0.0

# ## Task 0.3 - Higher-order functions

@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)

@pytest.mark.task0_3
@given(lists(small_floats, min_size=5, max_size=5), lists(small_floats, min_size=5, max_size=5))
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    """Test sum distributive property."""
    assert_close(sum(ls1) + sum(ls2), sum(addLists(ls1, ls2)))

@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), minitorch.operators.sum(ls))

@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod(x, y, z), x * y * z)

@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)

# ## Generic mathematical tests

one_arg, two_arg, _ = MathTest._tests()

@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)

@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float) -> None:
    name, base_fn = fn
    base_fn(t1, t2)

@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
