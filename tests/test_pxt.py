import hypothesis.strategies as st
import numpy as np
import tests.foo
from hypothesis import given


# Example using hypothesis
# interestingly, fails when using sys.maxsize ;)
@given(st.integers(0, 256), st.integers(0, 256))
def test_simple_rs_add(x, y):
    assert x + y == tests.foo.rs_i_add(x, y)


@given(st.integers(0, 256), st.integers(0, 256))
def test_no_rs_add(x, y):
    assert x + y == tests.foo.rs_no_add(x, y)


def test_cpp_np_add(a, b):
    c = tests.foo.cpp_np_add(a, b)
    assert c is not None


if hasattr(tests.foo, 'cu_multiply'):
    n = 256
    a = np.random.random((n,)).astype(np.float32)
    b = np.random.random((n,)).astype(np.float32)
    # Note that the grid argument was specified in the
    # decorator, but could also be provided here.
    c = tests.foo.cu_multiply(a, b, block=(n, 1, 1))
    assert np.allclose(c[:n] - a * b, 0.)


@given(st.integers(0, 256), st.integers(0, 256))
def test_cpp_i_add(x, y):
    assert x + y == tests.foo.cpp_i_add(x, y)

