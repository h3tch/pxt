import sys

from hypothesis import given
import hypothesis.strategies as st

import pxt
from foo import rs_i_add, rs_no_add, cpp_i_add, cpp_np_add, return_tuple


# import gc
# import objgraph

#  gc.set_debug(gc.DEBUG_LEAK)
#  gc.collect()

# Example using hypothesis
# interestingly, fails when using sys.maxsize ;)
@given(st.integers(0, 256), st.integers(0, 256))
def test_simple_rs_add(x, y):
    assert x + y == rs_i_add(x, y)

@given(st.integers(0, 256), st.integers(0, 256))
def test_no_rs_add(x, y):
    assert x + y == rs_no_add(x, y)

def test_cpp_np_add(a, b):
    c = cpp_np_add(a, b)
    # objgraph.show_growth()
    assert c

@given(st.integers(0, 256), st.integers(0, 256))
def test_cpp_i_add(x, y):
    assert x + y == cpp_i_add(x, y)

def test_cpp_tupple(a, b):
    tup = return_tuple(a, b)
    assert tuple(a,b) == return_tupple(a, b)
