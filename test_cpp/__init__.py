from typing import Tuple

import numpy as np

import pxt.build
import pxt.link


test_cpp = pxt.imp(pxt.build.cpp('test_cpp.cpp'), 'test_cpp.test_cpp')


@pxt.link.mod(test_cpp)
def i_add(a: int, b: int) -> int:
    print('call python i_add')
    return a + b


@pxt.link.mod(test_cpp)
def np_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    print('call python np_add')
    return a + b


@pxt.link.mod(test_cpp)
def return_tuple(a: np.ndarray, b: np.ndarray) -> Tuple:
    print('call python return_tuple')
    return a, b
