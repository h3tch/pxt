import logging
from typing import Tuple

import numpy as np
import pxt.link

try:
    from pxt.cuda import In, Out
    CUDA = True
except ImportError:
    CUDA = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cpp_lib = pxt.imp(pxt.build.cpp('cpp/test_cpp.cpp'), 'cpp.test_cpp')


@pxt.link.mod(cpp_lib, function='i_add')
def cpp_i_add(a: int, b: int) -> int:
    raise NotImplementedError


@pxt.link.mod(cpp_lib, function='np_add')
def cpp_np_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    raise NotImplementedError


@pxt.link.mod(cpp_lib)
def return_tuple(a: np.ndarray, b: np.ndarray) -> Tuple:
    raise NotImplementedError


@pxt.link.mod(cpp_lib, function='no_func')
def cpp_no_func(a: int, b: int) -> int:
    return a + b


if CUDA:
    @pxt.link.cuda('cu/multiply.cubin')
    @pxt.build.cuda('cu/multiply.cu')
    def multiply(out: Out, a: In, b: In, **kwargs):
        pass


rs_lib = pxt.build.rust('rs/Cargo.toml')


@pxt.link.lib(rs_lib, function='i_add')
def rs_i_add(a: int, b: int) -> int:
    raise NotImplementedError


@pxt.link.lib(rs_lib, function='i_no_add')
def rs_no_add(a: int, b: int) -> int:
    return a + b
