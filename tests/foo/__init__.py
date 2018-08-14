import logging
from typing import Tuple

import numpy as np
import pxt.link

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


try:
    from pxt.cuda import In, Out

    # 2) link the cuda kernel 'multiply' to the 'cu_multiply' function
    #    set a default value for the grid size
    #    pre-allocate a result object that will be returned by the function
    @pxt.link.cuda('cu/multiply.cubin', function='multiply', grid=(1, 1),
                   result_arg=0, result_object=np.empty((512,), dtype=np.float32))
    # 1) build multiply.cu -> cu/multiply.cubin
    @pxt.build.cuda('cu/multiply.cu')
    def cu_multiply(a: In, b: In, **kwargs) -> np.ndarray:
        raise NotImplementedError
except ImportError:
    pass


rs_lib = pxt.build.rust('rs/Cargo.toml')


@pxt.link.lib(rs_lib, function='i_add')
def rs_i_add(a: int, b: int) -> int:
    raise NotImplementedError


@pxt.link.lib(rs_lib, function='i_no_add')
def rs_no_add(a: int, b: int) -> int:
    return a + b
