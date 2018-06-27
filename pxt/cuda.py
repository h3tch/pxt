import atexit
import os
import sys
import typing

import numpy as np
import pycuda.compiler
import pycuda.driver
import pycuda.tools

_context = None

# Python function types for cuda.
In = typing.Union[pycuda.driver.In, np.ndarray]
Out = typing.Union[pycuda.driver.Out, np.ndarray]
InOut = typing.Union[pycuda.driver.InOut, np.ndarray]

# Map python types to pycuda.
py2cu = {
    In: pycuda.driver.In,
    Out: pycuda.driver.Out,
    InOut: pycuda.driver.InOut,
}


def initialize(auto_free_on_exit: bool=True, enable_profiling: bool=False):
    """
    Initialize the PyCuda runtime.

    Parameters
    ----------
    auto_free_on_exit : bool
        Automatically uninitialize CUDA on exit.
    enable_profiling : bool
        Initialize CUDA to be used with the System Profiler.
    """
    global _context
    if _context is None:
        pycuda.driver.init()
        device = pycuda.driver.Device(0)
        flags = pycuda.driver.ctx_flags
        _context = device.make_context(flags.SCHED_AUTO | flags.MAP_HOST)
        if auto_free_on_exit:
            atexit.register(free_context)

    if enable_profiling:
        start_profiler(auto_free_on_exit)


def free_context():
    """
    Free the PyCuda runtime.
    """
    global _context
    _context.pop()
    _context = None
    pycuda.tools.clear_context_caches()


def start_profiler(auto_stop_on_exit: bool=True,
                   lib_tools_injection_32: str=None,
                   lib_tools_injection_64: str=None):
    """
    Start profiling the application.

    Parameters
    ----------
    auto_stop_on_exit : bool
        Automatically stop profiling on exit. Note that stopping the profiling
        might be necessary in for the System Profiler to show the results.
    lib_tools_injection_32 : str
        Path to the tools injection 32 bit dynamic library.
    lib_tools_injection_64 : str
        Path to the tools injection 64 bit dynamic library.
    """

    # Initialize CUDA to be used with the System Profiler.

    if sys.platform == 'linux':
        if lib_tools_injection_32 is None:
            lib_tools_injection_32 = '/opt/nvidia/system_profiler/libToolsInjection32.so'
        if lib_tools_injection_64 is None:
            lib_tools_injection_64 = '/opt/nvidia/system_profiler/libToolsInjection64.so'
    else:
        raise NotImplementedError('nVidia system profiler support has not been '
                                  'implemented for the {} platform'.format(sys.platform))

    if os.path.isfile(lib_tools_injection_32) and 'CUDA_INJECTION32_PATH' not in os.environ:
        os.environ['CUDA_INJECTION32_PATH'] = lib_tools_injection_32
    if os.path.isfile(lib_tools_injection_64) and 'CUDA_INJECTION64_PATH' not in os.environ:
        os.environ['CUDA_INJECTION64_PATH'] = lib_tools_injection_64

    pycuda.driver.start_profiler()
    if auto_stop_on_exit:
        atexit.register(stop_profiler)


def stop_profiler():
    """
    Stop profiling the applications. Note that stopping the profiling
    might be necessary in for the System Profiler to show the results.
    """
    pycuda.driver.stop_profiler()


class BinModule(pycuda.compiler.CudaModule):
    """
    Creates a Module from a single binary cuda source object
    linked against the static CUDA runtime.
    """
    def __init__(self, cubin, arch=None):
        self._check_arch(arch)
        self.module = pycuda.driver.module_from_buffer(cubin)
        self._bind_module()


class CudaFunction(object):
    """
    A helper class to automatically convert function arguments from
    python types to pycuda compatible objects.
    """
    def __init__(self, module, func, arg_types):
        # Check whether any input type needs to be converted before being passed to the
        # external library function and get the respective converter function object.
        self._arg2cu = [py2cu[py_type] if py_type in py2cu else None for py_type in arg_types]

        self._module = module
        self._function = func

    def __call__(self, *args, **kwargs):
        # convert input arguments to c compatible types
        cu_args = [arg if arg2cu is None else arg2cu(arg) for arg2cu, arg in zip(self._arg2cu, args)]

        # call the cuda function
        return self._function(*cu_args, **kwargs)
