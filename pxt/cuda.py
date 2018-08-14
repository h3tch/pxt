import atexit
import contextlib
import os
import sys
import typing

import numpy as np
import pycuda.compiler
import pycuda.driver
import pycuda.tools

_initialized = False
_architecture = {}

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


def initialize(clear_context_caches_on_exit: bool=True):
    """
    Initialize the PyCuda runtime.
    """
    global _initialized
    if not _initialized:
        pycuda.driver.init()
        _initialized = True
        if clear_context_caches_on_exit:
            atexit.register(clear)


def clear():
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


def architecture(device=None):
    if device is None:
        device = 0

    if isinstance(device, pycuda.driver.Device):
        device_id = device.get_attribute(pycuda.driver.device_attribute.PCI_DEVICE_ID)
    else:
        initialize()
        device_id = device
        device = pycuda.driver.Device(device)

    if device_id in _architecture:
        return _architecture[device_id]

    major, minor = device.compute_capability()
    arch = 'sm_{}{}'.format(major, minor)

    # cache
    _architecture[device_id] = arch

    return arch


@contextlib.contextmanager
def activate(context):
    not_current = context.get_current() != context
    if not_current:
        context.push()
    try:
        yield context
    finally:
        if not_current:
            context.pop()


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
    def __init__(self, context, module, func, block, grid, arg_types, result_memory=None, result_arg=0):
        # Check whether any input type needs to be converted before being passed to the
        # external library function and get the respective converter function object.
        self._arg2cu = [py2cu[py_type] if py_type in py2cu else None for py_type in arg_types]

        self._context = context
        self._module = module
        self._function = func

        self._block = block
        self._grid = grid

        self._result_arg = result_arg
        self._result_memory = result_memory
        self._gpu_result_memory = None if result_memory is None else pycuda.driver.Out(result_memory)

    def __call__(self, *args, **kwargs):
        # if the creation context is not current make it
        # current so we can call the kernel function
        is_current = self._context.get_current() == self._context
        if not is_current:
            self._context.push()

        # convert input arguments to c compatible types
        cu_args = [arg if arg2cu is None else arg2cu(arg) for arg2cu, arg in zip(self._arg2cu, args)]

        if self._gpu_result_memory is not None:
            cu_args.insert(self._result_arg, self._gpu_result_memory)

        # set default values for block and grid size if not provided by the user
        if self._block is not None and 'block' not in kwargs:
            kwargs['block'] = self._block
        if self._grid is not None and 'grid' not in kwargs:
            kwargs['grid'] = self._grid

        # call the cuda function
        result = self._function(*cu_args, **kwargs)

        # if we changed the context we need to undo our changes
        if not is_current:
            self._context.pop()

        # return the result object if used
        if self._result_memory is not None:
            return self._result_memory

        return result
