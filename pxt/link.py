import ctypes
import glob
import importlib.util
import inspect
import os
import sys
from typing import Callable, Optional

import pxt.cpp
import pxt.helpers
import pxt.kwargs

# The library cache is use to cache the link
# libraries that have already been loaded.
_library_cache = dict()


def mod(module, **kwargs) -> Callable:
    """
    This function decorator links the decorated function to a external
    library function. When the decorated function is called, the external
    function will be used instead. If no suitable library function could
    be found, the decorated function itself will be used. In case the
    ``raise_exception`` argument is set to ``True`` a ```` will be
    raised if no library function could be found.

    Parameters
    ----------
    module : ModuleType
        The module in which the implementation of the function can be found.
    kwargs
        function_name : str, optional
            Provide the name of the function within the specified library.
            If not specified, the name of the decorated function will be used.
        function : str, optional
            Same as ``function_name``.
        replacement : Callable
            Replace the decorated function with the specified "replacement".
        raise_exception : bool, optional
            Raise an exception in case not suitable library function
            could be found (default ``False``).

    Returns
    -------
    wrapper : Callable
        The function wrapper return by the decorator function. See python decorator
        function specification for detailed information how decorators work.
    """
    return lib(module.__file__, **kwargs)


def lib(library: str,
        enable_fallback: bool=False,
        **kwargs) -> Callable:
    """
    This function decorator links the decorated function to a external
    library function. When the decorated function is called, the external
    function will be used instead. If no suitable library function could
    be found, the decorated function itself will be used.

    Parameters
    ----------
    library : str
        The relative or absolute path to the dynamic link library (Windows *.dll)
        or shared object (Linux *.so). Wildcards are supported. In case multiple
        libraries fulfill the search condition, the decorator will search for a
        linkable function in all of them in no specific order. The first function
        that fits to the name of the decorated function (or the name specified via
        the optional ``function_name`` argument) will be linked.
    enable_fallback : bool
        Enable python function fallback in case of an error (default `False`).
    kwargs
        function : str, optional
            Provide the name of the function within the specified library.
            If not specified, the name of the decorated function will be used.
        function_name : str, optional
            Same as `function`.
        replacement : Callable
            Replace the decorated function with the specified "replacement".

    Returns
    -------
    wrapper : Callable
        The function wrapper return by the decorator function. See python decorator
        function specification for detailed information how decorators work.
    """

    frame, _ = pxt.helpers.function_frame()

    def wrapper(func):
        # get the dll cache
        global _library_cache

        def find_function(path, name):
            """
            Find the first function in the list of matching library names.
            """
            # for all matching files
            for so in glob.glob(path):
                so = os.path.abspath(so)
                # if the library has not been loaded,
                # initialize it and add it to the library cache
                if so not in _library_cache:
                    dylib = ctypes.cdll.LoadLibrary(so)
                    _library_cache[so] = dylib
                # if the library has already been loaded, use the cached version
                else:
                    dylib = _library_cache[so]

                # find the function in the libray
                if hasattr(dylib, name):
                    return getattr(dylib, name)

        def _get_library_path():
            if func.__module__ == __name__:
                directory = os.path.split(__file__)[0]
            elif os.path.isabs(library):
                directory = os.path.split(library)[0]
            else:
                directory = sys.modules[func.__module__].__path__[0]
            return os.path.join(directory, library)

        # get environment settings (they override function arguments)
        fallback_enabled = pxt.helpers.env_default(frame, 'ENABLE_FALLBACK', enable_fallback)

        # CHECK FOR SIMPLE FUNCTION REPLACEMENT

        kw_args = pxt.kwargs.KwArgs(kwargs)
        replacement = kw_args.try_get('replacement')
        if replacement is not None:
            return replacement

        # PARSE INPUT PARAMETERS OF THE DECORATOR

        func_name = kw_args.try_get(['function', 'function_name'], func.__name__)
        library_path = _get_library_path()

        # FIND THE FIRST FUNCTION IN THE LIST OF MATCHING LIBRARY NAMES

        function_pointer = find_function(library_path, func_name)

        # WRAP THE FUNCTION WITH THE CTypeConverter TO AUTOMATICALLY CONVERT BETWEEN PYTHON AND C TYPES

        if function_pointer is not None:
            sig = inspect.signature(func)
            # Extract the argument types of the function.
            arg_types = [sig.parameters[name].annotation for name in sig.parameters]
            arg_default = tuple(sig.parameters[name].default for name in sig.parameters)
            # Extract the return type of the function.
            result_type = sig.return_annotation
            return pxt.cpp.CTypeConverter(function_pointer, arg_types, arg_default, result_type)

        # No suitable function could be found.
        # Raise an exception or continue using the decorated
        # function (it will not be replaced by a library function).
        return pxt.helpers.fallback(ValueError('Function "{}" could not be found in "{}".'
                                               .format(func_name, library_path)),
                                    func, fallback_enabled)

    return wrapper


def cuda(binary_file: Optional[str]=None,
         enable_fallback: bool=False,
         **kwargs) -> Callable:
    """
    Link the decorated function to a CUDA kernel function that can be
    executed like a normal Python function. Input, output and in/out
    type qualifiers can be used to specify how memory should be transferred
    between host and device. In case GPU memory is use not additional
    memory allocation will be performed.

    Parameters
    ----------
    binary_file : Optional[str]
        Path to the binary CUDA file. Optional, if the code is provided
        via the `binary` argument.
    enable_fallback : bool
        Enable python function fallback in case of an error (default `False`).
    kwargs
        function : str
            In case the kernel function and decorated function do not have
            the same name
        context : Context
            The PyCuda device context in which the kernel will be executed.
            If not provided, a new context will be created for the device
            specified by `device` or the default device otherwise.
        device : Device
            The PyCuda device for which a context should be created if
            it is not provided as a argument via `context` or `ctx`.
        binary : Optional[bytes]
            The binary code as a byte array.
        return_arg : int
            A index of the argument to be returned by the function.
        return_args : list
            A list of indices of the arguments to be returned by the function.
        return_mem : np.ndarray, list
            A numpy array or list of numpy arrays which will receive the ouput
            of the linked function and which will be returned by it.
        func : str
            Same as `function`.
        function_name : str
            Same as `function`.
        ctx :
            Same as `context`.
        dev :
            Same as `device`.

    Returns
    -------
    fn : Callable
        Returns a function that executes the CUDA kernel.

    Examples
    --------
    * Compile a CUDA file and link the python function to the CUDA kernel
      with the same name::py
      >>>@pxt.link.cuda('multiply.cubin')
      >>>@pxt.build.cuda('multiply.cu')
      >>>def multiply(result: Out, a: In, b: In, **kwargs):
      >>>    raise NotImplementedError

    * Link the python function to the CUDA kernel with a different name::py
      >>>@pxt.link.cuda('multiply.cubin', function='multiply')
      >>>@pxt.build.cuda('multiply.cu')
      >>>def cu_multiply(result: Out, a: In, b: In, **kwargs):
      >>>    raise NotImplementedError

    * Compile a file and return an argument of the kernel as the result. Multiple
      return arguments are also supported by providing a list or tuple of integers.
      The code assumes the CUDA kernel function has the following signature
      `__global__ void multiply(float *result, float *a, float *b)`::py
      >>>@pxt.link.cuda('multiply.cubin', return_arg=0, return_mem=np.empty((256,), np.float32))
      >>>@pxt.build.cuda('multiply.cu')
      >>>def multiply(a: In, b: In, **kwargs):
      >>>    raise NotImplementedError

    * Use existing binary kernel code::py
      >>>with open(binary_file, 'rb') as fp:
      >>>    binary_code = fp.read()
      >>>@pxt.link.cuda(binary=binary_code)
      >>>def multiply(result: Out, a: In, b: In, **kwargs):
      >>>    raise NotImplementedError

    * Compile a CUDA file and link the python function to the CUDA kernel,
      but create the context for device number 1::py
      >>>@pxt.link.cuda('multiply.cubin', device=1)
      >>>@pxt.build.cuda('multiply.cu')
      >>>def multiply(result: Out, a: In, b: In, **kwargs):
      >>>    raise NotImplementedError

    * Compile a CUDA file and link the python function to the CUDA kernel,
      but use an existing CUDA context::py
      >>>@pxt.link.cuda('multiply.cubin', context=existing_context)
      >>>@pxt.build.cuda('multiply.cu')
      >>>def multiply(result: Out, a: In, b: In, **kwargs):
      >>>    raise NotImplementedError
    """

    frame, _ = pxt.helpers.function_frame()

    def wrapper(func):
        import pxt

        # get the package name and folder of the function
        _, parent = pxt.helpers.function_frame()

        # get environment settings (they override function arguments)
        fallback_enabled = pxt.helpers.env_default(frame, 'ENABLE_FALLBACK', enable_fallback)

        # make sure pycuda is installed
        if importlib.util.find_spec('pycuda') is None:
            return pxt.helpers.fallback(RuntimeError('"pycuda" module could not be found. Please '
                                                     'make sure you have PyCuda installed.'),
                                        func, fallback_enabled)

        # get the package name and folder of the decorated function
        package_folder = os.path.dirname(parent.f_locals['__file__'])

        with pxt.helpers.chdir(package_folder):

            kw_args = pxt.kwargs.KwArgs(kwargs)

            # get the context to which the kernel should be linked
            # and make sure the context is active
            context = kw_args.try_get(['ctx', 'context'])
            if context is None:
                # get the device to which the kernel should be linked
                device = kw_args.try_get(['dev', 'device'])
                if device is None:
                    import pxt.cuda
                    import pycuda.driver
                    pxt.cuda.initialize()
                    device = pycuda.driver.Device(0)
                flags = kw_args.try_get('flags', 0)
                context = device.make_context(flags)
            else:
                context.push()

            # get the binary code of the module
            binary_code = kw_args.try_get('binary')
            if binary_code is None:
                # make sure the cuda source file exists
                if not os.path.exists(binary_file):
                    return pxt.helpers.fallback(FileNotFoundError('The file "{}" does not exist.'.format(binary_file)),
                                                func, fallback_enabled)
                with open(binary_file, 'rb') as fp:
                    binary_code = fp.read()

            # Get the function name of the module.
            # By default the name of the decorated function will be used
            func_name = pxt.kwargs.KwArgs(kwargs).try_get(['func', 'function', 'function_name'])
            if func_name is None:
                if func is None:
                    return pxt.helpers.fallback(AttributeError('If not used as a decorator,'
                                                               'a function name must be provided.'),
                                                func, fallback_enabled)
                func_name = func.__name__

            import pxt.cuda
            mod = pxt.cuda.BinModule(binary_code)
            fn = mod.get_function(func_name)

            # the context is no longer needed so we
            # restore the original state
            context.pop()

            sig = inspect.signature(func)
            arg_types = [sig.parameters[name].annotation for name in sig.parameters]

            block = kw_args.try_get('block')
            grid = kw_args.try_get('grid')
            returns = kw_args.try_get('returns')

            return pxt.cuda.CudaFunction(context, mod, fn, block, grid, arg_types, returns)

    return wrapper
