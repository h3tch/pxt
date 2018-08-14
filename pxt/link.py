import ctypes
import glob
import importlib.util
import inspect
import os
import sys
from typing import Callable

import pxt.cpp
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


def lib(library: str, **kwargs) -> Callable:
    """
    This function decorator links the decorated function to a external
    library function. When the decorated function is called, the external
    function will be used instead. If no suitable library function could
    be found, the decorated function itself will be used. In case the
    ``raise_exception`` argument is set to ``True`` a ```` will be
    raised if no library function could be found.

    Parameters
    ----------
    library : str
        The relative or absolute path to the dynamic link library (Windows *.dll)
        or shared object (Linux *.so). Wildcards are supported. In case multiple
        libraries fulfill the search condition, the decorator will search for a
        linkable function in all of them in no specific order. The first function
        that fits to the name of the decorated function (or the name specified via
        the optional ``function_name`` argument) will be linked.
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

        # CHECK FOR SIMPLE FUNCTION REPLACEMENT

        kw_args = pxt.kwargs.KwArgs(kwargs)
        replacement = kw_args.try_get('replacement')
        if replacement is not None:
            return replacement

        # PARSE INPUT PARAMETERS OF THE DECORATOR

        func_name = kw_args.try_get(['function', 'function_name'], func.__name__)
        raise_exception = kw_args.try_get(['exception', 'raise_exception'], False)
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

        # raise an exception if requested
        if raise_exception:
            raise ValueError('Function "{}" could not be found in "{}".'.format(func_name, library_path))

        # no suitable function could be found so continue using the decorated
        # function (it will not be replaced by a library function)
        return func

    return wrapper


def cuda(binary_file: str, **kwargs) -> Callable:
    def wrapper(func):
        # make sure pycuda is installed
        if importlib.util.find_spec('pycuda') is None:
            raise RuntimeError('"pycuda" module could not be found. Please '
                               'make sure you have PyCuda installed.')

        # get the package name and folder of the decorated function
        parent = inspect.currentframe().f_back
        if func is None:
            parent = parent.f_back
        package_folder = os.path.dirname(parent.f_locals['__file__'])

        import pxt

        with pxt.helpers.chdir(package_folder):

            kw_args = pxt.kwargs.KwArgs(kwargs)

            # get the context to which the kernel should be linked
            # and make sure the context is active
            context = kw_args.try_get(['ctx', 'context'], None)
            if context is None:
                # get the device to which the kernel should be linked
                device = kw_args.try_get(['dev', 'device'], None)
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
            binary_code = kw_args.try_get('binary', None)
            if binary_code is None:
                # make sure the cuda source file exists
                if not os.path.exists(binary_file):
                    raise FileNotFoundError('The file "{}" does not exist.'.format(binary_file))
                with open(binary_file, 'rb') as fp:
                    binary_code = fp.read()

            # get the function name of the module
            func_name = pxt.kwargs.KwArgs(kwargs).try_get(['func', 'function', 'function_name'], func.__name__)

            import pxt.cuda
            mod = pxt.cuda.BinModule(binary_code)
            fn = mod.get_function(func_name)

            # the context is no longer needed so we
            # restore the original state
            context.pop()

            sig = inspect.signature(func)
            arg_types = [sig.parameters[name].annotation for name in sig.parameters]

            block = kw_args.try_get('block', None)
            grid = kw_args.try_get('grid', None)
            result_object = kw_args.try_get(['result_object', 'return_memory'], None)
            result_arg = kw_args.try_get('result_arg', 0)

            return pxt.cuda.CudaFunction(context, mod, fn, block, grid, arg_types, result_object, result_arg)

    return wrapper
