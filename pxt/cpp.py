import ctypes
import inspect
import typing

import numpy as np


# Map python types to C types.
py2ctype = {
    bool: ctypes.c_bool,
    int: ctypes.c_longlong,
    float: ctypes.c_double,
    str: ctypes.c_char_p,
    tuple: ctypes.py_object,
    list: ctypes.py_object,
    dict: ctypes.py_object,
    object: ctypes.py_object,
    np.ndarray: ctypes.py_object,
    ctypes.c_void_p: ctypes.c_void_p
}

# Map numpy type IDs to C types.
# Used by the `c2numpy` function.
dtypeid_ctype = {
    np.dtype(np.void).num: ctypes.c_void_p,
    np.dtype(np.bool).num: ctypes.c_bool,
    np.dtype(np.int8).num: ctypes.c_int8,
    np.dtype(np.uint8).num: ctypes.c_uint8,
    np.dtype(np.int16).num: ctypes.c_int16,
    np.dtype(np.uint16).num: ctypes.c_uint16,
    np.dtype(np.int32).num: ctypes.c_int32,
    np.dtype(np.uint32).num: ctypes.c_uint32,
    np.dtype(np.int64).num: ctypes.c_int64,
    np.dtype(np.uint64).num: ctypes.c_uint64,
    np.dtype(np.int).num: ctypes.c_int,
    np.dtype(np.uint).num: ctypes.c_uint,
    np.dtype(np.float32).num: ctypes.c_float,
    np.dtype(np.float64).num: ctypes.c_double,
}

# Map numpy type IDs to numpy types.
# Used by the `c2numpy` function.
dtypeid_dtype = {
    np.dtype(np.void).num: np.void,
    np.dtype(np.bool).num: np.bool,
    np.dtype(np.int8).num: np.int8,
    np.dtype(np.uint8).num: np.uint8,
    np.dtype(np.int16).num: np.int16,
    np.dtype(np.uint16).num: np.uint16,
    np.dtype(np.int32).num: np.int32,
    np.dtype(np.uint32).num: np.uint32,
    np.dtype(np.int64).num: np.int64,
    np.dtype(np.uint64).num: np.uint64,
    np.dtype(np.int).num: np.int,
    np.dtype(np.uint).num: np.uint,
    np.dtype(np.float32).num: np.float32,
    np.dtype(np.float64).num: np.float64,
}


def _key_or_default(kwargs, key, default):
    try:
        return kwargs[key]
    except KeyError:
        return default


class CTypeConverter(object):
    """
    A helper class to automatically convert function arguments from
    python types to C types and the result C type back into a python type.
    """

    def __init__(self, function_pointer, arg_names, arg_types, arg_default, result_type):
        """
        Constructor.

        Parameters
        ----------
        function_pointer
            A function pointer to an extension function.
        arg_types
            The argument types of the respective decorated python function.
        arg_default
            The default values of the respective arguments.
        result_type
            The result type of the respective decorated python function.
        """
        # Set the input argument and result types of the function based on
        # `arg_types` and `result_type`. This way the ctypes API takes care
        # of converting basic types between C and Python and we only need
        # to take care of the more complex types (e.g. numpy arrays).
        self._function = function_pointer

        # set input argument types
        self._function.argtypes = [py2ctype[py_type] for py_type in arg_types]
        self._arg_names = arg_names
        self._arg_default = arg_default
        self._result_type = result_type

        # set result type
        if result_type in py2ctype:
            self._function.restype = py2ctype[result_type]
        elif isinstance(result_type, typing.TupleMeta):
            self._function.restype = ctypes.py_object
        elif result_type is inspect._empty:
            self._function.restype = None
        else:
            self._function.restype = result_type

    def __call__(self, *args, **kwargs):
        if len(args) < len(self._arg_default):
            defaults = tuple(_key_or_default(kwargs, name, default)
                             for default, name in zip(self._arg_default[len(args):],
                                                      self._arg_names[len(args):]))
            args = args + defaults

        result = self._function(*args)

        if self._result_type is not inspect._empty:
            return result
