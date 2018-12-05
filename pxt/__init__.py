import importlib.util
import os

import pxt.build
import pxt.cpp
import pxt.helpers
import pxt.kwargs
import pxt.link


def imp(filename, module_name=None):
    """
    Import the specified module file under the provided module name.

    Parameters
    ----------
    filename : str
        The location of the module.
    module_name : str
        The module name under which the module will be imported.

    Returns
    -------
    module : ModuleType
        Returns the imported module.
    """
    if module_name is None:
        module_name = pxt.helpers.get_module_name(filename)

    spec = importlib.util.spec_from_file_location(module_name, filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def module_path(module):
    return os.path.split(module.__file__)[0]
