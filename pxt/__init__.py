import importlib.util

import pxt.build
import pxt.helpers


def imp(filename, module_name=None):
    if module_name is None:
        module_name = pxt.helpers.get_module_name(filename)

    spec = importlib.util.spec_from_file_location(module_name, filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
