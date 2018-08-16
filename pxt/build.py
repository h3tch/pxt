import distutils.core
import importlib.util
import inspect
import os
import platform
import re
import shutil
import sys
import tempfile
import types
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import setuptools

import pxt.helpers
import pxt.kwargs

_kw_source = 'source'
_kw_include_dirs = 'include_dirs'
_cpp_include_pattern = re.compile(r'^\s*#include\s+"[^"]*"', re.MULTILINE)
_pxt_includes = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                 np.get_include()]


def everything(root_dir: str=''):
    """
    Build everything

    Parameters
    ----------
    root_dir : str
        The root directory in which to search for pxt code to compile (default
        is the current working directory).
    """
    import glob

    # get the package name and folder of the decorated function
    caller_file = inspect.currentframe().f_back.f_locals['__file__']

    # find all python files
    expression = os.path.join(root_dir, '**/*.py')
    module_names = set([f for f in glob.glob(expression, recursive=True)])

    # prevent recursive imports (do not import the package that tries to
    # build everything, because it would result in a loop)
    module_names = [f for f in module_names if os.path.abspath(f) != caller_file]

    # find all files in which pxt is imported and might therefore be used
    module_names = [f for f in module_names if 'import pxt' in open(f).read()]
    module_names = [os.path.dirname(f) if os.path.split(f)[1] == '__init__.py' else os.path.splitext(f)[0]
                    for f in module_names]

    # convert file names to modules
    modules = [f.replace(os.path.sep, '.') for f in module_names]

    # importing the modules will compile the code
    _ = [importlib.import_module(m) for m in modules]


def rust(cargo_file: str,
         name: Optional[str]=None,
         force: bool=False,
         enable_fallback: bool=False,
         **kwargs) -> Union[Callable, str]:
    """
    Compile and build a RUST source code file into a dynamic library.

    Parameters
    ----------
    cargo_file : str
        The RUST source file to be compiled as a dynamic library.
    name : Optional[str]
        The module name. If `None` (default) the name of specified in
        `Cargo.toml` will be used. If no name is specified in `Cargo.toml`,
    force : Optional[bool]
        Fore the file to be compiled if `True`. Otherwise only compile
        if there are any changes (default `False`).
    enable_fallback : Optional[bool]
        Enable python function fallback in case of an error (default `False`).
    kwargs
        Additional arguments for the RustExtension class.

    Returns
    -------
    wrapper : Callable, str
        If called as a decorator, a wrapper function will be returned.
        Otherwise the path to the binary file will be returned.
    """

    frame, _ = pxt.helpers.function_frame()

    def wrapper(func):
        def object2binding(obj):
            # use pxt default binding
            if obj is None:
                return setuptools_rust.utils.Binding.NoBinding

            # no conversion needed
            if isinstance(obj, setuptools_rust.utils.Binding):
                return obj

            # in case of a string, convert it to a binding type
            if isinstance(obj, str):
                obj = obj.lower()
                if obj == 'pyo3':
                    return setuptools_rust.utils.Binding.PyO3
                elif obj == 'rustcpython':
                    return setuptools_rust.utils.Binding.RustCPython
                elif obj == 'nobinding':
                    return setuptools_rust.utils.Binding.NoBinding

            raise ValueError('The specified python-rust binding "{}" '
                             'is not supported.'.format(obj))

        # get the package name and folder of the decorated function
        _, parent = pxt.helpers.function_frame(0 if func is not None else -1)

        # get environment settings (they override function arguments)
        force_enabled = pxt.helpers.env_default(frame, 'FORCE', force)
        fallback_enabled = pxt.helpers.env_default(frame, 'ENABLE_FALLBACK', enable_fallback)

        # gather relevant compile path information
        _, package_folder, tmp_path = _get_build_info(parent, cargo_file)

        # Change the current working directory to the package folder.
        # This way, the build function can be used the same way by
        # different packages (the working directory is always the package).
        with pxt.helpers.chdir(package_folder):
            # Use the module name specified in the Cargo.toml file
            # if no name is provided by the user.
            namespace = name
            if namespace is None:
                # make sure toml is installed
                if importlib.util.find_spec('toml') is None:
                    raise RuntimeError('"toml" module could not be found. Please '
                                       'make sure you have "toml" installed.')
                import toml
                config = toml.load(cargo_file)
                if 'lib' in config and 'name' in config['lib']:
                    namespace = config['lib']['name']

            # get binary file path
            binary_file = pxt.helpers.get_binary_name(os.path.join(package_folder, namespace))
            result = func if func is not None else binary_file

            # only check for changes if compilation should not be forced
            if not force_enabled:
                # get source file path
                binary_file = pxt.helpers.get_binary_name(os.path.join(package_folder, namespace))
                source_files = pxt.helpers.recursive_file_search(ext=['rs', 'toml'])

                # get target and source file timestamps
                binary_timestamp = os.path.getmtime(binary_file) if os.path.exists(binary_file) else 0
                source_timestamp = [os.path.getmtime(f) for f in source_files] if len(source_files) > 0 else [0]

                # if all binary files are newer than all source files
                # there are no changes that make compilation necessary
                if binary_timestamp > max(source_timestamp):
                    return result

            # evaluate the binding keyword argument for the rust-python binding to be used
            kwargs['binding'] = object2binding(kwargs['binding'] if 'binding' in kwargs else None)

            # make sure setuptools_rust is installed
            if importlib.util.find_spec('setuptools_rust') is None:
                return pxt.helpers.fallback(RuntimeError('"setuptools_rust" module could not be found. Please '
                                            'make sure you have "setuptools-rust" installed.'),
                                            result, fallback_enabled)

            # build the rust extension
            import setuptools_rust
            extension = setuptools_rust.RustExtension(namespace, cargo_file, **kwargs)
            setuptools.setup(script_args=['build_ext', '--build-temp={}'.format(tmp_path),
                                          '--force', '--inplace'],
                             rust_extensions=[extension], zip_safe=False)

            return result

    # ether use this function as a decorator or a normal function
    return wrapper if pxt.helpers.is_called_as_decorator() else wrapper(None)


def cpp(file: str,
        force: bool=False,
        enable_fallback: bool=False,
        **kwargs) -> Union[Callable, str]:
    """
    Compile and build a C++ source code file into a dynamic library.

    Parameters
    ----------
    file : str
        The C++ source file to be compiled as a dynamic library.
    force : bool
        Fore the file to be compiled if `True`. Otherwise only compile
        if there are any changes.
    enable_fallback : Optional[bool]
        Enable python function fallback in case of an error (default `False`).
    kwargs
        Additional arguments for the Extension class.

    Returns
    -------
    wrapper : Callable, str
        If called as a decorator, a wrapper function will be returned.
        Otherwise the path to the binary file will be returned.
    """

    frame, _ = pxt.helpers.function_frame()

    # the function wrapper to be returned in case the `cpp` function is used as a decorator
    def wrapper(func):
        global _kw_include_dirs, _pxt_includes

        # get the package name and folder of the decorated function
        _, parent = pxt.helpers.function_frame(0 if func is not None else -1)

        # get environment settings (they override function arguments)
        force_enabled = pxt.helpers.env_default(frame, 'FORCE', force)
        fallback_enabled = pxt.helpers.env_default(frame, 'ENABLE_FALLBACK', enable_fallback)

        # gather relevant compile path information
        namespace, package_folder, tmp_folder = _get_build_info(parent, file)

        binary_file = pxt.helpers.get_binary_name(file)
        binary_name = os.path.split(binary_file)[1]
        dst_path = os.path.join(package_folder, binary_name)
        result = func if func is not None else dst_path

        with pxt.helpers.chdir(package_folder):
            # use keyword argument helper class
            kw_args = pxt.kwargs.KwArgs(kwargs)

            with pxt.helpers.temporary_environ(**kw_args.extract('CC', 'CXX')):

                # only check for changes if compilation should not be forced
                if not force_enabled:
                    # get target and source file paths
                    source_files = pxt.helpers.recursive_file_search(ext=['cpp', 'cxx'])

                    # get target and source file timestamps
                    binary_timestamp = os.path.getmtime(binary_file) if os.path.exists(binary_file) else 0
                    source_timestamp = [os.path.getmtime(f) for f in source_files] if len(source_files) > 0 else [0]

                    # if all binary files are newer than all source files
                    # there are no changes that make compilation necessary
                    if binary_timestamp > max(source_timestamp):
                        return result

                # add pxt include directories
                include_dirs = kw_args.append(_kw_include_dirs, _pxt_includes)
                kw_args[_kw_include_dirs] = [_module_dir(d) if isinstance(d, types.ModuleType) else d
                                             for d in include_dirs]

                # create distutils extension
                extension = distutils.core.Extension(namespace, sources=[file], **kw_args)

                # compile the extension
                if not hasattr(sys, 'argv'):
                    setattr(sys, 'argv', [os.path.basename(file)])
                distutils.core.setup(script_args=['build', '--build-base={}'.format(tmp_folder), '--force'],
                                     ext_modules=[extension], zip_safe=False)

                # copy the compiled files from the temporary
                # directory to their destination folder
                tmp_path = pxt.helpers.recursive_file_search(tmp_folder, binary_name)
                if len(tmp_path) == 0:
                    return pxt.helpers.fallback(FileNotFoundError('Could not find "{}" in the output folder "{}". '
                                                'Compilation might have failed'.format(binary_name, tmp_folder)),
                                                result, fallback_enabled)
                shutil.copyfile(tmp_path[0], dst_path)

                # A new module has been created and should be immediately importable
                # by other modules. Hence, the import caches need to be invalidated.
                importlib.invalidate_caches()

                return result

    # ether use this function as a decorator or a normal function
    return wrapper if pxt.helpers.is_called_as_decorator() else wrapper(None)


def cuda(file: str,
         force: bool=False,
         enable_fallback: bool=False,
         **kwargs) -> Optional[Callable]:
    """
    Compile a CUDA source code file into binary CUDA code.

    Examples
    --------
    Compile a file and return the file name of the binary file::py
    >>>binary_file = pxt.build.cuda('multiply.cu')

    Compile the code in 'multiply.cu' to binary code using `@pxt.build.cuda`
    and link the `multiply(a, b)` Python function to the `multiply(out, a, b)`
    CUDA kernel of the binary file 'multiply.cubin'::py
    >>>@pxt.link.cuda('multiply.cubin')
    >>>@pxt.build.cuda('multiply.cu')
    >>>def multiply(result: Out, a: In, b: In, **kwargs):
    >>>    raise NotImplementedError

    Parameters
    ----------
    file : str
        The CUDA source file to be compiled into binary CUDA code.
        Or the CUDA binary output file (`*.cubin`) in which the binary
        code should be stored.
    force : bool
        Fore the file to be compiled if `True`. Otherwise only compile
        if there are any changes.
    enable_fallback : Optional[bool]
        Enable python function fallback in case of an error (default `False`).
    kwargs
        Additional arguments for PyCuda compile method.

        source : str
            Provide the source code as a python string.
        include_dirs : list
            Additional include directories for the CUDA compiler.
        options : list
            Additional nvcc compiler options.
        no_extern_c : bool
            Do not sourround the CUDA code with 'extern "C"' (default `False`).
        arch : str
            Compile for a specific architecture.
        cache_dir : str
            Specify where to cache files of the nvcc compiler.

    Returns
    -------
    wrapper : Callable, None
        If called as a decorator, a wrapper function will be returned.
        Otherwise the path to the binary file will be returned.
    """

    frame, _ = pxt.helpers.function_frame()

    def wrapper(func):
        global _kw_include_dirs, _kw_source, _pxt_includes

        import pxt

        # get name of the binary file
        file_name, file_ext = os.path.splitext(file)
        binary_file = file if file_ext == '.cubin' else file_name + '.cubin'

        result = func if func is not None else binary_file

        # get the package name and folder of the decorated function
        _, parent = pxt.helpers.function_frame(0 if func is not None else -1)
        package_folder = os.path.dirname(parent.f_locals['__file__'])

        # get environment settings (they override function arguments)
        force_enabled = pxt.helpers.env_default(frame, 'FORCE', force)
        fallback_enabled = pxt.helpers.env_default(frame, 'ENABLE_FALLBACK', enable_fallback)

        # make sure pycuda is installed
        if importlib.util.find_spec('pycuda') is None:
            return pxt.helpers.fallback(RuntimeError('"pycuda" module could not be found. Please '
                                                     'make sure you have PyCuda installed.'),
                                        result, fallback_enabled)

        with pxt.helpers.chdir(package_folder):
            kw_args = pxt.kwargs.KwArgs(kwargs)

            # only check for changes if compilation should not be forced
            if not force_enabled and os.path.exists(file):
                # get the source file list
                inc_dirs = kw_args.append(_kw_include_dirs, _pxt_includes)
                inc_dirs = [_module_dir(d) if isinstance(d, types.ModuleType) else d for d in inc_dirs]
                kw_args[_kw_include_dirs] = inc_dirs
                source_files = pxt.helpers.get_source_list(file, inc_dirs, _find_c_include_files)

                # get target and source file timestamps
                binary_timestamp = os.path.getmtime(binary_file) if os.path.exists(binary_file) else 0
                source_timestamp = [os.path.getmtime(f) for f in source_files] if len(source_files) > 0 else [0]

                # if all binary files are newer than all source files
                # there are no changes that make compilation necessary
                if binary_timestamp > max(source_timestamp):
                    return result

            # get the source code
            if 'source' in kw_args:
                # from argument
                source = kw_args['source']
            elif os.path.exists(file):
                # from file
                with open(file, 'r') as fp:
                    source = fp.read()
            else:
                return pxt.helpers.fallback(FileNotFoundError(file), result, fallback_enabled)

            # get compute capability of the architecture
            if 'arch' in kw_args:
                # from argument
                arch = kw_args['arch']
            else:
                # get the compute capabilities of the device
                import pxt.cuda
                device = kw_args.try_get(['device', 'device_id'], None)
                arch = pxt.cuda.architecture(device)

            # compile the source code
            import pycuda.compiler
            binary = pycuda.compiler.compile(source, arch=arch,
                                             no_extern_c=kw_args.try_get('no_extern_c', False),
                                             options=kw_args.try_get('options', None),
                                             cache_dir=kw_args.try_get('cache_dir', None),
                                             include_dirs=kw_args.try_get('include_dirs', []))

            # save the binary
            with open(binary_file, 'wb') as fp:
                fp.write(binary)

            return result

    # ether use this function as a decorator or a normal function
    return wrapper if pxt.helpers.is_called_as_decorator() else wrapper(None)


def cython(file: str,
           enable_fallback: bool=False,
           **kwargs) -> Union[Callable, str]:
    """
    Compile and build a Cython source code file into a dynamic library.

    Parameters
    ----------
    file : str
        The Cython source file to be compiled as a dynamic library.
    enable_fallback : Optional[bool]
        Enable python function fallback in case of an error (default `False`).
    kwargs
        Additional arguments for the Extension class.

    Returns
    -------
    wrapper : Callable, str
        If called as a decorator, a wrapper function will be returned.
        Otherwise the path to the binary file will be returned.
    """

    frame, _ = pxt.helpers.function_frame()

    def wrapper(func):
        global _kw_include_dirs, _pxt_includes

        # get the package name and folder of the decorated function
        _, parent = pxt.helpers.function_frame(0 if func is not None else -1)

        # gather relevant compile path information
        _, package_folder, tmp_folder = _get_build_info(parent, file)
        namespace = os.path.splitext(os.path.split(file)[1])[0]
        binary_file = pxt.helpers.get_binary_name(file)
        binary_name = os.path.split(binary_file)[1]
        dst_path = os.path.join(package_folder, binary_name)
        result = func if func is not None else dst_path

        fallback_enabled = pxt.helpers.env_default(frame, 'ENABLE_FALLBACK', enable_fallback)

        # make sure Cython is installed
        if importlib.util.find_spec('Cython') is None:
            return pxt.helpers.fallback(RuntimeError('"Cython" module could not be found. Please '
                                        'make sure you have Cython installed.'),
                                        result, fallback_enabled)

        import Cython.Build

        # use keyword argument helper class
        kw_args = pxt.kwargs.KwArgs(kwargs)
        include_dirs = kw_args.append(_kw_include_dirs, _pxt_includes)

        with pxt.helpers.chdir(package_folder):

            # compile the extension
            extension = distutils.core.Extension(namespace, sources=[file], **kw_args)
            extensions = Cython.Build.cythonize(extension, include_path=include_dirs)
            setuptools.setup(script_args=['build', '--build-base={}'.format(tmp_folder)],
                             ext_modules=extensions, zip_safe=False)

            # copy the compiled files from the temporary
            # directory to their destination folder
            tmp_path = pxt.helpers.recursive_file_search(tmp_folder, binary_name)
            if len(tmp_path) == 0:
                return pxt.helpers.fallback(FileNotFoundError('Could not find "{}" in the output folder "{}". '
                                            'Compilation might have failed'.format(binary_name, tmp_folder)),
                                            result, fallback_enabled)
            shutil.copyfile(tmp_path[0], dst_path)

            # A new module has been created and should be immediately importable
            # by other modules. Hence, the import caches need to be invalidated.
            importlib.invalidate_caches()

            return result

    # ether use this function as a decorator or a normal function
    return wrapper if pxt.helpers.is_called_as_decorator() else wrapper(None)


def _find_c_include_files(source: str) -> List[str]:
    """
    Parse the specified source code for #include "..." files.

    Parameters
    ----------
    source : str
        C/C++ source code to be parsed.

    Returns
    -------
    include_files : List[str]
        Returns all include files in the source code.
    """
    global _cpp_include_pattern
    include_file_iter = _cpp_include_pattern.findall(source)
    return [include[include.index('"') + 1:-1] for include in include_file_iter]


def _get_build_info(parent_frame: types.FrameType, path: str) -> Tuple[str, str, str]:
    """
    Extract some commonly used build information from
    the provided input parameters.

    Parameters
    ----------
    parent_frame : types.FrameType
        The frame of the caller.
    path : str
        The relative or absolute path of the source file.

    Returns
    -------
    namespace : str
        The namespace of the binary module.
    package_folder : str
        The folder of the package from which the build
        function/decorator is called.
    tmp_folder : str
        The temporary folder in which build and compile
        files should be stored.
    """
    package_folder = os.path.dirname(parent_frame.f_locals['__file__'])
    package_name = parent_frame.f_locals['__package__']

    # get the absolute path of the source file
    abs_path = path if os.path.isabs(path) else os.path.abspath(os.path.join(package_folder, path))

    if not os.path.exists(abs_path):
        raise FileNotFoundError('The file {} does not exist.'.format(abs_path))

    # get the namespace fo the module that should be compiled
    module_path = os.path.splitext(abs_path)[0]
    if not module_path.startswith(package_folder):
        raise AssertionError('The source file {} has to be located in the same module'
                             'or submodule as the @cpp decorated function.'.format(module_path))
    namespace = package_name + module_path[len(package_folder):].replace(os.path.sep, '.')
    # generate path for temporary files
    if platform.system() == 'Windows':
        module_path = module_path.replace(':', '')
    elif module_path.startswith('/'):
        module_path = module_path[1:]

    tmp_folder = os.path.join(tempfile.gettempdir(), 'pxt', module_path)

    return namespace, package_folder, tmp_folder


def _module_dir(module: types.ModuleType) -> str:
    """
    Get the absolute path to the module directory.

    Parameters
    ----------
    module : ModuleType
        The module.

    Returns
    -------
    directory : str
        Returns the absolute path to the module directory.
    """
    return os.path.abspath(os.path.join(os.path.split(module.__file__)[0], os.pardir))
