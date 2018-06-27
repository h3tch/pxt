import distutils.core
import importlib.util
import inspect
import os
import platform
import re
import shutil
import tempfile
import types
import setuptools
from typing import Callable, Union, Optional

import numpy as np

import pxt.helpers
import pxt.kwargs

kw_include_dirs = 'include_dirs'

_build_cache = []

_pxt_includes = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                 np.get_include()]

_cpp_include_pattern = re.compile(r'^\s*#include\s+"[^"]*"', re.MULTILINE)


def rust(cargo_file: str, name: str=None, force: bool=False, **kwargs) -> Union[Callable, str]:
    """
    Compile and build a RUST source code file into a dynamic library.

    Parameters
    ----------
    cargo_file : str
        The RUST source file to be compiled as a dynamic library.
    name : str
        The module name. If `None` (default) the name of specified in
        `Cargo.toml` will be used. If no name is specified in `Cargo.toml`,
    force : bool
        Fore the file to be compiled if `True`. Otherwise only compile
        if there are any changes (default `False`).
    kwargs
        Additional arguments for the RustExtension class.

    Returns
    -------
    wrapper : Callable, str
        If called as a decorator, a wrapper function will be returned.
        Otherwise the path to the binary file will be returned.
    """
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

        # make sure setuptools_rust is installed
        if importlib.util.find_spec('setuptools_rust') is None:
            raise RuntimeError('"setuptools_rust" module could not be found. Please '
                               'make sure you have "setuptools-rust" installed.')

        # import setuptools
        import setuptools_rust.utils

        # get the package name and folder of the decorated function
        parent_frame = inspect.currentframe().f_back
        if func is None:
            parent_frame = parent_frame.f_back

        # gather relevant compile path information
        _, package_folder, tmp_path = _get_build_infos(parent_frame, cargo_file)

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

            # only check for changes if compilation should not be forced
            if not force:
                # get source file path
                binary_file = pxt.helpers.get_binary_name(os.path.join(package_folder, namespace))
                source_files = pxt.helpers.recursive_file_search(ext=['rs', 'toml'])

                # get target and source file timestamps
                binary_timestamp = os.path.getmtime(binary_file) if os.path.exists(binary_file) else 0
                source_timestamp = [os.path.getmtime(f) for f in source_files] if len(source_files) > 0 else [0]

                # if all binary files are newer than all source files
                # there are no changes that make compilation necessary
                if binary_timestamp > max(source_timestamp):
                    return func if func is not None else binary_file

            # evaluate the binding keyword argument for the rust-python binding to be used
            kwargs['binding'] = object2binding(kwargs['binding'] if 'binding' in kwargs else None)

            # build the rust extension
            extension = setuptools_rust.RustExtension(namespace, cargo_file, **kwargs)
            setuptools.setup(script_args=['build_ext', '--build-temp={}'.format(tmp_path),
                                          '--force', '--inplace'],
                             rust_extensions=[extension], zip_safe=False)

            return func if func is not None else binary_file

    # ether use this function as a decorator or a normal function
    return wrapper if pxt.helpers.is_called_as_decorator() else wrapper(None)


def cpp(file: str, force: bool=False, **kwargs) -> Union[Callable, str]:
    """
    Compile and build a C++ source code file into a dynamic library.

    Parameters
    ----------
    file : str
        The C++ source file to be compiled as a dynamic library.
    force : bool
        Fore the file to be compiled if `True`. Otherwise only compile
        if there are any changes.
    kwargs
        Additional arguments for the Extension class.

    Returns
    -------
    wrapper : Callable, str
        If called as a decorator, a wrapper function will be returned.
        Otherwise the path to the binary file will be returned.
    """

    # the function wrapper to be returned in case the `cpp` function is used as a decorator
    def wrapper(func):
        global kw_include_dirs, _pxt_includes

        # get the package name and folder of the decorated function
        parent = inspect.currentframe().f_back
        if func is None:
            parent = parent.f_back

        # gather relevant compile path information
        namespace, package_folder, tmp_folder = _get_build_infos(parent, file)

        with pxt.helpers.chdir(package_folder):
            # use keyword argument helper class
            kw_args = pxt.kwargs.KwArgs(kwargs)

            with pxt.helpers.temporary_environ(**kw_args.extract('CC', 'CXX')):
                binary_file = pxt.helpers.get_binary_name(file)
                binary_name = os.path.split(binary_file)[1]
                dst_path = os.path.join(package_folder, binary_name)

                # only check for changes if compilation should not be forced
                if not force:
                    # get target and source file paths
                    source_files = pxt.helpers.recursive_file_search(ext=['cpp', 'cxx'])

                    # get target and source file timestamps
                    binary_timestamp = os.path.getmtime(binary_file) if os.path.exists(binary_file) else 0
                    source_timestamp = [os.path.getmtime(f) for f in source_files] if len(source_files) > 0 else [0]

                    # if all binary files are newer than all source files
                    # there are no changes that make compilation necessary
                    if binary_timestamp > max(source_timestamp):
                        return func if func is not None else dst_path

                # add pxt include directories
                include_dirs = kw_args.append(kw_include_dirs, _pxt_includes)
                kw_args[kw_include_dirs] = [_module_dir(d) if isinstance(d, types.ModuleType) else d
                                            for d in include_dirs]

                # create distutils extension
                extension = distutils.core.Extension(namespace, sources=[file], **kw_args)

                # compile the extension
                distutils.core.setup(script_args=['build', '--build-base={}'.format(tmp_folder), '--force'],
                                     ext_modules=[extension], zip_safe=False)

                # copy the compiled files from the temporary
                # directory to their destination folder
                tmp_path = pxt.helpers.recursive_file_search(tmp_folder, binary_name)
                if len(tmp_path) == 0:
                    raise FileNotFoundError('Could not find "{}" in the output folder "{}". '
                                            'Compilation might have failed'.format(binary_name, tmp_folder))
                shutil.copyfile(tmp_path[0], dst_path)

                # A new module has been created and should be immediately importable
                # by other modules. Hence, the import caches need to be invalidated.
                importlib.invalidate_caches()

                return func if func is not None else dst_path

    # ether use this function as a decorator or a normal function
    return wrapper if pxt.helpers.is_called_as_decorator() else wrapper(None)


def cuda(file: str, force: bool=False, **kwargs) -> Optional[Callable]:
    """
    Compile a CUDA source code file into binary CUDA code.

    Parameters
    ----------
    file : str
        The CUDA source file to be compiled into binary CUDA code.
    force : bool
        Fore the file to be compiled if `True`. Otherwise only compile
        if there are any changes.
    kwargs
        Additional arguments for the Extension class.

    Returns
    -------
    wrapper : Callable, str
        If called as a decorator, a wrapper function will be returned.
        Otherwise the path to the binary file will be returned.
    """

    def wrapper(func):
        global kw_include_dirs, _pxt_includes

        # make sure pycuda is installed
        if importlib.util.find_spec('pycuda') is None:
            raise RuntimeError('"pycuda" module could not be found. Please '
                               'make sure you have PyCuda installed.')

        file_path = file if os.path.isabs(file) else file
        binary_file = os.path.splitext(file_path)[0] + '.cubin'

        # get the package name and folder of the decorated function
        parent = inspect.currentframe().f_back
        if func is None:
            parent = parent.f_back
        package_folder = os.path.dirname(parent.f_locals['__file__'])

        with pxt.helpers.chdir(package_folder):
            # make sure the cuda source file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError('The file "{}" does not exist.'.format(file_path))

            # only check for changes if compilation should not be forced
            if not force:
                # get the source file list
                include_dirs = pxt.kwargs.KwArgs(kwargs).append(kw_include_dirs, _pxt_includes)
                include_dirs = [_module_dir(d) if isinstance(d, types.ModuleType) else d
                                for d in include_dirs]
                source_files = pxt.helpers.get_source_list(file_path, include_dirs, _find_c_include_files)

                # get target and source file timestamps
                binary_timestamp = os.path.getmtime(binary_file) if os.path.exists(binary_file) else 0
                source_timestamp = [os.path.getmtime(f) for f in source_files] if len(source_files) > 0 else [0]

                # if all binary files are newer than all source files
                # there are no changes that make compilation necessary
                if binary_timestamp > max(source_timestamp):
                    return func

            # initialize cuda
            import pxt.cuda as pxt_cuda
            import pycuda.compiler
            pxt_cuda.initialize()

            # compile the source file
            with open(file_path, 'r') as fp:
                source = fp.read()
            binary = pycuda.compiler.compile(source)

            # save the binary
            with open(binary_file, 'wb') as fp:
                fp.write(binary)

            return func

    # ether use this function as a decorator or a normal function
    return wrapper if pxt.helpers.is_called_as_decorator() else wrapper(None)


def cython(file: str, **kwargs) -> Union[Callable, str]:
    """
    Compile and build a Cython source code file into a dynamic library.

    Parameters
    ----------
    file : str
        The Cython source file to be compiled as a dynamic library.
    kwargs
        Additional arguments for the Extension class.

    Returns
    -------
    wrapper : Callable, str
        If called as a decorator, a wrapper function will be returned.
        Otherwise the path to the binary file will be returned.
    """
    def wrapper(func):
        global kw_include_dirs, _pxt_includes

        # make sure Cython is installed
        if importlib.util.find_spec('Cython') is None:
            raise RuntimeError('"Cython" module could not be found. Please '
                               'make sure you have Cython installed.')

        import Cython.Build

        # get the package name and folder of the decorated function
        parent = inspect.currentframe().f_back
        if func is None:
            parent = parent.f_back

        # gather relevant compile path information
        _, package_folder, tmp_folder = _get_build_infos(parent, file)
        namespace = os.path.splitext(os.path.split(file)[1])[0]

        # use keyword argument helper class
        kw_args = pxt.kwargs.KwArgs(kwargs)
        include_dirs = kw_args.append(kw_include_dirs, _pxt_includes)

        with pxt.helpers.chdir(package_folder):
            binary_file = pxt.helpers.get_binary_name(file)
            binary_name = os.path.split(binary_file)[1]
            dst_path = os.path.join(package_folder, binary_name)

            # compile the extension
            extension = distutils.core.Extension(namespace, sources=[file], **kw_args)
            extensions = Cython.Build.cythonize(extension, include_path=include_dirs)
            setuptools.setup(script_args=['build', '--build-base={}'.format(tmp_folder)],
                             ext_modules=extensions, zip_safe=False)

            # copy the compiled files from the temporary
            # directory to their destination folder
            tmp_path = pxt.helpers.recursive_file_search(tmp_folder, binary_name)
            if len(tmp_path) == 0:
                raise FileNotFoundError('Could not find "{}" in the output folder "{}". '
                                        'Compilation might have failed'.format(binary_name, tmp_folder))
            shutil.copyfile(tmp_path[0], dst_path)

            # A new module has been created and should be immediately importable
            # by other modules. Hence, the import caches need to be invalidated.
            importlib.invalidate_caches()

            return func if func is not None else dst_path

    # ether use this function as a decorator or a normal function
    return wrapper if pxt.helpers.is_called_as_decorator() else wrapper(None)


def _find_c_include_files(source: str):
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


def _get_build_infos(parent_frame: types.FrameType, file: str):
    """
    Extract some commonly used build information from
    the provided input parameters.

    Parameters
    ----------
    parent_frame : types.FrameType
        The frame of the caller.
    file : str
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
    file_path = file if os.path.isabs(file) else os.path.abspath(os.path.join(package_folder, file))

    if not os.path.exists(file_path):
        raise FileNotFoundError('The file {} does not exist.'.format(file_path))

    # get the namespace fo the module that should be compiled
    module_path = os.path.splitext(file_path)[0]
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
    str
        Returns the absolute path to the module directory.
    """
    return os.path.abspath(os.path.join(os.path.split(module.__file__)[0], os.pardir))
