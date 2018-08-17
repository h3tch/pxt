import contextlib
import distutils.sysconfig
import glob
import inspect
import itertools
import os
import warnings
from typing import Callable, List, Union


def get_source_list(filename: str, include_folders: List[str], find_function: Callable) -> List[str]:
    """
    Recursively extract all (include) files used by the specified file.

    Parameters
    ----------
    filename : str
        Glob filename to search for.
    include_folders : list
        Other folders in which include files can be located.
    find_function : callable
        The function used to search for include files.

    Returns
    -------
    file_list : list
        Returns a list of all files found in the recursive search.
    """
    def gather_include_files(file, existing_include_files=set()):
        # read the content of the file
        with open(file, 'r') as file_object:
            content = file_object.read()
            # use the find function to find all used (included) files
            for include_file in find_function(content):
                # search for the include files in all include
                # folders and store the found include files
                file_list = [os.path.join(d, include_file) for d in include_folders]
                file_list = [os.path.abspath(f) for f in file_list
                             if os.path.exists(f) and f not in existing_include_files]
                _ = [existing_include_files.add(f) for f in file_list]
                # recursively search in the included files for other include files
                for f in file_list:
                    gather_include_files(f, existing_include_files)

        return existing_include_files

    # search all files matching the glob file pattern
    file_paths = glob.glob(filename)
    include_sets = [gather_include_files(f) for f in glob.glob(filename)]

    # return a list of unique filenames
    return list(set(file_paths + list(itertools.chain.from_iterable(include_sets))))


def get_binary_name(binary_file: str, ext: str=None) -> str:
    """
    Convert a filename to its binary counterpart.

    Parameters
    ----------
    binary_file : str
        The filename to be converted into the binary filename.
    ext : str
        Replace the extension of the result with the specified extension.

    Returns
    -------
    binary_name : str
        The binary filename of the specified file.
    """
    # replace the file extension with the python extension suffix of the architecture
    target_file = os.path.splitext(binary_file)[0] + distutils.sysconfig.get_config_var('EXT_SUFFIX')

    # replace the suffix extension if requested
    if ext is not None:
        # make sure the extension starts with a dot
        if ext[0] != '.':
            ext = '.' + ext
        # change the extension
        target_file = os.path.splitext(target_file)[0] + ext

    return target_file


def get_module_name(module_file: str) -> str:
    """
    Get module name from the module file name.

    Parameters
    ----------
    module_file : str
        Path to the module file.

    Returns
    -------
    name : str
        Returns the name of the module.
    """
    # get the suffix of extension modules (modules can be both,
    # source files or binary files)
    suffix = distutils.sysconfig.get_config_var('EXT_SUFFIX')

    # extract the module filename
    filename = os.path.split(module_file)[1]

    # is the module a binary module (then it ends with the
    # extension suffix)
    if filename.endswith(suffix):
        return filename[:-len(suffix)]

    # get the name and the extension of the module file
    file_name, file_ext = os.path.splitext(filename)
    # get the architecture and the extension of extension modules
    suffix_arch, suffix_ext = os.path.splitext(suffix)

    # in case the module does not follow the common extension
    # naming (e.g., user uses her own file extensions)
    if filename.endswith(suffix_arch + file_ext):
        return filename[:-(len(suffix_arch) + len(file_ext))]

    # in case we cannot identify the module type
    # we return the filename up to the first dot
    dot_index = filename.index('.')
    return filename if dot_index < 0 else filename[:dot_index]


def is_called_as_decorator() -> bool:
    """
    Check if the caller is been used as a decorator.

    Returns
    -------
    bool
        Returns `True` if the function, calling `is_called_as_decorator`,
        has been called as a decorator.
    """
    # get parent frame of the caller of this function
    parent = inspect.currentframe().f_back.f_back

    # make sure the parent file is a valid filename
    # (python internal files usually fail this test)
    if not os.path.exists(parent.f_code.co_filename):
        return False

    # read the lines up to the parent frame position in the file
    with open(parent.f_code.co_filename, 'r') as fp:
        lines = fp.readlines()[:parent.f_lineno]

    # Parse the line for a function and search for the "@"
    # at the beginning of the function. If no "@" was could
    # be found, the calling function is not used as a decorator.
    lines[-1] = lines[-1].rstrip()
    balance = 0
    for r, line in enumerate(lines[::-1]):
        for c, letter in enumerate(line[::-1]):
            if letter == '(':
                balance += 1
            elif letter == ')':
                balance -= 1
            if balance == 0:
                while r+1 < len(lines) and lines[-r-2].rstrip().endswith('\\'):
                    r += 1
                return lines[-r-1].lstrip().startswith('@')

    raise SyntaxError('Could not identify whether the call is a decorator or a function.')


def recursive_file_search(root: str=None, name: str=None, ext: Union[str, List[str]]=None) -> List[str]:
    """
    Recursively search for files with the specified name and/or extension.

    Parameters
    ----------
    root : str
        The root path from which on to search recursively.
    name : str
        The filename to search for (can include the extension).
    ext : str, list
        Search for all files with this extension(s).

    Returns
    -------
    files : List[str]
        A list of all files containing the specified extension.
    """
    # If no root directory has been specified,
    # use the current wording directory.
    if root is None:
        root = os.getcwd()

    # If no name has been specified,
    # use search for all names.
    if name is None:
        name = '*'
    # Separate name and extension so we can add
    # the file name extension to the extension list.
    name, name_ext = os.path.splitext(name)

    if ext is None:
        # If no extension has been specified
        # use the file name extension (which can be empty).
        ext = [name_ext]
    elif isinstance(ext, str):
        # make sure ext is a list
        ext = [ext]

    # add the file name extension to the extension list
    if len(name_ext) > 0 or len(ext) == 0:
        ext.append(name_ext)

    # make sure every extension starts with a point
    ext = [e if len(e) == 0 or e[0] == '.' else '.' + e for e in ext]

    # search for all files and merge them into a single list
    iterator = itertools.chain(*[glob.glob(os.path.join(root, '**', name + e), recursive=True)
                                 for e in ext])
    return list(set([f for f in iterator]))


@contextlib.contextmanager
def chdir(directory: str) -> str:
    """
    Change the active working directory within a `with` code block.

    Parameters
    ----------
    directory : str
        The new working directory.

    Returns
    -------
    cwd : str
        Returns the working directory before `chdir` was called.
    """
    cwd = os.getcwd()

    try:
        os.chdir(directory)
        yield cwd
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def temporary_environ(**kwargs):
    """
    Temporarily set/change one or more environment variables for
    within a `with` code block.

    Parameters
    ----------
    kwargs
        A list of keyword arguments, where keys are the names of the
        environment variable. The respective value will be temporarily
        set for the environment variable.

    Returns
    -------
    value : dict
        Returns the current value(s) of the environment variable(s) or
        `None` if the variable does not exist yet. The keys of the
        dictionary are the same as in the provided keyword arguments.
    """
    old_values = [os.environ[name] if name in os.environ else None
                  for name, value in kwargs.items()]
    old = dict(zip(kwargs.keys(), old_values))

    try:
        for name, value in kwargs.items():
            os.environ[name] = value
        yield old
    finally:
        for name, old_value in old.items():
            if old_value is None:
                del os.environ[name]
            else:
                os.environ[name] = old_value


def function_frame(back=0):
    frame = inspect.currentframe().f_back
    parent = frame.f_back
    while back < 0:
        frame = parent
        parent = parent.f_back
        back += 1
    return frame, parent


def environ(*args):
    def boolify(s):
        s = s.lower()
        if s == 'true':
            return True
        if s == 'false':
            return False
        raise ValueError

    def autoconvert(s):
        for fn in (boolify, int, float):
            try:
                return fn(s)
            except ValueError:
                pass
        return s

    for arg in args[:-1]:
        if isinstance(arg, str) and arg in os.environ:
            return autoconvert(os.environ[arg])
    return args[-1]


def env_default(frame, variable_name, default):
    package = frame.f_globals['__name__'].replace('.', '_').upper()
    fn = '{}_{}'.format(package, frame.f_code.co_name.upper())
    env = ['{}_{}'.format(package, variable_name), '{}_{}'.format(fn, variable_name)]
    return environ(*env, default)


def fallback(ex, fallback_function, fallback_enabled):
    if fallback_enabled:
        warnings.warn(str(ex))
        return fallback_function
    raise ex
