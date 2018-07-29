.. pxt documentation master file, created by
   sphinx-quickstart on Sun Jul 15 18:09:19 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

*pxt* is a small python package to easily compile code of other languages
and link them to python code in a simple and flexible manner.

It currently supports *C++*, *RUST* and *CUDA* as extensions. A simple use-case
of *pxt* is the following:

Imagine you have written a simple function in Python to add two numbers

.. code-block:: python

   def add(a, b):
      return a + b

and at some point in you decide you wanted to replace it with a more performant
version. There are other Python tools, like `Pythran <https://pythran.readthedocs.io/en/latest/>`,
or `numba <https://numba.pydata.org>`, which convert python code to other languages
(usually *C++* extensions). These tools are already quite major and if you are just
concerned with speeding up some code parts, they are certainly easier to use than
*pxt*. However, sometimes, you need more control over the generated code for
optimization reasons or because you need to use a specific API (e.g. CUDA) that is
not supported by other Python tools.

*pxt* uses a different approach. It gives you more control over how you want to
replace a function. For instance, to replace it with an extension function in
`code.so` having the same name, you use the following decorator:

.. code-block:: python

   @pxt.link.mod('code.so')
   def add(a: int, b: int) -> int:
      raise RuntimeError('Function `add` was not linked to a module.')
      return a + b

Note that type aliases were added to the python function. This is necessary for *pxt*
to correctly convert the input arguments to c-types, which can be passed to the extension
function. The respective C source code `source.cpp` looks as follows:

.. code-block:: C

   #include "pxt/pxt.h"

   EXPORT int add(int a, int b)
   {
      return a + b;
   }

   MOD_INIT(code, NULL)

We could also make this function compatible with `numpy` and, just to demonstrate
the functionality, give it a different name than the Python function:

.. code-block:: C

   #include "pxt/pxt.h"

   EXPORT PyObject* np_add(PyObject* A, PyObject* B)
   {
      // We need to lock the Python GIL if we use numpy memory.
      PyGIL lock();

      // Use the pxt NpyArray helper class to deal with numpy objects.
      NpyArray a(A);
      NpyArray b(B);
      NpyArray c(a.type(), a.ndim(), a.shape());

      // add
      int ptr_a = a.cast<int>();
      int ptr_b = b.cast<int>();
      int ptr_c = c.cast<int>();
      int n = a.size();

      for (int i = 0; i < n; i++)
         ptr_c[i] = ptr_a[i] + ptr_b[i];

      // return c
      return c.py_return();
   }

   MOD_INIT(code, NULL)

And the respective Python code:

.. code-block:: python

   @pxt.link.mod('code.so', function='np_add')
   def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
      raise RuntimeError('Function `add` was not linked to a module.')
      return a + b


In case `code.so` is not already compiled, *pxt* provides build functions
to compile source code, namely:

.. code-block:: python

   pxt.build.cpp('code.cpp')
   pxt.build.rust('code.cpp')
   pxt.build.cuda('code.cpp')
   pxt.build.cython('code.cpp')

*pxt* also provides a function to import the compiled binary file. This can
be necessary as extensions can include architecture details like
`code.cp36-win_amd64.pyd`.

.. code-block:: python

   binary_file = pxt.build.cpp('code.cpp')
   pxt.imp(binary_file, 'code.code')


As previously mentioned, *pxt* can also deal with RUST and CUDA extensions.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pxt package <pxt>
   Samples <samples>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
