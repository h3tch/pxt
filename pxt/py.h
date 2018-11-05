#pragma once


// PYTHON MODULE INITIALIZATION

#if PY_MAJOR_VERSION >= 3

    #define MOD_INIT_EX(name, methods, doc) PyMODINIT_FUNC PyInit_##name(void) \
    { \
        static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, \
            #name,   /* module name */ \
            doc,     /* module doc */ \
            -1,      /* module size */ \
            methods, /* module methods */ \
            NULL,    /* module reload */ \
            NULL,    /* module traverse */ \
            NULL,    /* module clear */ \
            NULL,    /* module free */ \
        }; \
        PyObject* module = PyModule_Create(&moduledef); \
        if (module == NULL) \
            return NULL; \
        import_array(); \
        return module; \
    }

#else

    #define MOD_INIT_EX(name, methods, doc) PyMODINIT_FUNC init##name(void) \
    { \
        PyObject* module = Py_InitModule3(#name, methods, doc); \
        if (module == NULL) \
            return NULL; \
        import_array(); \
        return module; \
    }

#endif


///
/// \fn      void MOD_INIT_EX(token name, array methods)
///
/// \brief   Make this shared object a Python module.
///
/// \detail  Provide the necessary Python initialization functions
///          to make functions marked with `EXPORT` available to
///          the python setuptools.
///
#define MOD_INIT(name, methods) MOD_INIT_EX(name, methods, "")


///
/// \fn      array_element MOD_METHOD_EX(function func, int flags, const char* doc)
///
/// \brief   Specify a function that can be imported by the Python interpreter.
///
#define MOD_METHOD_EX(func, flags, doc) {#func, func, flags, doc}


///
/// \fn      array_element MOD_METHOD(function func, int flags)
///
/// \brief   Specify a function that can be imported by the Python interpreter.
///
#define MOD_METHOD(func, flags) MOD_METHOD_EX(func, flags, "")


// Utility macro to return a value and exit the function in case an error occurred.

///
/// \fn      PyReturnOnErr(T return_value)
///
/// \brief   Tell python that an error occurred and return the
///          specified value (`return return_value`).
///
#define PyReturnOnErr(return_value) \
{ \
    if (PyErr_Occurred()) \
        return return_value; \
}


///
/// \fn      PyReturnErr(err_type, const char* err_message, T return_value)
///
/// \brief   Tell python that an error occurred and return the
///          specified value (`return return_value`).
///
#define PyReturnErr(err_type, err_message, return_value) \
{ \
    PyErr_SetString(err_type, err_message); \
    return return_value; \
}
