#ifndef _PXT_H
#define _PXT_H

#include "Python.h"
#include "numpy/arrayobject.h"
#include <stddef.h>
#include <stdarg.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <type_traits>
#include <iostream>


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

#define MOD_INIT(name, methods) MOD_INIT_EX(name, methods, "")

#define MOD_METHOD_EX(func, flags, doc) {#func, func, flags, doc}

#define MOD_METHOD(func, flags) MOD_METHOD_EX(func, flags, "")


// EXPORT FUNCTION DECLARATION

#ifdef _MSC_VER
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif


// MAXIMUM NUMBER OF SUPPORTED ARRAY DIMENSIONS

const int MAX_DATA_DIMS = 3;


// SUPPORTED DATA TYPES

enum NpyType : int32_t
{
    boolean = 0,
    int8 = 1,
    uint8 = 2,
    int16 = 3,
    uint16 = 4,
    int32 = 5,
    uint32 = 6,
    Int = 7,
    Uint = 8,
    int64 = 9,
    uint64 = 10,
    float32 = 11,
    float64 = 12,
    none = 20,
};


// Numpy array helper class

class NpyArray
{
public:
    // Create an empty PyData structure.
    NpyArray() : data(0)
    {
        free();
    }

    NpyArray(PyObject* o)
    {
        PyArrayObject* a = (PyArrayObject*) PyArray_FromAny(
			o, 0, 0, 0, NPY_ARRAY_C_CONTIGUOUS, 0);

        if (a == 0)
        {
            PyErr_SetString(PyExc_ValueError, "Could not convert 'PyObject*' to 'NpyArray'.");
            return;
        }

        data = PyArray_DATA(a);
		type = (NpyType)PyArray_TYPE(a);
		ndim = PyArray_NDIM(a);
		for (int i = 0; i < (int)ndim; i++)
		    dims[i] = (uint32_t)PyArray_DIM(a, i);
		owner = false;
    }

    // Create a new PyData structure and allocate memory.
    NpyArray(NpyType type, int ndim, uint32_t* dims) : data(0)
    {
        alloc(type, ndim, dims);
    }

    // Create a new PyData structure and allocate memory.
    NpyArray(NpyType type, int ndim, uint32_t dim0, ...) : data(0)
    {
        va_list args;
        va_start(args, dim0);

        uint32_t dims[MAX_DATA_DIMS];
        dims[0] = dim0;
        for (int i = 1; i < MAX_DATA_DIMS; i++)
            dims[i] = i < ndim ? va_arg(args, uint32_t) : 0;

        va_end(args);

        alloc(type, ndim, dims);
    }

    PyObject* asPyObject()
    {
        void* ptr = data;

        // In the case of numpy owning the data, we need to copy it,
        // because PyData is only guaranteed to be valid within an
        // external function call and has no knowledge of when the owner
        // releases/free the data. Hence, returning a PyData to python
        // without owning the data can lead to segmentation faults if the
        // owner frees the memory while the PyData object is still used.
        if (!owner)
        {
            auto n = nbytes();
            ptr = std::malloc(n);
            std::memcpy(ptr, data, n);
        }

        npy_intp np_dims[NPY_MAXDIMS];
        for (uint8_t i = 0; i < ndim; i++)
            np_dims[i] = (npy_intp)dims[i];

        // create new numpy array from existing data
        auto ndarray = PyArray_SimpleNewFromData((int)ndim, np_dims, (int)type, ptr);
        // make the numpy array own the memory so we do not have to manage it
        PyArray_ENABLEFLAGS((PyArrayObject*)ndarray, NPY_OWNDATA);

        return (PyObject*)ndarray;
    }

    // Allocate memory and initialize the PyData structure.
    void alloc(NpyType type, int ndim, uint32_t* dims)
    {
        free();
        this->type = type;
        this->ndim = ndim;

        for (int i = 0; i < MAX_DATA_DIMS; i++)
            this->dims[i] = i < ndim ? dims[i] : 0;

        this->data = std::malloc(nbytes());
        this->owner = true;
    }

    // Free the allocated memory if necessary and initialize
    // the structure to be empty.
    void free()
    {
        type = NpyType::boolean;
        ndim = 0;
        for (int i = 0; i < MAX_DATA_DIMS; i++)
            dims[i] = 0;
        if (owner && data != 0)
        {
            std::free(data);
            data = 0;
        }
        owner = false;
    }

    // The size of a single element.
    int element_size()
    {
        return element_size(type);
    }

    // The size of a single element of the specified type.
    static int element_size(NpyType type)
    {
        size_t size[] = {
            sizeof(bool),
            sizeof(int8_t),
            sizeof(uint8_t),
            sizeof(int16_t),
            sizeof(uint16_t),
            sizeof(int32_t),
            sizeof(uint32_t),
            sizeof(int),
            sizeof(unsigned int),
            sizeof(int64_t),
            sizeof(uint64_t),
            sizeof(float),
            sizeof(double),
        };
        const int32_t t = (int32_t)type;
        return 0 <= t && t <= 12 ? size[t] : 0;
    }

    // The number of elements the PyData structure can hold.
    int size()
    {
        return size(ndim, dims);
    }

    // The number of elements of the specified array dimensions.
    static int size(int ndim, uint32_t* dims)
    {
        if (ndim <= 0 || ndim > MAX_DATA_DIMS)
            return 0;
        int n = 1;
        for (int i = 0; i < ndim; i++)
            n *= dims[i];
        return n;
    }

    // The number of bytes the PyData structure can hold.
    int nbytes()
    {
        return size() * element_size();
    }

    // The number of elements of the specified array.
    static int nbytes(NpyType type, int ndim, uint32_t* dims)
    {
        return size(ndim, dims) * element_size(type);
    }

    template<typename T>
    T* cast()
    {
        return (T*)data;
    }

    void* data;
    NpyType type;
    int ndim;
    uint32_t dims[MAX_DATA_DIMS];
    bool owner;
};


// CONVERT PXT TYPES TO PYTHON OBJECTS

// Convert C types to Python types.
inline PyObject* c2py(bool value) { return PyBool_FromLong((long)value); }
inline PyObject* c2py(int8_t value) { return PyLong_FromLong((long)value); }
inline PyObject* c2py(int16_t value) { return PyLong_FromLong((long)value); }
inline PyObject* c2py(int32_t value) { return PyLong_FromLong((long)value); }
inline PyObject* c2py(int64_t value) { return PyLong_FromLongLong((long long)value); }
inline PyObject* c2py(uint8_t value) { return PyLong_FromUnsignedLong((unsigned long)value); }
inline PyObject* c2py(uint16_t value) { return PyLong_FromUnsignedLong((unsigned long)value); }
inline PyObject* c2py(uint32_t value) { return PyLong_FromUnsignedLong((unsigned long)value); }
inline PyObject* c2py(uint64_t value) { return PyLong_FromUnsignedLongLong((unsigned long long)value); }
inline PyObject* c2py(float value) { return PyFloat_FromDouble((double)value); }
inline PyObject* c2py(double value) { return PyFloat_FromDouble((double)value); }
inline PyObject* c2py(PyObject* value)
{
    if (value == Py_None)
        Py_RETURN_NONE;
    else if (value == Py_True)
        Py_RETURN_TRUE;
    else if (value == Py_False)
        Py_RETURN_FALSE;
    return value;
}


// COMPILE A PYTHON OBJECT TO BE RETURNED TO PYTHON CODE FROM PXT TYPES

// This function is called by `args2tuple(vector, T, Args...)`
// once there are no arguments left to be converted into Python
// objects. It therefore ends the recursive calls of `args2tuple`.
void args2tuple(std::vector<PyObject*>& result) { }


// Convert a C and PXT types into a python object using a variable
// argument function. The function will be called recursively until
// there are no arguments left to be converted.
template<typename T, typename... Args>
void args2tuple(std::vector<PyObject*>& result, T value, Args... args)
{
    auto obj = c2py(value);
    result.push_back(obj);
    args2tuple(result, args...);
}


// Convert a list of C and PXT types into a Python compatible
// result object. The number of arguments can vary.
template<typename... Args>
PyObject* PyResult(Args... args)
{
    // convert the input arguments into python objects and store
    // them in a vector which we later convert into a python tuple
    std::vector<PyObject*> result;
    args2tuple(result, args...);

    // if there is only one object to be returned,
    // we do not need to create a tuple for it
    if (result.size() == 1)
        return result[0];

    // store the list of objects in a python tuple
    auto tuple = PyTuple_New(result.size());
    int i = 0;
    for (auto obj : result)
        PyTuple_SetItem(tuple, i++, obj);

    return tuple;
}

#endif