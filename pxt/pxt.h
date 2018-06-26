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
    invalid = -1,
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
    NpyArray() : numpy_array(0), owner(false)
    {
    }

    NpyArray(const NpyArray& other) : numpy_array(other.numpy_array), owner(false)
    {
    }

    NpyArray(PyObject* o, NpyType type=NpyType::invalid, int ndim=0) : NpyArray()
    {
        PyArray_Descr* dtype = type < 0 ? 0 : PyArray_DescrFromType((int)type);
        numpy_array = (PyArrayObject*)PyArray_FromAny(o, dtype, ndim, ndim, NPY_ARRAY_C_CONTIGUOUS, 0);
    }

    NpyArray(NpyType type, int ndim, uint32_t* dims) : NpyArray()
    {
        npy_intp np_dims[NPY_MAXDIMS];
        for (uint8_t i = 0; i < ndim; i++)
            np_dims[i] = (npy_intp)dims[i];

        numpy_array = (PyArrayObject*)PyArray_SimpleNew(ndim, np_dims, (int)type);
        owner = true;
    }

    NpyArray(NpyType type, int ndim, npy_intp* dims) : NpyArray()
    {
        numpy_array = (PyArrayObject*)PyArray_SimpleNew(ndim, dims, (int)type);
        owner = true;
    }

    NpyArray(NpyType type, int ndim, uint32_t dim0, ...) : NpyArray()
    {
        va_list args;
        va_start(args, dim0);

        npy_intp np_dims[MAX_DATA_DIMS];
        np_dims[0] = dim0;
        for (int i = 1; i < MAX_DATA_DIMS; i++)
            np_dims[i] = i < ndim ? (npy_intp)va_arg(args, uint32_t) : 0;

        va_end(args);

        numpy_array = (PyArrayObject*)PyArray_SimpleNew(ndim, np_dims, (int)type);
        owner = true;
    }

    ~NpyArray()
    {
        if (owner)
            PyArray_XDECREF(numpy_array);
        numpy_array = 0;
        owner = false;
    }

    static NpyArray empty_like(NpyArray& other)
    {
        return NpyArray(other.type(), other.ndim(), other.shape());
    }

    inline PyObject* py_return()
    {
        if (owner)
            Py_XINCREF((PyObject*)numpy_array);
        return (PyObject*)numpy_array;
    }

    inline NpyType type()
    {
        return (NpyType)PyArray_TYPE(numpy_array);
    }

    inline PyArray_Descr* dtype()
    {
        return PyArray_DTYPE(numpy_array);
    }

    inline int ndim()
    {
        return PyArray_NDIM(numpy_array);
    }

    inline npy_intp* shape()
    {
        return PyArray_SHAPE(numpy_array);
    }

    inline int shape(int i)
    {
        return (int)PyArray_DIM(numpy_array, i);
    }

    template<typename T>
    inline T& get(int i0, int i1=0, int i2=0)
    {
        npy_intp ptr[MAX_DATA_DIMS];
        ptr[0] = i0;
        ptr[1] = i1;
        ptr[2] = i2;
        return *(T*)PyArray_GetPtr(numpy_array, ptr);
    }

    inline int element_size(NpyType type)
    {
        return (int)PyArray_ITEMSIZE(numpy_array);
    }

    inline int size()
    {
        return (int)PyArray_SIZE(numpy_array);
    }

    inline int nbytes()
    {
        return (int)PyArray_NBYTES(numpy_array);
    }

    inline void* data()
    {
        return PyArray_DATA(numpy_array);
    }

    template<typename T>
    inline T* cast()
    {
        return (T*)PyArray_DATA(numpy_array);
    }

    int refcount()
    {
        return numpy_array ? PyArray_REFCOUNT(numpy_array) : 0;
    }

private:
    PyArrayObject* numpy_array;
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
inline PyObject* c2py(NpyArray value) { return value.py_return(); }
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

#define PyReturnOnErr(return_value) { if (PyErr_Occurred()) { return return_value; } }

#endif