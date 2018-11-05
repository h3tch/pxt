#pragma once

#include "numpy/numpyconfig.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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


// EXPORT FUNCTION DECLARATION

#ifndef EXPORT
    #ifdef _MSC_VER
        #define EXPORT extern "C" __declspec(dllexport)
    #else
        #define EXPORT extern "C"
    #endif
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


// PYTHON GLOBAL INTERPRETER LOCK

class PyGIL
{
public:
    PyGIL();
    ~PyGIL();

private:
    PyGILState_STATE gil_state;
    bool release;
};


// NUMPY ARRAY HELPER CLASS

class NpyArray
{
public:
    NpyArray();
    NpyArray(const NpyArray& other);
    NpyArray(PyObject* o, NpyType type=NpyType::invalid, int ndim=0);
    NpyArray(NpyType type, int ndim, uint32_t dim0, ...) ;
    NpyArray(NpyType type, int ndim, npy_intp* dims, void* data=0);
    NpyArray(NpyType type, int ndim, uint32_t* dims, void* data=0);
    ~NpyArray();

    static NpyArray empty_like(const NpyArray& other);

    inline PyObject* py_return();

    inline NpyType type() const;
    inline int element_size(NpyType type) const;

    inline PyArray_Descr* dtype() const;
    inline int ndim() const;
    inline npy_intp* shape() const;
    inline int shape(int i) const;
    inline int size() const;
    inline int nbytes() const;

    template<typename T>
    inline T& get(int i0, int i1=0, int i2=0);
    inline void* data();
    inline const void* data() const;

    template<typename T>
    inline T* cast();

    template<typename T>
    inline const T* cast() const;

    int refcount() const;

private:
    void incref();
    void decref();
    void new_array(NpyType type, int ndim, npy_intp* dims, void* data=0);

private:
    PyArrayObject* numpy_array;
};


// CONVERT PXT TYPES TO PYTHON OBJECTS

// Convert C types to Python types.
inline PyObject* c2py(bool value);
inline PyObject* c2py(int8_t value);
inline PyObject* c2py(int16_t value);
inline PyObject* c2py(int32_t value);
inline PyObject* c2py(int64_t value);
inline PyObject* c2py(uint8_t value);
inline PyObject* c2py(uint16_t value);
inline PyObject* c2py(uint32_t value);
inline PyObject* c2py(uint64_t value);
inline PyObject* c2py(float value);
inline PyObject* c2py(double value);
inline PyObject* c2py(NpyArray value);
inline PyObject* c2py(PyObject* value);


// COMPILE A PYTHON OBJECT TO BE RETURNED TO PYTHON CODE FROM PXT TYPES

void args2tuple(std::vector<PyObject*>& result);

template<typename T, typename... Args>
void args2tuple(std::vector<PyObject*>& result, T value, Args... args);

template<typename... Args>
PyObject* PyResult(Args... args);


#include "./pxt.inl"
#include "./py.h"
