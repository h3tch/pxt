#pragma once


///
/// \brief  Lock the Python global interpreter lock (GIL).
///
PyGIL::PyGIL() : gil_state()
{
    this->release = PyGILState_Check() == 0;
    if (this->release)
        this->gil_state = PyGILState_Ensure();
}


///
/// \brief  Unlock the Python global interpreter lock (GIL).
///
PyGIL::~PyGIL()
{
    if (this->release)
        PyGILState_Release(this->gil_state);
}


///
/// \brief  Create an invalid NpyArray that does not point to a numpy array.
///
NpyArray::NpyArray() : numpy_array(nullptr)
{
}


///
/// \brief  Create a new reference to a numpy array.
///
NpyArray::NpyArray(const NpyArray& other)
    : numpy_array(other.numpy_array)
{
    this->incref();
}


///
/// \brief  Create a new NpyArray using the data of a PyObject*
///         that is a numpy array.
///
NpyArray::NpyArray(PyObject* o, NpyType type, int ndim) : NpyArray()
{
    PyArray_Descr* dtype = type < 0 ? 0 : PyArray_DescrFromType((int)type);
    numpy_array = (PyArrayObject*)PyArray_FromAny(o, dtype, ndim, ndim,
                                                  NPY_ARRAY_C_CONTIGUOUS, 0);
}


///
/// \brief  Instantiate a new NpyArray and the respective numpy array.
///
NpyArray::NpyArray(NpyType type, int ndim, uint32_t dim0, ...) : NpyArray()
{
    va_list args;
    va_start(args, dim0);

    npy_intp np_dims[MAX_DATA_DIMS];
    np_dims[0] = dim0;
    for (int i = 1; i < MAX_DATA_DIMS; i++)
        np_dims[i] = i < ndim ? (npy_intp)va_arg(args, uint32_t) : 0;

    va_end(args);

    new_array(type, ndim, np_dims);
}


///
/// \brief  Instantiate a new NpyArray with the respective numpy array
///         and initialize it with the provided data.
///
NpyArray::NpyArray(NpyType type, int ndim, npy_intp* dims, void* data) : NpyArray()
{
    new_array(type, ndim, dims, data);
}


///
/// \brief  Instantiate a new NpyArray with the respective numpy array
///         and initialize it with the provided data.
///
NpyArray::NpyArray(NpyType type, int ndim, uint32_t* dims, void* data) : NpyArray()
{
    npy_intp np_dims[NPY_MAXDIMS];
    for (uint8_t i = 0; i < ndim; i++)
        np_dims[i] = (npy_intp)dims[i];

    new_array(type, ndim, np_dims, data);
}


///
/// \brief  Release a numpy reference.
///
NpyArray::~NpyArray()
{
    this->decref();
}


bool NpyArray::valid()
{
    return numpy_array != nullptr;
}


const bool NpyArray::valid() const
{
    return numpy_array != nullptr;
}


///
/// \brief  Create a new uninitialized NpyArray of the same size
///         and type as the specified NpyArray.
///
NpyArray NpyArray::empty_like(const NpyArray& other)
{
    return NpyArray(other.type(), other.ndim(), other.shape());
}


///
/// \brief  Get a Python object that can be returned to the
///         Python interpreter.
///
PyObject* NpyArray::py_return()
{
    PyObject* result = (PyObject*)numpy_array;
    if (PyArray_CHKFLAGS(numpy_array, NPY_ARRAY_OWNDATA))
        this->numpy_array = 0;
    return result;
}


///
/// \brief  Get the NpyType of the array elements.
///
NpyType NpyArray::type() const
{
    return (NpyType)PyArray_TYPE(numpy_array);
}


///
/// \brief  Get the numpy type of the array elements.
///
PyArray_Descr* NpyArray::dtype() const
{
    return PyArray_DTYPE(numpy_array);
}


///
/// \brief  Get the number of array dimensions.
///
int NpyArray::ndim() const
{
    return PyArray_NDIM(numpy_array);
}


///
/// \brief  Get the number of element
///         for each dimension.
///
npy_intp* NpyArray::shape() const
{
    return PyArray_SHAPE(numpy_array);
}


///
/// \brief  Get the number of element
///         for the specified dimension.
///
int NpyArray::shape(int i) const
{
    return (int)PyArray_DIM(numpy_array, i);
}


///
/// \brief  Get a specific element of the array.
///
template<typename T>
T& NpyArray::get(int i0, int i1, int i2)
{
    npy_intp ptr[MAX_DATA_DIMS];
    ptr[0] = i0;
    ptr[1] = i1;
    ptr[2] = i2;
    return *(T*)PyArray_GetPtr(numpy_array, ptr);
}


///
/// \brief  The size in bytes of a single element.
///
int NpyArray::element_size(NpyType type) const
{
    return (int)PyArray_ITEMSIZE(numpy_array);
}


///
/// \brief  The number of elements
///         in the whole array.
///
int NpyArray::size() const
{
    return (int)PyArray_SIZE(numpy_array);
}


///
/// \brief  The size in bytes of the whole array.
///
int NpyArray::nbytes() const
{
    return (int)PyArray_NBYTES(numpy_array);
}


///
/// \brief  Get a pointer to the array data.
///
void* NpyArray::data()
{
    return PyArray_DATA(numpy_array);
}


///
/// \brief  Get a constant pointer to the array data.
///
const void* NpyArray::data() const
{
    return (const void*)PyArray_DATA(numpy_array);
}


///
/// \brief  Get a pointer to the array data.
///
template<typename T>
T* NpyArray::cast()
{
    return (T*)PyArray_DATA(numpy_array);
}


///
/// \brief  Get a constant pointer to the array data.
///
template<typename T>
const T* NpyArray::cast() const
{
    return (const T*)PyArray_DATA(numpy_array);
}


///
/// \brief  Get the current reference count of the numpy array.
///
int NpyArray::refcount() const
{
    return numpy_array ? (int)PyArray_REFCOUNT(numpy_array) : 0;
}


///
/// \brief  Increase the reference count of the numpy array.
///
void NpyArray::incref()
{
    if (this->numpy_array)
        Py_INCREF(this->numpy_array);
}


///
/// \brief  Decrease the reference count of the numpy array.
///
void NpyArray::decref()
{
    if (this->numpy_array)
    {
        bool del = PyArray_REFCOUNT(this->numpy_array) == 1;
        Py_DECREF((PyObject*)this->numpy_array);
        if (del)
            this->numpy_array = 0;
    }
}


///
/// \brief  Create a new NpyArray and respective numpy array of the specified size
///         and initialize it using the specified data (if provided).
///
void NpyArray::new_array(NpyType type, int ndim, npy_intp* dims, void* data)
{
    if (data != 0)
    {
        this->numpy_array = (PyArrayObject*)PyArray_SimpleNewFromData(ndim, dims, (int)type, data);
        PyArray_ENABLEFLAGS(numpy_array, NPY_ARRAY_OWNDATA);
    }
    else
    {
        this->numpy_array = (PyArrayObject*)PyArray_SimpleNew(ndim, dims, (int)type);
    }
}


// COMPILE A PYTHON OBJECT TO BE RETURNED TO PYTHON CODE FROM PXT TYPES


///
/// \brief   End template parameter pack of `args2tuple`.
///
/// \detail  This function is called by `args2tuple(vector, T, Args...)`
///          once there are no arguments left to be converted into Python
///          objects. It therefore ends the recursive calls of `args2tuple`.
///
void args2tuple(std::vector<PyObject*>& result) { }


///
/// \brief   Convert a C and PXT types into a python objects.
///
/// \detail  Convert a C and PXT types into a python object using a template
///          parameter pack (C++11). The function will be called recursively
///          until there are no arguments left to be converted.
///
template<typename T, typename... Args>
void args2tuple(std::vector<PyObject*>& result, T value, Args... args)
{
    auto obj = c2py(value);
    result.push_back(obj);
    args2tuple(result, args...);
}


///
/// \brief  Convert a list of C and PXT types into a Python compatible
///         result object. The number of arguments can vary.
///
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


// CONVERT PXT TYPES TO PYTHON OBJECTS

PyObject* c2py(bool value) { return PyBool_FromLong((long)value); }
PyObject* c2py(int8_t value) { return PyLong_FromLong((long)value); }
PyObject* c2py(int16_t value) { return PyLong_FromLong((long)value); }
PyObject* c2py(int32_t value) { return PyLong_FromLong((long)value); }
PyObject* c2py(int64_t value) { return PyLong_FromLongLong((long long)value); }
PyObject* c2py(uint8_t value) { return PyLong_FromUnsignedLong((unsigned long)value); }
PyObject* c2py(uint16_t value) { return PyLong_FromUnsignedLong((unsigned long)value); }
PyObject* c2py(uint32_t value) { return PyLong_FromUnsignedLong((unsigned long)value); }
PyObject* c2py(uint64_t value) { return PyLong_FromUnsignedLongLong((unsigned long long)value); }
PyObject* c2py(float value) { return PyFloat_FromDouble((double)value); }
PyObject* c2py(double value) { return PyFloat_FromDouble((double)value); }
PyObject* c2py(NpyArray value) { return value.py_return(); }
PyObject* c2py(PyObject* value)
{
    if (value == Py_None)
        Py_RETURN_NONE;
    else if (value == Py_True)
        Py_RETURN_TRUE;
    else if (value == Py_False)
        Py_RETURN_FALSE;
    return value;
}
