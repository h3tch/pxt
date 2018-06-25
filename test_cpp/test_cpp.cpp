#include "pxt/pxt.h"

EXPORT int i_add(int a, int b)
{
    return a + b;
}

template<typename T>
void add(T* a, T* b, T* c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

EXPORT PyObject* np_add(PyObject* a_, PyObject* b_)
{
    NpyArray a(a_);
    NpyArray b(b_);

    if (PyErr_Occurred())
        return 0;

    if (a.type != b.type || a.size() != b.size())
    {
        PyErr_SetString(PyExc_TypeError,
                        "assert(a.type == b.type && a.size() == b.size())");
        return 0;
    }

    NpyArray c(a.type, a.ndim, a.dims);

    switch(a.type)
    {
    case NpyType::boolean:
        add(a.cast<uint8_t>(), b.cast<uint8_t>(), c.cast<uint8_t>(), a.size());
        break;
    case NpyType::int8:
        add(a.cast<int8_t>(), b.cast<int8_t>(), c.cast<int8_t>(), a.size());
        break;
    case NpyType::uint8:
        add(a.cast<uint8_t>(), b.cast<uint8_t>(), c.cast<uint8_t>(), a.size());
        break;
    case NpyType::int16:
        add(a.cast<int16_t>(), b.cast<int16_t>(), c.cast<int16_t>(), a.size());
        break;
    case NpyType::uint16:
        add(a.cast<uint16_t>(), b.cast<uint16_t>(), c.cast<uint16_t>(), a.size());
        break;
    case NpyType::int32:
        add(a.cast<int32_t>(), b.cast<int32_t>(), c.cast<int32_t>(), a.size());
        break;
    case NpyType::uint32:
        add(a.cast<uint32_t>(), b.cast<uint32_t>(), c.cast<uint32_t>(), a.size());
        break;
    case NpyType::int64:
        add(a.cast<int64_t>(), b.cast<int64_t>(), c.cast<int64_t>(), a.size());
        break;
    case NpyType::uint64:
        add(a.cast<uint64_t>(), b.cast<uint64_t>(), c.cast<uint64_t>(), a.size());
        break;
    case NpyType::Int:
        add(a.cast<int>(), b.cast<int>(), c.cast<int>(), a.size());
        break;
    case NpyType::Uint:
        add(a.cast<unsigned int>(), b.cast<unsigned int>(), c.cast<unsigned int>(), a.size());
        break;
    case NpyType::float32:
        add(a.cast<float>(), b.cast<float>(), c.cast<float>(), a.size());
        break;
    case NpyType::float64:
        add(a.cast<double>(), b.cast<double>(), c.cast<double>(), a.size());
        break;
    default: break;
    }

    std::cout << "asPyObjecto\n";
    return c.asPyObject();
}

EXPORT PyObject* return_tuple(PyObject* a, PyObject* b)
{
    std::cout << "\n";
    return PyResult(1, 2, 3, a, b);
}

MOD_INIT(test_cpp, NULL)
