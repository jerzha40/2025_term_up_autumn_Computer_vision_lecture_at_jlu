// fastmodule.c
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

// your C kernel (compiled in the same build)
double sumsq(const double *x, int n);

static PyObject *py_sumsq(PyObject *self, PyObject *args)
{
    PyObject *obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;

    // Convert to contiguous float64 NumPy array (no copy if already matching)
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        obj, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!arr)
        return NULL;

    if (PyArray_NDIM(arr) != 1)
    {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_TypeError, "x must be 1D float64 array");
        return NULL;
    }

    const double *data = (const double *)PyArray_DATA(arr);
    const int n = (int)PyArray_SIZE(arr);

    double out;
    // release GIL around CPU-bound C loop
    Py_BEGIN_ALLOW_THREADS
        out = sumsq(data, n);
    Py_END_ALLOW_THREADS

        Py_DECREF(arr);
    return PyFloat_FromDouble(out);
}

static PyMethodDef FastMethods[] = {
    {"sumsq", py_sumsq, METH_VARARGS, "Sum of squares over a float64 1D array."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef fastmodule = {
    PyModuleDef_HEAD_INIT, "fast", "Fast C extension demo", -1, FastMethods};

PyMODINIT_FUNC PyInit_fast(void)
{
    import_array(); // initialize NumPy C-API
    return PyModule_Create(&fastmodule);
}
