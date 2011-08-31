#include <Python.h>
#include <stdbool.h>

void mandelbrot_row_calc(int *nStep_ptr, float *deltaStep_ptr, float *start_x_ptr, float *start_y_ptr, int *mandelSize_ptr, int *innerLoopSize_ptr, int *outerLoopSize_ptr, unsigned int *results);

static PyObject *mandel_get_row(PyObject *self, PyObject *args) {
    float x_start, y_start, x_delta;
    int points_per_row;
    unsigned int *pixel_values;
    PyObject *tuple, *newVal;
    int i;
    int mandelSize = 16;
    int innerLoop = 500;
    int outerLoop = 2;

    if (!PyArg_ParseTuple(args, "ifffi", &points_per_row, &x_delta, &x_start, &y_start, &mandelSize)) return NULL;
    tuple = PyTuple_New(points_per_row);

    pixel_values = (unsigned int *) malloc(sizeof(unsigned int)*points_per_row);
    mandelbrot_row_calc(&points_per_row, &x_delta, &x_start, &y_start, &mandelSize, &innerLoop, &outerLoop, pixel_values);
		 
    for (i=0; i < points_per_row; i++) {
        newVal = PyInt_FromLong(pixel_values[i]);
        Py_INCREF(newVal);
        PyTuple_SetItem(tuple, i, newVal);
    }
    free(pixel_values);
    return tuple;
}


static PyMethodDef MandelMethods[] = {
          {"get_row",  mandel_get_row, METH_VARARGS, ""},
          {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initmandel(void) {
    PyObject *m;

    m = Py_InitModule("mandel", MandelMethods);
    if (m == NULL) return;

}

