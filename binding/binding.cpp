// binding/binding.cpp
//
// Zero-dependency CPython C-API extension module (`log3._C`) exposing the
// entire log3 C++17 ecosystem to Python (Core, Rng, N-D Tensor with PEP 3118
// zero-copy Buffer Protocol, AVX2/NEON SIMD Kernels, BitNet b1.58 Layers &
// GQA Transformer, Zero-Copy .l3 MMap Checkpoints, Reverse-Mode Autograd with
// Straight-Through Estimator QAT, and SGD/AdamW Optimizers).
//
// Why direct CPython C-API (<Python.h>) instead of pybind11:
//   * Compiles with -fno-exceptions and -fno-rtti (matching log3's core flags).
//   * Zero external C++ build dependencies (no pybind11/nanobind required).
//   * Native PEP 3118 Buffer Protocol (`memoryview(tensor)` / `np.asarray(tensor)`)
//     with zero intermediate copies over 64-byte cache-aligned Storage.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

#include "log3/log3.h"

namespace {

using namespace l3;

#define L3_PY_CFUNC(fn) reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(fn))

// =============================================================================
// Error Translation: l3::Status -> Python Exception
// =============================================================================
PyObject* set_python_error(const Status& st) noexcept {
  if (st.is_ok()) {
    Py_RETURN_NONE;
  }
  PyObject* exc_type = PyExc_RuntimeError;
  switch (st.code()) {
    case StatusCode::kInvalidArgument:
    case StatusCode::kShapeMismatch:
    case StatusCode::kTypeMismatch:
    case StatusCode::kFailedPrecondition:
    case StatusCode::kAlreadyExists:
      exc_type = PyExc_ValueError;
      break;
    case StatusCode::kOutOfRange:
      exc_type = PyExc_IndexError;
      break;
    case StatusCode::kNotFound:
      exc_type = PyExc_KeyError;
      break;
    case StatusCode::kOutOfMemory:
      exc_type = PyExc_MemoryError;
      break;
    case StatusCode::kIoError:
      exc_type = PyExc_OSError;
      break;
    case StatusCode::kUnimplemented:
    case StatusCode::kUnsupported:
      exc_type = PyExc_NotImplementedError;
      break;
    default:
      exc_type = PyExc_RuntimeError;
      break;
  }
  PyErr_Format(exc_type, "[log3::%s] %s", status_code_name(st.code()),
               st.message());
  return nullptr;
}

// Parse a DType from either a string ("f32", "float32", "t2", "ternary2", ...)
// or an integer DType enum value.
bool parse_dtype_obj(PyObject* obj, DType* out) noexcept {
  if (obj == nullptr || obj == Py_None) {
    *out = DType::kFloat32;
    return true;
  }
  if (PyLong_Check(obj)) {
    long v = PyLong_AsLong(obj);
    if (v >= 0 && v < static_cast<long>(kNumDTypes)) {
      *out = static_cast<DType>(v);
      return true;
    }
    PyErr_SetString(PyExc_ValueError, "Invalid integer DType code.");
    return false;
  }
  if (PyUnicode_Check(obj)) {
    Py_ssize_t len = 0;
    const char* s = PyUnicode_AsUTF8AndSize(obj, &len);
    if (!s) return false;
    StrView sv(s, static_cast<Size>(len));
    if (sv == "f32" || sv == "float32" || sv == "float") {
      *out = DType::kFloat32;
      return true;
    }
    if (sv == "f16" || sv == "float16" || sv == "half") {
      *out = DType::kFloat16;
      return true;
    }
    if (sv == "bf16" || sv == "bfloat16") {
      *out = DType::kBFloat16;
      return true;
    }
    if (sv == "f64" || sv == "float64" || sv == "double") {
      *out = DType::kFloat64;
      return true;
    }
    if (sv == "i32" || sv == "int32" || sv == "int") {
      *out = DType::kInt32;
      return true;
    }
    if (sv == "i64" || sv == "int64" || sv == "long") {
      *out = DType::kInt64;
      return true;
    }
    if (sv == "i16" || sv == "int16" || sv == "short") {
      *out = DType::kInt16;
      return true;
    }
    if (sv == "i8" || sv == "int8") {
      *out = DType::kInt8;
      return true;
    }
    if (sv == "u8" || sv == "uint8" || sv == "byte") {
      *out = DType::kUInt8;
      return true;
    }
    if (sv == "bool" || sv == "bool_") {
      *out = DType::kBool;
      return true;
    }
    if (sv == "t2" || sv == "ternary2" || sv == "bitnet") {
      *out = DType::kTernary2;
      return true;
    }
    if (sv == "i4" || sv == "int4") {
      *out = DType::kInt4;
      return true;
    }
    if (sv == "u4" || sv == "uint4") {
      *out = DType::kUInt4;
      return true;
    }
    PyErr_Format(PyExc_ValueError, "Unknown DType string '%s'.", s);
    return false;
  }
  PyErr_SetString(PyExc_TypeError, "dtype must be a string (e.g. 'f32', 't2') or int.");
  return false;
}

// Parse a Python sequence of ints (or a single int) into an l3::Shape.
bool parse_shape_obj(PyObject* obj, Shape* out) noexcept {
  if (PyLong_Check(obj)) {
    Index d = static_cast<Index>(PyLong_AsLongLong(obj));
    if (PyErr_Occurred()) return false;
    auto res = Shape::from_dims({d});
    if (!res.is_ok()) {
      set_python_error(res.status());
      return false;
    }
    *out = res.take_value();
    return true;
  }
  PyObject* seq = PySequence_Fast(obj, "shape must be a sequence of integers or an int");
  if (!seq) return false;
  const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  if (n > kMaxTensorDims) {
    Py_DECREF(seq);
    PyErr_Format(PyExc_ValueError, "Shape rank %zd exceeds kMaxTensorDims (%lld)",
                 n, static_cast<long long>(kMaxTensorDims));
    return false;
  }
  Index dims[kMaxTensorDims];
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
    dims[i] = static_cast<Index>(PyLong_AsLongLong(item));
    if (PyErr_Occurred()) {
      Py_DECREF(seq);
      return false;
    }
  }
  Py_DECREF(seq);
  auto res = Shape::from_dims(Span<const Index>(dims, static_cast<Index>(n)));
  if (!res.is_ok()) {
    set_python_error(res.status());
    return false;
  }
  *out = res.take_value();
  return true;
}

// Parse a Python sequence of ints into a SmallVec<Index, 128>.
bool parse_index_vec(PyObject* obj, SmallVec<Index, 128>* out) noexcept {
  PyObject* seq = PySequence_Fast(obj, "Expected a sequence of integers");
  if (!seq) return false;
  const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  out->clear();
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
    Index val = static_cast<Index>(PyLong_AsLongLong(item));
    if (PyErr_Occurred()) {
      Py_DECREF(seq);
      return false;
    }
    if (!out->push_back(val).is_ok()) {
      Py_DECREF(seq);
      PyErr_NoMemory();
      return false;
    }
  }
  Py_DECREF(seq);
  return true;
}

// =============================================================================
// Forward Declarations of Python Type Objects
// =============================================================================
extern PyTypeObject PyTensorType;
extern PyTypeObject PyRngType;
extern PyTypeObject PyNoGradGuardType;
extern PyTypeObject PyRMSNormType;
extern PyTypeObject PyBitLinearType;
extern PyTypeObject PyLinearType;
extern PyTypeObject PyEmbeddingType;
extern PyTypeObject PyRotaryEmbeddingType;
extern PyTypeObject PyKVCacheType;
extern PyTypeObject PyGQAType;
extern PyTypeObject PyBitNetFFNType;
extern PyTypeObject PyTransformerBlockType;
extern PyTypeObject PyBitNetTransformerType;
extern PyTypeObject PyModelWriterType;
extern PyTypeObject PyModelReaderType;
extern PyTypeObject PySGDType;
extern PyTypeObject PyAdamWType;

// =============================================================================
// 1. PyTensor Object (Wraps l3::Tensor + PEP 3118 Zero-Copy Buffer Protocol)
// =============================================================================
struct PyTensor {
  PyObject_HEAD
  Tensor tensor;
  Py_ssize_t buf_shape[kMaxTensorDims];
  Py_ssize_t buf_strides[kMaxTensorDims];
};

PyObject* wrap_tensor(Tensor&& t) noexcept {
  PyTensor* self = PyObject_New(PyTensor, &PyTensorType);
  if (!self) return nullptr;
  new (&self->tensor) Tensor(std::move(t));
  return reinterpret_cast<PyObject*>(self);
}

PyObject* wrap_result_tensor(Result<Tensor>&& res) noexcept {
  if (!res.is_ok()) {
    return set_python_error(res.status());
  }
  return wrap_tensor(res.take_value());
}

bool check_tensor(PyObject* obj, Tensor** out) noexcept {
  if (!PyObject_TypeCheck(obj, &PyTensorType)) {
    PyErr_SetString(PyExc_TypeError, "Expected a log3.Tensor object.");
    return false;
  }
  *out = &reinterpret_cast<PyTensor*>(obj)->tensor;
  return true;
}

void PyTensor_dealloc(PyTensor* self) noexcept {
  self->tensor.~Tensor();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyTensor_init(PyTensor* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"shape", "dtype", "requires_grad", nullptr};
  PyObject* shape_obj = nullptr;
  PyObject* dtype_obj = nullptr;
  int req_grad = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOp",
                                   const_cast<char**>(kwlist), &shape_obj,
                                   &dtype_obj, &req_grad)) {
    return -1;
  }
  new (&self->tensor) Tensor();
  if (!shape_obj) {
    return 0;
  }
  Shape s;
  if (!parse_shape_obj(shape_obj, &s)) return -1;
  DType dt = DType::kFloat32;
  if (!parse_dtype_obj(dtype_obj, &dt)) return -1;
  auto res = Tensor::zeros(s, dt);
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->tensor = res.take_value();
  if (req_grad) {
    Status st = self->tensor.set_requires_grad(true);
    if (!st.is_ok()) {
      set_python_error(st);
      return -1;
    }
  }
  return 0;
}

// PEP 3118 Zero-Copy Buffer Protocol
int PyTensor_getbuffer(PyObject* exporter, Py_buffer* view, int flags) noexcept {
  PyTensor* self = reinterpret_cast<PyTensor*>(exporter);
  Tensor& t = self->tensor;
  if (!t.is_valid()) {
    PyErr_SetString(PyExc_BufferError, "Cannot export buffer from an empty Tensor.");
    return -1;
  }
  if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE && !t.is_mutable()) {
    PyErr_SetString(PyExc_BufferError, "Tensor storage is read-only.");
    return -1;
  }
  if (t.is_quantized() && !t.is_contiguous()) {
    PyErr_SetString(PyExc_BufferError, "Sub-byte quantized tensor must be contiguous for buffer export.");
    return -1;
  }

  const char* fmt = "B";
  Py_ssize_t itemsize = 1;
  switch (t.dtype()) {
    case DType::kFloat32: fmt = "f"; itemsize = 4; break;
    case DType::kFloat64: fmt = "d"; itemsize = 8; break;
    case DType::kFloat16: fmt = "e"; itemsize = 2; break;
    case DType::kBFloat16: fmt = "H"; itemsize = 2; break;
    case DType::kInt32:   fmt = "i"; itemsize = 4; break;
    case DType::kInt64:   fmt = "q"; itemsize = 8; break;
    case DType::kInt16:   fmt = "h"; itemsize = 2; break;
    case DType::kInt8:    fmt = "b"; itemsize = 1; break;
    case DType::kUInt8:   fmt = "B"; itemsize = 1; break;
    case DType::kBool:    fmt = "?"; itemsize = 1; break;
    case DType::kTernary2:
    case DType::kInt4:
    case DType::kUInt4:
      view->buf = const_cast<byte*>(t.raw_data());
      view->obj = exporter;
      Py_INCREF(exporter);
      view->len = static_cast<Py_ssize_t>(t.nbytes());
      view->readonly = !t.is_mutable();
      view->itemsize = 1;
      view->format = (flags & PyBUF_FORMAT) ? const_cast<char*>("B") : nullptr;
      view->ndim = 1;
      self->buf_shape[0] = view->len;
      self->buf_strides[0] = 1;
      view->shape = (flags & PyBUF_ND) ? self->buf_shape : nullptr;
      view->strides = (flags & PyBUF_STRIDES) ? self->buf_strides : nullptr;
      view->suboffsets = nullptr;
      view->internal = nullptr;
      return 0;
    default:
      break;
  }

  const Index r = t.rank();
  for (Index i = 0; i < r; ++i) {
    self->buf_shape[i] = static_cast<Py_ssize_t>(t.dim(i));
    self->buf_strides[i] = static_cast<Py_ssize_t>(t.stride(i)) * itemsize;
  }

  view->buf = const_cast<byte*>(t.raw_data());
  view->obj = exporter;
  Py_INCREF(exporter);
  view->len = static_cast<Py_ssize_t>(t.numel()) * itemsize;
  view->readonly = !t.is_mutable();
  view->itemsize = itemsize;
  view->format = (flags & PyBUF_FORMAT) ? const_cast<char*>(fmt) : nullptr;
  view->ndim = static_cast<int>(r);
  view->shape = (flags & PyBUF_ND) ? self->buf_shape : nullptr;
  view->strides = (flags & PyBUF_STRIDES) ? self->buf_strides : nullptr;
  view->suboffsets = nullptr;
  view->internal = nullptr;
  return 0;
}

PyBufferProcs PyTensor_as_buffer = {
    PyTensor_getbuffer,
    nullptr,
};

// Tensor Properties
PyObject* PyTensor_get_shape(PyTensor* self, void*) noexcept {
  const Shape& s = self->tensor.shape();
  PyObject* tup = PyTuple_New(s.rank());
  if (!tup) return nullptr;
  for (Index i = 0; i < s.rank(); ++i) {
    PyTuple_SET_ITEM(tup, i, PyLong_FromLongLong(s.dim(i)));
  }
  return tup;
}

PyObject* PyTensor_get_strides(PyTensor* self, void*) noexcept {
  const Strides& st = self->tensor.strides();
  const Index r = self->tensor.rank();
  PyObject* tup = PyTuple_New(r);
  if (!tup) return nullptr;
  for (Index i = 0; i < r; ++i) {
    PyTuple_SET_ITEM(tup, i, PyLong_FromLongLong(st[i]));
  }
  return tup;
}

PyObject* PyTensor_get_ndim(PyTensor* self, void*) noexcept {
  return PyLong_FromLongLong(self->tensor.rank());
}

PyObject* PyTensor_get_numel(PyTensor* self, void*) noexcept {
  return PyLong_FromLongLong(self->tensor.numel());
}

PyObject* PyTensor_get_nbytes(PyTensor* self, void*) noexcept {
  return PyLong_FromSize_t(self->tensor.nbytes());
}

PyObject* PyTensor_get_byte_offset(PyTensor* self, void*) noexcept {
  return PyLong_FromSize_t(self->tensor.byte_offset());
}

PyObject* PyTensor_get_dtype(PyTensor* self, void*) noexcept {
  return PyUnicode_FromString(dtype_name(self->tensor.dtype()));
}

PyObject* PyTensor_get_is_contiguous(PyTensor* self, void*) noexcept {
  return PyBool_FromLong(self->tensor.is_contiguous());
}

PyObject* PyTensor_get_is_mutable(PyTensor* self, void*) noexcept {
  return PyBool_FromLong(self->tensor.is_mutable());
}

PyObject* PyTensor_get_is_quantized(PyTensor* self, void*) noexcept {
  return PyBool_FromLong(self->tensor.is_quantized());
}

PyObject* PyTensor_get_quant_scale(PyTensor* self, void*) noexcept {
  return PyFloat_FromDouble(static_cast<double>(self->tensor.quant_scale()));
}

int PyTensor_set_quant_scale(PyTensor* self, PyObject* val, void*) noexcept {
  if (!val) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete quant_scale.");
    return -1;
  }
  double v = PyFloat_AsDouble(val);
  if (PyErr_Occurred()) return -1;
  self->tensor.set_quant_scale(static_cast<f32>(v));
  return 0;
}

PyObject* PyTensor_get_requires_grad(PyTensor* self, void*) noexcept {
  return PyBool_FromLong(self->tensor.requires_grad());
}

int PyTensor_set_requires_grad(PyTensor* self, PyObject* val, void*) noexcept {
  if (!val) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete requires_grad.");
    return -1;
  }
  int truth = PyObject_IsTrue(val);
  if (truth < 0) return -1;
  Status st = self->tensor.set_requires_grad(truth != 0);
  if (!st.is_ok()) {
    set_python_error(st);
    return -1;
  }
  return 0;
}

PyObject* PyTensor_get_is_leaf(PyTensor* self, void*) noexcept {
  return PyBool_FromLong(self->tensor.is_leaf());
}

PyObject* PyTensor_get_has_grad(PyTensor* self, void*) noexcept {
  return PyBool_FromLong(self->tensor.has_grad());
}

PyObject* PyTensor_get_grad(PyTensor* self, void*) noexcept {
  const Tensor* g = self->tensor.grad();
  if (!g || !g->is_valid()) {
    Py_RETURN_NONE;
  }
  return wrap_tensor(g->view());
}

// Tensor Methods
PyObject* PyTensor_view(PyTensor* self, PyObject*) noexcept {
  return wrap_tensor(self->tensor.view());
}

PyObject* PyTensor_detach(PyTensor* self, PyObject*) noexcept {
  return wrap_tensor(self->tensor.detach());
}

PyObject* PyTensor_clone(PyTensor* self, PyObject*) noexcept {
  return wrap_result_tensor(self->tensor.clone());
}

PyObject* PyTensor_contiguous(PyTensor* self, PyObject*) noexcept {
  return wrap_result_tensor(self->tensor.contiguous());
}

PyObject* PyTensor_reshape(PyTensor* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"shape", nullptr};
  PyObject* shape_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &shape_obj)) {
    return nullptr;
  }
  Shape s;
  if (!parse_shape_obj(shape_obj, &s)) return nullptr;
  return wrap_result_tensor(self->tensor.reshape(s));
}

PyObject* PyTensor_transpose(PyTensor* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"dim0", "dim1", nullptr};
  long long d0 = 0, d1 = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "LL", const_cast<char**>(kwlist), &d0, &d1)) {
    return nullptr;
  }
  return wrap_result_tensor(self->tensor.transpose(static_cast<Index>(d0), static_cast<Index>(d1)));
}

PyObject* PyTensor_permute(PyTensor* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"dims", nullptr};
  PyObject* dims_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &dims_obj)) {
    return nullptr;
  }
  SmallVec<Index, 128> perm;
  if (!parse_index_vec(dims_obj, &perm)) return nullptr;
  return wrap_result_tensor(self->tensor.permute(Span<const Index>(perm.data(), perm.size())));
}

PyObject* PyTensor_slice(PyTensor* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"axis", "start", "end", nullptr};
  long long axis = 0, start = 0, end = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "LLL", const_cast<char**>(kwlist),
                                   &axis, &start, &end)) {
    return nullptr;
  }
  return wrap_result_tensor(self->tensor.slice(static_cast<Index>(axis),
                                               static_cast<Index>(start),
                                               static_cast<Index>(end)));
}

PyObject* PyTensor_unsqueeze(PyTensor* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"axis", nullptr};
  long long axis = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "L", const_cast<char**>(kwlist), &axis)) {
    return nullptr;
  }
  return wrap_result_tensor(self->tensor.unsqueeze(static_cast<Index>(axis)));
}

PyObject* PyTensor_squeeze(PyTensor* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"axis", nullptr};
  long long axis = -999;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|L", const_cast<char**>(kwlist), &axis)) {
    return nullptr;
  }
  return wrap_result_tensor(self->tensor.squeeze(static_cast<Index>(axis)));
}

PyObject* PyTensor_backward(PyTensor* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"grad_output", nullptr};
  PyObject* grad_obj = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", const_cast<char**>(kwlist), &grad_obj)) {
    return nullptr;
  }
  const Tensor* grad_ptr = nullptr;
  if (grad_obj != nullptr && grad_obj != Py_None) {
    Tensor* tmp = nullptr;
    if (!check_tensor(grad_obj, &tmp)) return nullptr;
    grad_ptr = tmp;
  }
  Status st = self->tensor.backward(grad_ptr);
  if (!st.is_ok()) return set_python_error(st);
  Py_RETURN_NONE;
}

PyObject* PyTensor_zero_grad(PyTensor* self, PyObject*) noexcept {
  self->tensor.zero_grad();
  Py_RETURN_NONE;
}

PyObject* PyTensor_fill_(PyTensor* self, PyObject* args) noexcept {
  double val = 0.0;
  if (!PyArg_ParseTuple(args, "d", &val)) return nullptr;
  if (!self->tensor.is_valid() || !self->tensor.is_mutable() || !self->tensor.is_contiguous()) {
    PyErr_SetString(PyExc_ValueError, "fill_ requires a valid, mutable, contiguous Tensor.");
    return nullptr;
  }
  if (self->tensor.dtype() == DType::kFloat32) {
    f32* p = self->tensor.mutable_data_ptr<f32>();
    const f32 fv = static_cast<f32>(val);
    for (Index i = 0; i < self->tensor.numel(); ++i) p[i] = fv;
  } else if (self->tensor.dtype() == DType::kInt32) {
    i32* p = self->tensor.mutable_data_ptr<i32>();
    const i32 iv = static_cast<i32>(val);
    for (Index i = 0; i < self->tensor.numel(); ++i) p[i] = iv;
  } else {
    PyErr_SetString(PyExc_NotImplementedError, "fill_ currently supports f32 and i32 tensors.");
    return nullptr;
  }
  Py_INCREF(self);
  return reinterpret_cast<PyObject*>(self);
}

PyObject* PyTensor_to_list(PyTensor* self, PyObject*) noexcept {
  auto c_res = self->tensor.contiguous();
  if (!c_res.is_ok()) return set_python_error(c_res.status());
  Tensor c = c_res.take_value();
  const Index n = c.numel();
  PyObject* lst = PyList_New(n);
  if (!lst) return nullptr;
  if (c.dtype() == DType::kFloat32) {
    const f32* p = c.data_ptr<f32>();
    for (Index i = 0; i < n; ++i) {
      PyList_SET_ITEM(lst, i, PyFloat_FromDouble(static_cast<double>(p[i])));
    }
  } else if (c.dtype() == DType::kInt32) {
    const i32* p = c.data_ptr<i32>();
    for (Index i = 0; i < n; ++i) {
      PyList_SET_ITEM(lst, i, PyLong_FromLong(p[i]));
    }
  } else if (c.dtype() == DType::kInt64) {
    const i64* p = c.data_ptr<i64>();
    for (Index i = 0; i < n; ++i) {
      PyList_SET_ITEM(lst, i, PyLong_FromLongLong(p[i]));
    }
  } else if (c.dtype() == DType::kInt8) {
    const i8* p = c.data_ptr<i8>();
    for (Index i = 0; i < n; ++i) {
      PyList_SET_ITEM(lst, i, PyLong_FromLong(p[i]));
    }
  } else if (c.dtype() == DType::kUInt8) {
    const u8* p = c.data_ptr<u8>();
    for (Index i = 0; i < n; ++i) {
      PyList_SET_ITEM(lst, i, PyLong_FromUnsignedLong(p[i]));
    }
  } else {
    Py_DECREF(lst);
    PyErr_Format(PyExc_NotImplementedError, "to_list() not supported for dtype %s",
                 dtype_name(c.dtype()));
    return nullptr;
  }
  return lst;
}

PyObject* PyTensor_item(PyTensor* self, PyObject*) noexcept {
  if (self->tensor.numel() != 1) {
    PyErr_Format(PyExc_ValueError, "item() requires a 1-element Tensor (got %lld elements)",
                 static_cast<long long>(self->tensor.numel()));
    return nullptr;
  }
  if (self->tensor.dtype() == DType::kFloat32) {
    return PyFloat_FromDouble(static_cast<double>(self->tensor.data_ptr<f32>()[0]));
  }
  if (self->tensor.dtype() == DType::kInt32) {
    return PyLong_FromLong(self->tensor.data_ptr<i32>()[0]);
  }
  if (self->tensor.dtype() == DType::kInt64) {
    return PyLong_FromLongLong(self->tensor.data_ptr<i64>()[0]);
  }
  PyErr_Format(PyExc_NotImplementedError, "item() not implemented for dtype %s",
               dtype_name(self->tensor.dtype()));
  return nullptr;
}

PyObject* PyTensor_repr(PyTensor* self) noexcept {
  const Tensor& t = self->tensor;
  if (!t.is_valid()) {
    return PyUnicode_FromString("Tensor(invalid/empty)");
  }
  char shape_buf[128];
  int pos = 0;
  shape_buf[pos++] = '(';
  for (Index i = 0; i < t.rank(); ++i) {
    pos += std::snprintf(shape_buf + pos, sizeof(shape_buf) - pos,
                         (i + 1 < t.rank()) ? "%lld, " : "%lld",
                         static_cast<long long>(t.dim(i)));
  }
  if (t.rank() == 1) {
    pos += std::snprintf(shape_buf + pos, sizeof(shape_buf) - pos, ",");
  }
  shape_buf[pos++] = ')';
  shape_buf[pos] = '\0';

  return PyUnicode_FromFormat(
      "Tensor(shape=%s, dtype='%s', nbytes=%zu, requires_grad=%s, quant_scale=%.4f)",
      shape_buf, dtype_name(t.dtype()), t.nbytes(),
      t.requires_grad() ? "True" : "False",
      static_cast<double>(t.quant_scale()));
}

// Number protocol operators (+, -, *, @) dispatching through l3::autograd
PyObject* PyTensor_nb_add(PyObject* a, PyObject* b) noexcept {
  Tensor *ta = nullptr, *tb = nullptr;
  if (!check_tensor(a, &ta) || !check_tensor(b, &tb)) return nullptr;
  return wrap_result_tensor(autograd::add(*ta, *tb));
}

PyObject* PyTensor_nb_sub(PyObject* a, PyObject* b) noexcept {
  Tensor *ta = nullptr, *tb = nullptr;
  if (!check_tensor(a, &ta) || !check_tensor(b, &tb)) return nullptr;
  return wrap_result_tensor(autograd::sub(*ta, *tb));
}

PyObject* PyTensor_nb_mul(PyObject* a, PyObject* b) noexcept {
  Tensor *ta = nullptr, *tb = nullptr;
  if (!check_tensor(a, &ta) || !check_tensor(b, &tb)) return nullptr;
  return wrap_result_tensor(autograd::mul(*ta, *tb));
}

PyObject* PyTensor_nb_matmul(PyObject* a, PyObject* b) noexcept {
  Tensor *ta = nullptr, *tb = nullptr;
  if (!check_tensor(a, &ta) || !check_tensor(b, &tb)) return nullptr;
  return wrap_result_tensor(autograd::matmul(*ta, *tb));
}

PyNumberMethods PyTensor_as_number = {
    PyTensor_nb_add,    // nb_add
    PyTensor_nb_sub,    // nb_subtract
    PyTensor_nb_mul,    // nb_multiply
    nullptr,            // nb_remainder
    nullptr,            // nb_divmod
    nullptr,            // nb_power
    nullptr,            // nb_negative
    nullptr,            // nb_positive
    nullptr,            // nb_absolute
    nullptr,            // nb_bool
    nullptr,            // nb_invert
    nullptr,            // nb_lshift
    nullptr,            // nb_rshift
    nullptr,            // nb_and
    nullptr,            // nb_xor
    nullptr,            // nb_or
    nullptr,            // nb_int
    nullptr,            // nb_reserved
    nullptr,            // nb_float
    nullptr,            // nb_inplace_add
    nullptr,            // nb_inplace_subtract
    nullptr,            // nb_inplace_multiply
    nullptr,            // nb_inplace_remainder
    nullptr,            // nb_inplace_power
    nullptr,            // nb_inplace_lshift
    nullptr,            // nb_inplace_rshift
    nullptr,            // nb_inplace_and
    nullptr,            // nb_inplace_xor
    nullptr,            // nb_inplace_or
    nullptr,            // nb_floor_divide
    nullptr,            // nb_true_divide
    nullptr,            // nb_inplace_floor_divide
    nullptr,            // nb_inplace_true_divide
    nullptr,            // nb_index
    PyTensor_nb_matmul, // nb_matrix_multiply
    nullptr,            // nb_inplace_matrix_multiply
};

PyGetSetDef PyTensor_getset[] = {
    {"shape", reinterpret_cast<getter>(PyTensor_get_shape), nullptr,
     "Tuple of tensor dimensions.", nullptr},
    {"strides", reinterpret_cast<getter>(PyTensor_get_strides), nullptr,
     "Tuple of element strides per dimension.", nullptr},
    {"ndim", reinterpret_cast<getter>(PyTensor_get_ndim), nullptr,
     "Number of tensor dimensions (rank).", nullptr},
    {"rank", reinterpret_cast<getter>(PyTensor_get_ndim), nullptr,
     "Alias for ndim.", nullptr},
    {"numel", reinterpret_cast<getter>(PyTensor_get_numel), nullptr,
     "Total number of logical elements.", nullptr},
    {"nbytes", reinterpret_cast<getter>(PyTensor_get_nbytes), nullptr,
     "Total storage bytes occupied by this tensor.", nullptr},
    {"byte_offset", reinterpret_cast<getter>(PyTensor_get_byte_offset), nullptr,
     "Byte offset inside the underlying Storage.", nullptr},
    {"dtype", reinterpret_cast<getter>(PyTensor_get_dtype), nullptr,
     "Data type name string (e.g. 'f32', 't2').", nullptr},
    {"is_contiguous", reinterpret_cast<getter>(PyTensor_get_is_contiguous), nullptr,
     "True if the tensor is row-major contiguous in memory.", nullptr},
    {"is_mutable", reinterpret_cast<getter>(PyTensor_get_is_mutable), nullptr,
     "True if underlying storage is writable.", nullptr},
    {"is_quantized", reinterpret_cast<getter>(PyTensor_get_is_quantized), nullptr,
     "True if dtype is a sub-byte quantized format ('t2', 'i4', 'u4').", nullptr},
    {"quant_scale", reinterpret_cast<getter>(PyTensor_get_quant_scale),
     reinterpret_cast<setter>(PyTensor_set_quant_scale),
     "Per-tensor quantization scale factor (f32).", nullptr},
    {"requires_grad", reinterpret_cast<getter>(PyTensor_get_requires_grad),
     reinterpret_cast<setter>(PyTensor_set_requires_grad),
     "True if reverse-mode autograd tracks gradients for this tensor.", nullptr},
    {"is_leaf", reinterpret_cast<getter>(PyTensor_get_is_leaf), nullptr,
     "True if requires_grad is True and created directly by the user.", nullptr},
    {"has_grad", reinterpret_cast<getter>(PyTensor_get_has_grad), nullptr,
     "True if a gradient tensor has been accumulated.", nullptr},
    {"grad", reinterpret_cast<getter>(PyTensor_get_grad), nullptr,
     "Accumulated gradient Tensor (or None).", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef PyTensor_methods[] = {
    {"view", L3_PY_CFUNC(PyTensor_view), METH_NOARGS,
     "view() -> Tensor\nReturn an O(1) zero-copy shallow alias sharing Storage."},
    {"detach", L3_PY_CFUNC(PyTensor_detach), METH_NOARGS,
     "detach() -> Tensor\nReturn a zero-copy view detached from the autograd graph."},
    {"clone", L3_PY_CFUNC(PyTensor_clone), METH_NOARGS,
     "clone() -> Tensor\nAllocate a deep contiguous copy of this tensor."},
    {"contiguous", L3_PY_CFUNC(PyTensor_contiguous), METH_NOARGS,
     "contiguous() -> Tensor\nReturn self.view() if contiguous, else self.clone()."},
    {"reshape", L3_PY_CFUNC(PyTensor_reshape),
     METH_VARARGS | METH_KEYWORDS,
     "reshape(shape: Sequence[int]) -> Tensor\nZero-copy reshape of a contiguous tensor."},
    {"transpose", L3_PY_CFUNC(PyTensor_transpose),
     METH_VARARGS | METH_KEYWORDS,
     "transpose(dim0: int, dim1: int) -> Tensor\nSwap two axes in O(1) without copying data."},
    {"permute", L3_PY_CFUNC(PyTensor_permute),
     METH_VARARGS | METH_KEYWORDS,
     "permute(dims: Sequence[int]) -> Tensor\nPermute tensor axes in O(1) without copying data."},
    {"slice", L3_PY_CFUNC(PyTensor_slice),
     METH_VARARGS | METH_KEYWORDS,
     "slice(axis: int, start: int, end: int) -> Tensor\nZero-copy slice along axis in [start, end)."},
    {"unsqueeze", L3_PY_CFUNC(PyTensor_unsqueeze),
     METH_VARARGS | METH_KEYWORDS,
     "unsqueeze(axis: int) -> Tensor\nInsert a size-1 dimension at axis in O(1)."},
    {"squeeze", L3_PY_CFUNC(PyTensor_squeeze),
     METH_VARARGS | METH_KEYWORDS,
     "squeeze(axis: int = -999) -> Tensor\nRemove size-1 dimension(s) in O(1)."},
    {"backward", L3_PY_CFUNC(PyTensor_backward),
     METH_VARARGS | METH_KEYWORDS,
     "backward(grad_output: Optional[Tensor] = None) -> None\nRun reverse-mode autodiff from this tensor."},
    {"zero_grad", L3_PY_CFUNC(PyTensor_zero_grad), METH_NOARGS,
     "zero_grad() -> None\nReset accumulated gradient to zero."},
    {"fill_", L3_PY_CFUNC(PyTensor_fill_), METH_VARARGS,
     "fill_(value: float) -> Tensor\nIn-place fill of all elements with a scalar value."},
    {"to_list", L3_PY_CFUNC(PyTensor_to_list), METH_NOARGS,
     "to_list() -> list[float | int]\nCopy flat tensor elements to a Python list."},
    {"item", L3_PY_CFUNC(PyTensor_item), METH_NOARGS,
     "item() -> float | int\nExtract the scalar value of a 1-element tensor."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyTensorType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.Tensor",                          // tp_name
    sizeof(PyTensor),                          // tp_basicsize
    0,                                         // tp_itemsize
    reinterpret_cast<destructor>(PyTensor_dealloc), // tp_dealloc
    0,                                         // tp_vectorcall_offset
    nullptr,                                   // tp_getattr
    nullptr,                                   // tp_setattr
    nullptr,                                   // tp_as_async
    reinterpret_cast<reprfunc>(PyTensor_repr), // tp_repr
    &PyTensor_as_number,                       // tp_as_number
    nullptr,                                   // tp_as_sequence
    nullptr,                                   // tp_as_mapping
    nullptr,                                   // tp_hash
    nullptr,                                   // tp_call
    nullptr,                                   // tp_str
    nullptr,                                   // tp_getattro
    nullptr,                                   // tp_setattro
    &PyTensor_as_buffer,                       // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  // tp_flags
    "64-byte aligned N-dimensional Tensor with BitNet b1.58 sub-byte packing,\n"
    "PEP 3118 zero-copy buffer protocol, and reverse-mode Autograd support.",
    nullptr,                                   // tp_traverse
    nullptr,                                   // tp_clear
    nullptr,                                   // tp_richcompare
    0,                                         // tp_weaklistoffset
    nullptr,                                   // tp_iter
    nullptr,                                   // tp_iternext
    PyTensor_methods,                          // tp_methods
    nullptr,                                   // tp_members
    PyTensor_getset,                           // tp_getset
    nullptr,                                   // tp_base
    nullptr,                                   // tp_dict
    nullptr,                                   // tp_descr_get
    nullptr,                                   // tp_descr_set
    0,                                         // tp_dictoffset
    reinterpret_cast<initproc>(PyTensor_init), // tp_init
    nullptr,                                   // tp_alloc
    PyType_GenericNew,                         // tp_new
};

// =============================================================================
// 2. PyRng Object (Deterministic Xoshiro256++ PRNG)
// =============================================================================
struct PyRng {
  PyObject_HEAD
  Rng rng;
};

void PyRng_dealloc(PyRng* self) noexcept {
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyRng_init(PyRng* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"seed", nullptr};
  unsigned long long seed = 0x853c49e6748fea9bULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|K", const_cast<char**>(kwlist), &seed)) {
    return -1;
  }
  new (&self->rng) Rng(static_cast<u64>(seed));
  return 0;
}

PyObject* PyRng_reseed(PyRng* self, PyObject* args) noexcept {
  unsigned long long seed = 0;
  if (!PyArg_ParseTuple(args, "K", &seed)) return nullptr;
  self->rng.reseed(static_cast<u64>(seed));
  Py_RETURN_NONE;
}

PyObject* PyRng_next_u64(PyRng* self, PyObject*) noexcept {
  return PyLong_FromUnsignedLongLong(self->rng.next_u64());
}

PyObject* PyRng_next_u32(PyRng* self, PyObject*) noexcept {
  return PyLong_FromUnsignedLong(self->rng.next_u32());
}

PyObject* PyRng_bounded_u32(PyRng* self, PyObject* args) noexcept {
  unsigned int bound = 1;
  if (!PyArg_ParseTuple(args, "I", &bound) || bound == 0) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "bound must be > 0");
    return nullptr;
  }
  return PyLong_FromUnsignedLong(self->rng.bounded_u32(bound));
}

PyObject* PyRng_uniform_f32(PyRng* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"low", "high", nullptr};
  double low = 0.0, high = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|dd", const_cast<char**>(kwlist), &low, &high)) {
    return nullptr;
  }
  return PyFloat_FromDouble(self->rng.uniform_f32(static_cast<f32>(low), static_cast<f32>(high)));
}

PyObject* PyRng_normal_f32(PyRng* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"mean", "stddev", nullptr};
  double mean = 0.0, stddev = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|dd", const_cast<char**>(kwlist), &mean, &stddev)) {
    return nullptr;
  }
  return PyFloat_FromDouble(self->rng.normal_f32(static_cast<f32>(mean), static_cast<f32>(stddev)));
}

PyObject* PyRng_fill_normal_(PyRng* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"tensor", "mean", "stddev", nullptr};
  PyObject* t_obj = nullptr;
  double mean = 0.0, stddev = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|dd", const_cast<char**>(kwlist),
                                   &t_obj, &mean, &stddev)) {
    return nullptr;
  }
  Tensor* t = nullptr;
  if (!check_tensor(t_obj, &t)) return nullptr;
  if (!t->is_valid() || !t->is_mutable() || !t->is_contiguous() || t->dtype() != DType::kFloat32) {
    PyErr_SetString(PyExc_ValueError, "fill_normal_ requires a contiguous mutable f32 Tensor.");
    return nullptr;
  }
  f32* ptr = t->mutable_data_ptr<f32>();
  for (Index i = 0; i < t->numel(); ++i) {
    ptr[i] = self->rng.normal_f32(static_cast<f32>(mean), static_cast<f32>(stddev));
  }
  Py_INCREF(t_obj);
  return t_obj;
}

PyObject* PyRng_fill_kaiming_normal_(PyRng* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"tensor", "fan_in", nullptr};
  PyObject* t_obj = nullptr;
  long long fan_in = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OL", const_cast<char**>(kwlist),
                                   &t_obj, &fan_in)) {
    return nullptr;
  }
  if (fan_in <= 0) {
    PyErr_SetString(PyExc_ValueError, "fan_in must be > 0.");
    return nullptr;
  }
  Tensor* t = nullptr;
  if (!check_tensor(t_obj, &t)) return nullptr;
  if (!t->is_valid() || !t->is_mutable() || !t->is_contiguous() || t->dtype() != DType::kFloat32) {
    PyErr_SetString(PyExc_ValueError, "fill_kaiming_normal_ requires a contiguous mutable f32 Tensor.");
    return nullptr;
  }
  const f32 std_dev = 1.0f / std::sqrt(static_cast<f32>(fan_in));
  f32* ptr = t->mutable_data_ptr<f32>();
  for (Index i = 0; i < t->numel(); ++i) {
    ptr[i] = self->rng.normal_f32(0.0f, std_dev);
  }
  Py_INCREF(t_obj);
  return t_obj;
}

PyMethodDef PyRng_methods[] = {
    {"reseed", L3_PY_CFUNC(PyRng_reseed), METH_VARARGS,
     "reseed(seed: int) -> None\nReset PRNG state from a 64-bit seed."},
    {"next_u64", L3_PY_CFUNC(PyRng_next_u64), METH_NOARGS,
     "next_u64() -> int\nDraw a 64-bit unsigned random integer."},
    {"next_u32", L3_PY_CFUNC(PyRng_next_u32), METH_NOARGS,
     "next_u32() -> int\nDraw a 32-bit unsigned random integer."},
    {"bounded_u32", L3_PY_CFUNC(PyRng_bounded_u32), METH_VARARGS,
     "bounded_u32(bound: int) -> int\nDraw an unbiased integer in [0, bound)."},
    {"uniform_f32", L3_PY_CFUNC(PyRng_uniform_f32),
     METH_VARARGS | METH_KEYWORDS,
     "uniform_f32(low: float = 0.0, high: float = 1.0) -> float\nBit-exact uniform float draw."},
    {"normal_f32", L3_PY_CFUNC(PyRng_normal_f32),
     METH_VARARGS | METH_KEYWORDS,
     "normal_f32(mean: float = 0.0, stddev: float = 1.0) -> float\nMarsaglia polar Gaussian draw."},
    {"fill_normal_", L3_PY_CFUNC(PyRng_fill_normal_),
     METH_VARARGS | METH_KEYWORDS,
     "fill_normal_(tensor: Tensor, mean: float = 0.0, stddev: float = 1.0) -> Tensor\nFill f32 tensor in-place with Gaussian noise."},
    {"fill_kaiming_normal_", L3_PY_CFUNC(PyRng_fill_kaiming_normal_),
     METH_VARARGS | METH_KEYWORDS,
     "fill_kaiming_normal_(tensor: Tensor, fan_in: int) -> Tensor\nFill f32 tensor in-place with N(0, 1/sqrt(fan_in))."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyRngType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.Rng",
    sizeof(PyRng),
    0,
    reinterpret_cast<destructor>(PyRng_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Deterministic cross-platform Xoshiro256++ pseudo-random number generator.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyRng_methods,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyRng_init),
    nullptr, PyType_GenericNew,
};

// =============================================================================
// 3. PyNoGradGuard Context Manager (`with log3.NoGradGuard():`)
// =============================================================================
struct PyNoGradGuard {
  PyObject_HEAD
  bool prev_state;
};

void PyNoGradGuard_dealloc(PyNoGradGuard* self) noexcept {
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyNoGradGuard_init(PyNoGradGuard* self, PyObject*, PyObject*) noexcept {
  self->prev_state = autograd::GradMode::is_enabled();
  return 0;
}

PyObject* PyNoGradGuard_enter(PyNoGradGuard* self, PyObject*) noexcept {
  self->prev_state = autograd::GradMode::is_enabled();
  autograd::GradMode::set_enabled(false);
  Py_INCREF(self);
  return reinterpret_cast<PyObject*>(self);
}

PyObject* PyNoGradGuard_exit(PyNoGradGuard* self, PyObject*) noexcept {
  autograd::GradMode::set_enabled(self->prev_state);
  Py_RETURN_FALSE;
}

PyMethodDef PyNoGradGuard_methods[] = {
    {"__enter__", L3_PY_CFUNC(PyNoGradGuard_enter), METH_NOARGS,
     "Disable autograd graph construction inside the with-block."},
    {"__exit__", L3_PY_CFUNC(PyNoGradGuard_exit), METH_VARARGS,
     "Restore previous autograd GradMode state."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyNoGradGuardType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.NoGradGuard",
    sizeof(PyNoGradGuard),
    0,
    reinterpret_cast<destructor>(PyNoGradGuard_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT,
    "RAII / Python context manager that disables Autograd graph construction.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyNoGradGuard_methods,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyNoGradGuard_init),
    nullptr, PyType_GenericNew,
};

// =============================================================================
// 4. Neural Network Modules (RMSNorm, BitLinear, Linear, Embedding, RoPE,
//    KVCache, GroupedQueryAttention, BitNetFeedForward, TransformerBlock,
//    BitNetTransformer)
// =============================================================================

// --- RMSNorm ---
struct PyRMSNorm {
  PyObject_HEAD
  nn::RMSNorm layer;
};

void PyRMSNorm_dealloc(PyRMSNorm* self) noexcept {
  self->layer.~RMSNorm();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyRMSNorm_init(PyRMSNorm* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"dim", "eps", nullptr};
  long long dim = 0;
  double eps = 1e-5;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "L|d", const_cast<char**>(kwlist), &dim, &eps)) {
    return -1;
  }
  new (&self->layer) nn::RMSNorm();
  auto res = nn::RMSNorm::create(static_cast<Index>(dim), static_cast<f32>(eps));
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->layer = res.take_value();
  return 0;
}

PyObject* PyRMSNorm_forward(PyRMSNorm* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"x", nullptr};
  PyObject* x_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &x_obj)) {
    return nullptr;
  }
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;
  auto out_res = Tensor::empty(x->shape(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  Status st = self->layer.forward(*x, &out);
  if (!st.is_ok()) return set_python_error(st);
  return wrap_tensor(std::move(out));
}

PyObject* PyRMSNorm_get_weight(PyRMSNorm* self, void*) noexcept {
  return wrap_tensor(self->layer.weight().view());
}

PyGetSetDef PyRMSNorm_getset[] = {
    {"weight", reinterpret_cast<getter>(PyRMSNorm_get_weight), nullptr,
     "Learnable scale parameter tensor of shape (dim,).", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef PyRMSNorm_methods[] = {
    {"forward", L3_PY_CFUNC(PyRMSNorm_forward),
     METH_VARARGS | METH_KEYWORDS,
     "forward(x: Tensor) -> Tensor\nApply Root Mean Square Layer Normalization."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyRMSNormType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.RMSNorm",
    sizeof(PyRMSNorm),
    0,
    reinterpret_cast<destructor>(PyRMSNorm_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    reinterpret_cast<ternaryfunc>(PyRMSNorm_forward),
    nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Root Mean Square Layer Normalization (RMSNorm) module.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyRMSNorm_methods, nullptr, PyRMSNorm_getset, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyRMSNorm_init),
    nullptr, PyType_GenericNew,
};

// --- BitLinear ---
struct PyBitLinear {
  PyObject_HEAD
  nn::BitLinear layer;
};

void PyBitLinear_dealloc(PyBitLinear* self) noexcept {
  self->layer.~BitLinear();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyBitLinear_init(PyBitLinear* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"in_features", "out_features", nullptr};
  long long in_f = 0, out_f = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "LL", const_cast<char**>(kwlist), &in_f, &out_f)) {
    return -1;
  }
  new (&self->layer) nn::BitLinear();
  auto res = nn::BitLinear::create(static_cast<Index>(in_f), static_cast<Index>(out_f));
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->layer = res.take_value();
  return 0;
}

PyObject* PyBitLinear_load_weights_from_f32(PyBitLinear* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"w_f32", nullptr};
  PyObject* w_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &w_obj)) {
    return nullptr;
  }
  Tensor* w = nullptr;
  if (!check_tensor(w_obj, &w)) return nullptr;
  Status st = self->layer.load_weights_from_f32(*w);
  if (!st.is_ok()) return set_python_error(st);
  Py_RETURN_NONE;
}

PyObject* PyBitLinear_forward(PyBitLinear* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"x", nullptr};
  PyObject* x_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &x_obj)) {
    return nullptr;
  }
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;

  Index out_dims[kMaxTensorDims];
  const Index r = x->rank();
  for (Index i = 0; i < r - 1; ++i) out_dims[i] = x->dim(i);
  out_dims[r - 1] = self->layer.out_features();

  auto s_res = Shape::from_dims(Span<const Index>(out_dims, r));
  if (!s_res.is_ok()) return set_python_error(s_res.status());
  auto out_res = Tensor::empty(s_res.take_value(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();

  Status st = self->layer.forward(*x, &out);
  if (!st.is_ok()) return set_python_error(st);
  return wrap_tensor(std::move(out));
}

PyObject* PyBitLinear_get_weight(PyBitLinear* self, void*) noexcept {
  return wrap_tensor(self->layer.weight().view());
}

PyObject* PyBitLinear_get_in_features(PyBitLinear* self, void*) noexcept {
  return PyLong_FromLongLong(self->layer.in_features());
}

PyObject* PyBitLinear_get_out_features(PyBitLinear* self, void*) noexcept {
  return PyLong_FromLongLong(self->layer.out_features());
}

PyGetSetDef PyBitLinear_getset[] = {
    {"weight", reinterpret_cast<getter>(PyBitLinear_get_weight), nullptr,
     "2-bit packed kTernary2 weight tensor of shape (out_features, in_features).", nullptr},
    {"in_features", reinterpret_cast<getter>(PyBitLinear_get_in_features), nullptr,
     "Input feature dimension (multiple of 4).", nullptr},
    {"out_features", reinterpret_cast<getter>(PyBitLinear_get_out_features), nullptr,
     "Output feature dimension.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef PyBitLinear_methods[] = {
    {"load_weights_from_f32", L3_PY_CFUNC(PyBitLinear_load_weights_from_f32),
     METH_VARARGS | METH_KEYWORDS,
     "load_weights_from_f32(w_f32: Tensor) -> None\nQuantize FP32 weights into 2-bit kTernary2 format in-place."},
    {"forward", L3_PY_CFUNC(PyBitLinear_forward),
     METH_VARARGS | METH_KEYWORDS,
     "forward(x: Tensor) -> Tensor\nCompute BitNet b1.58 ternary projection y = x @ W_t2^T * scale."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyBitLinearType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.BitLinear",
    sizeof(PyBitLinear),
    0,
    reinterpret_cast<destructor>(PyBitLinear_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    reinterpret_cast<ternaryfunc>(PyBitLinear_forward),
    nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "BitNet b1.58 2-bit ternary linear projection layer (4 weights/byte).",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyBitLinear_methods, nullptr, PyBitLinear_getset, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyBitLinear_init),
    nullptr, PyType_GenericNew,
};

// --- Linear (FP32 Dense) ---
struct PyLinear {
  PyObject_HEAD
  nn::Linear layer;
};

void PyLinear_dealloc(PyLinear* self) noexcept {
  self->layer.~Linear();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyLinear_init(PyLinear* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"in_features", "out_features", "has_bias", nullptr};
  long long in_f = 0, out_f = 0;
  int has_bias = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "LL|p", const_cast<char**>(kwlist),
                                   &in_f, &out_f, &has_bias)) {
    return -1;
  }
  new (&self->layer) nn::Linear();
  auto res = nn::Linear::create(static_cast<Index>(in_f), static_cast<Index>(out_f), has_bias != 0);
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->layer = res.take_value();
  return 0;
}

PyObject* PyLinear_forward(PyLinear* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"x", nullptr};
  PyObject* x_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &x_obj)) {
    return nullptr;
  }
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;

  Index out_dims[kMaxTensorDims];
  const Index r = x->rank();
  for (Index i = 0; i < r - 1; ++i) out_dims[i] = x->dim(i);
  out_dims[r - 1] = self->layer.out_features();

  auto s_res = Shape::from_dims(Span<const Index>(out_dims, r));
  if (!s_res.is_ok()) return set_python_error(s_res.status());
  auto out_res = Tensor::empty(s_res.take_value(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();

  Status st = self->layer.forward(*x, &out);
  if (!st.is_ok()) return set_python_error(st);
  return wrap_tensor(std::move(out));
}

PyObject* PyLinear_get_weight(PyLinear* self, void*) noexcept {
  return wrap_tensor(self->layer.weight().view());
}

PyObject* PyLinear_get_bias(PyLinear* self, void*) noexcept {
  if (!self->layer.bias().is_valid()) Py_RETURN_NONE;
  return wrap_tensor(self->layer.bias().view());
}

PyGetSetDef PyLinear_getset[] = {
    {"weight", reinterpret_cast<getter>(PyLinear_get_weight), nullptr,
     "FP32 weight tensor of shape (out_features, in_features).", nullptr},
    {"bias", reinterpret_cast<getter>(PyLinear_get_bias), nullptr,
     "Optional FP32 bias tensor of shape (out_features,).", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef PyLinear_methods[] = {
    {"forward", L3_PY_CFUNC(PyLinear_forward),
     METH_VARARGS | METH_KEYWORDS,
     "forward(x: Tensor) -> Tensor\nCompute dense FP32 linear projection."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyLinearType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.Linear",
    sizeof(PyLinear),
    0,
    reinterpret_cast<destructor>(PyLinear_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    reinterpret_cast<ternaryfunc>(PyLinear_forward),
    nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Standard dense FP32 linear projection layer.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyLinear_methods, nullptr, PyLinear_getset, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyLinear_init),
    nullptr, PyType_GenericNew,
};

// --- Embedding ---
struct PyEmbedding {
  PyObject_HEAD
  nn::Embedding layer;
};

void PyEmbedding_dealloc(PyEmbedding* self) noexcept {
  self->layer.~Embedding();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyEmbedding_init(PyEmbedding* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"vocab_size", "embedding_dim", nullptr};
  long long vs = 0, dim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "LL", const_cast<char**>(kwlist), &vs, &dim)) {
    return -1;
  }
  new (&self->layer) nn::Embedding();
  auto res = nn::Embedding::create(static_cast<Index>(vs), static_cast<Index>(dim));
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->layer = res.take_value();
  return 0;
}

PyObject* PyEmbedding_forward(PyEmbedding* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"tokens", nullptr};
  PyObject* tok_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &tok_obj)) {
    return nullptr;
  }
  SmallVec<Index, 128> tokens;
  if (!parse_index_vec(tok_obj, &tokens)) return nullptr;
  auto out_res = Tensor::empty({static_cast<Index>(tokens.size()), self->layer.embedding_dim()},
                               DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  Status st = self->layer.forward(Span<const Index>(tokens.data(), tokens.size()), &out);
  if (!st.is_ok()) return set_python_error(st);
  return wrap_tensor(std::move(out));
}

PyObject* PyEmbedding_get_weight(PyEmbedding* self, void*) noexcept {
  return wrap_tensor(self->layer.weight().view());
}

PyGetSetDef PyEmbedding_getset[] = {
    {"weight", reinterpret_cast<getter>(PyEmbedding_get_weight), nullptr,
     "Embedding table tensor of shape (vocab_size, embedding_dim).", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef PyEmbedding_methods[] = {
    {"forward", L3_PY_CFUNC(PyEmbedding_forward),
     METH_VARARGS | METH_KEYWORDS,
     "forward(tokens: Sequence[int]) -> Tensor\nLookup token embeddings of shape (seq_len, embedding_dim)."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyEmbeddingType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.Embedding",
    sizeof(PyEmbedding),
    0,
    reinterpret_cast<destructor>(PyEmbedding_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    reinterpret_cast<ternaryfunc>(PyEmbedding_forward),
    nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Token lookup table embedding module.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyEmbedding_methods, nullptr, PyEmbedding_getset, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyEmbedding_init),
    nullptr, PyType_GenericNew,
};

// --- RotaryEmbedding ---
struct PyRotaryEmbedding {
  PyObject_HEAD
  nn::RotaryEmbedding rope;
};

void PyRotaryEmbedding_dealloc(PyRotaryEmbedding* self) noexcept {
  self->rope.~RotaryEmbedding();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyRotaryEmbedding_init(PyRotaryEmbedding* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"head_dim", "theta_base", nullptr};
  long long h_dim = 0;
  double theta = 10000.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "L|d", const_cast<char**>(kwlist),
                                   &h_dim, &theta)) {
    return -1;
  }
  new (&self->rope) nn::RotaryEmbedding();
  auto res = nn::RotaryEmbedding::create(static_cast<Index>(h_dim), static_cast<f32>(theta));
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->rope = res.take_value();
  return 0;
}

PyObject* PyRotaryEmbedding_apply(PyRotaryEmbedding* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"qk", "start_pos", nullptr};
  PyObject* qk_obj = nullptr;
  long long start_pos = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|L", const_cast<char**>(kwlist),
                                   &qk_obj, &start_pos)) {
    return nullptr;
  }
  Tensor* qk = nullptr;
  if (!check_tensor(qk_obj, &qk)) return nullptr;
  Status st = self->rope.apply(qk, static_cast<Index>(start_pos));
  if (!st.is_ok()) return set_python_error(st);
  Py_INCREF(qk_obj);
  return qk_obj;
}

PyObject* PyRotaryEmbedding_get_head_dim(PyRotaryEmbedding* self, void*) noexcept {
  return PyLong_FromLongLong(self->rope.head_dim());
}

PyGetSetDef PyRotaryEmbedding_getset[] = {
    {"head_dim", reinterpret_cast<getter>(PyRotaryEmbedding_get_head_dim), nullptr,
     "Attention head dimension (even integer).", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef PyRotaryEmbedding_methods[] = {
    {"apply", L3_PY_CFUNC(PyRotaryEmbedding_apply),
     METH_VARARGS | METH_KEYWORDS,
     "apply(qk: Tensor, start_pos: int = 0) -> Tensor\nIn-place Rotary Position Embedding on (seq_len, num_heads, head_dim)."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyRotaryEmbeddingType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.RotaryEmbedding",
    sizeof(PyRotaryEmbedding),
    0,
    reinterpret_cast<destructor>(PyRotaryEmbedding_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Rotary Position Embedding (RoPE) module.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyRotaryEmbedding_methods, nullptr, PyRotaryEmbedding_getset, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyRotaryEmbedding_init),
    nullptr, PyType_GenericNew,
};

// --- KVCache ---
struct PyKVCache {
  PyObject_HEAD
  nn::KVCache cache;
};

void PyKVCache_dealloc(PyKVCache* self) noexcept {
  self->cache.~KVCache();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyKVCache_init(PyKVCache* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"max_seq_len", "num_kv_heads", "head_dim", nullptr};
  long long max_seq = 0, n_kv = 0, h_dim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "LLL", const_cast<char**>(kwlist),
                                   &max_seq, &n_kv, &h_dim)) {
    return -1;
  }
  new (&self->cache) nn::KVCache();
  auto res = nn::KVCache::create(static_cast<Index>(max_seq),
                                 static_cast<Index>(n_kv),
                                 static_cast<Index>(h_dim));
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->cache = res.take_value();
  return 0;
}

PyObject* PyKVCache_reset(PyKVCache* self, PyObject*) noexcept {
  self->cache.reset();
  Py_RETURN_NONE;
}

PyObject* PyKVCache_get_current_length(PyKVCache* self, void*) noexcept {
  return PyLong_FromLongLong(self->cache.current_length());
}

PyObject* PyKVCache_get_max_length(PyKVCache* self, void*) noexcept {
  return PyLong_FromLongLong(self->cache.max_length());
}

PyGetSetDef PyKVCache_getset[] = {
    {"current_length", reinterpret_cast<getter>(PyKVCache_get_current_length), nullptr,
     "Current number of cached sequence positions.", nullptr},
    {"max_length", reinterpret_cast<getter>(PyKVCache_get_max_length), nullptr,
     "Maximum sequence capacity of the KV cache.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef PyKVCache_methods[] = {
    {"reset", L3_PY_CFUNC(PyKVCache_reset), METH_NOARGS,
     "reset() -> None\nReset cached token count to 0 without deallocating memory."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyKVCacheType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.KVCache",
    sizeof(PyKVCache),
    0,
    reinterpret_cast<destructor>(PyKVCache_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Pre-allocated Key-Value Cache for zero-allocation autoregressive decoding.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyKVCache_methods, nullptr, PyKVCache_getset, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyKVCache_init),
    nullptr, PyType_GenericNew,
};

// --- GroupedQueryAttention ---
struct PyGQA {
  PyObject_HEAD
  nn::GroupedQueryAttention attn;
};

void PyGQA_dealloc(PyGQA* self) noexcept {
  self->attn.~GroupedQueryAttention();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyGQA_init(PyGQA* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"dim", "num_heads", "num_kv_heads", nullptr};
  long long dim = 0, nh = 0, nkv = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "LLL", const_cast<char**>(kwlist),
                                   &dim, &nh, &nkv)) {
    return -1;
  }
  new (&self->attn) nn::GroupedQueryAttention();
  auto res = nn::GroupedQueryAttention::create(static_cast<Index>(dim),
                                               static_cast<Index>(nh),
                                               static_cast<Index>(nkv));
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->attn = res.take_value();
  return 0;
}

PyObject* PyGQA_forward(PyGQA* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"x", "cache", nullptr};
  PyObject* x_obj = nullptr;
  PyObject* cache_obj = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist),
                                   &x_obj, &cache_obj)) {
    return nullptr;
  }
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;
  nn::KVCache* cache_ptr = nullptr;
  if (cache_obj != nullptr && cache_obj != Py_None) {
    if (!PyObject_TypeCheck(cache_obj, &PyKVCacheType)) {
      PyErr_SetString(PyExc_TypeError, "cache must be a log3.KVCache or None.");
      return nullptr;
    }
    cache_ptr = &reinterpret_cast<PyKVCache*>(cache_obj)->cache;
  }
  auto out_res = Tensor::empty(x->shape(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  Status st = self->attn.forward(*x, &out, cache_ptr);
  if (!st.is_ok()) return set_python_error(st);
  return wrap_tensor(std::move(out));
}

PyMethodDef PyGQA_methods[] = {
    {"forward", L3_PY_CFUNC(PyGQA_forward),
     METH_VARARGS | METH_KEYWORDS,
     "forward(x: Tensor, cache: Optional[KVCache] = None) -> Tensor\nRun BitNet b1.58 Grouped-Query Attention."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyGQAType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.GroupedQueryAttention",
    sizeof(PyGQA),
    0,
    reinterpret_cast<destructor>(PyGQA_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    reinterpret_cast<ternaryfunc>(PyGQA_forward),
    nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "BitNet b1.58 Grouped-Query Attention (GQA / MHA / MQA) module.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyGQA_methods, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyGQA_init),
    nullptr, PyType_GenericNew,
};

// --- BitNetFeedForward ---
struct PyBitNetFFN {
  PyObject_HEAD
  nn::BitNetFeedForward ffn;
};

void PyBitNetFFN_dealloc(PyBitNetFFN* self) noexcept {
  self->ffn.~BitNetFeedForward();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyBitNetFFN_init(PyBitNetFFN* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"dim", "hidden_dim", nullptr};
  long long dim = 0, h_dim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "LL", const_cast<char**>(kwlist),
                                   &dim, &h_dim)) {
    return -1;
  }
  new (&self->ffn) nn::BitNetFeedForward();
  auto res = nn::BitNetFeedForward::create(static_cast<Index>(dim), static_cast<Index>(h_dim));
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->ffn = res.take_value();
  return 0;
}

PyObject* PyBitNetFFN_forward(PyBitNetFFN* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"x", nullptr};
  PyObject* x_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &x_obj)) {
    return nullptr;
  }
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;
  auto out_res = Tensor::empty(x->shape(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  Status st = self->ffn.forward(*x, &out);
  if (!st.is_ok()) return set_python_error(st);
  return wrap_tensor(std::move(out));
}

PyMethodDef PyBitNetFFN_methods[] = {
    {"forward", L3_PY_CFUNC(PyBitNetFFN_forward),
     METH_VARARGS | METH_KEYWORDS,
     "forward(x: Tensor) -> Tensor\nRun SwiGLU BitNet b1.58 Feed-Forward block w2(silu(w1(x)) * w3(x))."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyBitNetFFNType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.BitNetFeedForward",
    sizeof(PyBitNetFFN),
    0,
    reinterpret_cast<destructor>(PyBitNetFFN_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    reinterpret_cast<ternaryfunc>(PyBitNetFFN_forward),
    nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "BitNet b1.58 SwiGLU Feed-Forward Network module.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyBitNetFFN_methods, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyBitNetFFN_init),
    nullptr, PyType_GenericNew,
};

// --- TransformerBlock ---
struct PyTransformerBlock {
  PyObject_HEAD
  nn::TransformerBlock block;
};

void PyTransformerBlock_dealloc(PyTransformerBlock* self) noexcept {
  self->block.~TransformerBlock();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyTransformerBlock_init(PyTransformerBlock* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {
      "layer_id", "vocab_size", "dim", "hidden_dim", "num_layers", "num_heads",
      "num_kv_heads", "max_seq_len", "norm_eps", "rope_theta", nullptr};
  long long layer_id = 0;
  long long vocab_size = 32000;
  long long dim = 512;
  long long hidden_dim = 1376;
  long long num_layers = 4;
  long long num_heads = 8;
  long long num_kv_heads = 4;
  long long max_seq_len = 2048;
  double norm_eps = 1e-5;
  double rope_theta = 10000.0;

  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "|LLLLLLLLdd", const_cast<char**>(kwlist),
          &layer_id, &vocab_size, &dim, &hidden_dim, &num_layers, &num_heads,
          &num_kv_heads, &max_seq_len, &norm_eps, &rope_theta)) {
    return -1;
  }

  nn::TransformerConfig cfg;
  cfg.vocab_size = static_cast<Index>(vocab_size);
  cfg.dim = static_cast<Index>(dim);
  cfg.hidden_dim = static_cast<Index>(hidden_dim);
  cfg.num_layers = static_cast<Index>(num_layers);
  cfg.num_heads = static_cast<Index>(num_heads);
  cfg.num_kv_heads = static_cast<Index>(num_kv_heads);
  cfg.max_seq_len = static_cast<Index>(max_seq_len);
  cfg.norm_eps = static_cast<f32>(norm_eps);
  cfg.rope_theta = static_cast<f32>(rope_theta);

  new (&self->block) nn::TransformerBlock();
  auto res = nn::TransformerBlock::create(static_cast<Index>(layer_id), cfg);
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->block = res.take_value();
  return 0;
}

PyObject* PyTransformerBlock_forward(PyTransformerBlock* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"x", "cache", nullptr};
  PyObject* x_obj = nullptr;
  PyObject* cache_obj = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist),
                                   &x_obj, &cache_obj)) {
    return nullptr;
  }
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;
  nn::KVCache* cache_ptr = nullptr;
  if (cache_obj != nullptr && cache_obj != Py_None) {
    if (!PyObject_TypeCheck(cache_obj, &PyKVCacheType)) {
      PyErr_SetString(PyExc_TypeError, "cache must be a log3.KVCache or None.");
      return nullptr;
    }
    cache_ptr = &reinterpret_cast<PyKVCache*>(cache_obj)->cache;
  }
  auto out_res = Tensor::empty(x->shape(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  Status st = self->block.forward(*x, &out, cache_ptr);
  if (!st.is_ok()) return set_python_error(st);
  return wrap_tensor(std::move(out));
}

PyMethodDef PyTransformerBlock_methods[] = {
    {"forward", L3_PY_CFUNC(PyTransformerBlock_forward),
     METH_VARARGS | METH_KEYWORDS,
     "forward(x: Tensor, cache: Optional[KVCache] = None) -> Tensor\nRun Pre-RMSNorm + GQA + Pre-RMSNorm + SwiGLU FFN with residuals."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyTransformerBlockType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.TransformerBlock",
    sizeof(PyTransformerBlock),
    0,
    reinterpret_cast<destructor>(PyTransformerBlock_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    reinterpret_cast<ternaryfunc>(PyTransformerBlock_forward),
    nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Pre-RMSNorm BitNet b1.58 Transformer decoder layer.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyTransformerBlock_methods, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyTransformerBlock_init),
    nullptr, PyType_GenericNew,
};

// --- BitNetTransformer ---
struct PyBitNetTransformer {
  PyObject_HEAD
  nn::BitNetTransformer model;
};

void PyBitNetTransformer_dealloc(PyBitNetTransformer* self) noexcept {
  self->model.~BitNetTransformer();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyBitNetTransformer_init(PyBitNetTransformer* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {
      "vocab_size", "dim", "hidden_dim", "num_layers", "num_heads",
      "num_kv_heads", "max_seq_len", "norm_eps", "rope_theta", nullptr};
  long long vocab_size = 32000;
  long long dim = 512;
  long long hidden_dim = 1376;
  long long num_layers = 4;
  long long num_heads = 8;
  long long num_kv_heads = 4;
  long long max_seq_len = 2048;
  double norm_eps = 1e-5;
  double rope_theta = 10000.0;

  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "|LLLLLLLdd", const_cast<char**>(kwlist),
          &vocab_size, &dim, &hidden_dim, &num_layers, &num_heads,
          &num_kv_heads, &max_seq_len, &norm_eps, &rope_theta)) {
    return -1;
  }

  nn::TransformerConfig cfg;
  cfg.vocab_size = static_cast<Index>(vocab_size);
  cfg.dim = static_cast<Index>(dim);
  cfg.hidden_dim = static_cast<Index>(hidden_dim);
  cfg.num_layers = static_cast<Index>(num_layers);
  cfg.num_heads = static_cast<Index>(num_heads);
  cfg.num_kv_heads = static_cast<Index>(num_kv_heads);
  cfg.max_seq_len = static_cast<Index>(max_seq_len);
  cfg.norm_eps = static_cast<f32>(norm_eps);
  cfg.rope_theta = static_cast<f32>(rope_theta);

  new (&self->model) nn::BitNetTransformer();
  auto res = nn::BitNetTransformer::create(cfg);
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->model = res.take_value();
  return 0;
}

PyObject* PyBitNetTransformer_forward(PyBitNetTransformer* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"tokens", nullptr};
  PyObject* tok_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &tok_obj)) {
    return nullptr;
  }
  SmallVec<Index, 128> tokens;
  if (!parse_index_vec(tok_obj, &tokens)) return nullptr;

  Tensor logits;
  Status st = self->model.forward(Span<const Index>(tokens.data(), tokens.size()), &logits);
  if (!st.is_ok()) return set_python_error(st);
  return wrap_tensor(std::move(logits));
}

PyObject* PyBitNetTransformer_generate(PyBitNetTransformer* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"prompt", "max_new_tokens", "temperature", nullptr};
  PyObject* prompt_obj = nullptr;
  long long max_new = 16;
  double temp = 0.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OL|d", const_cast<char**>(kwlist),
                                   &prompt_obj, &max_new, &temp)) {
    return nullptr;
  }
  SmallVec<Index, 128> prompt;
  if (!parse_index_vec(prompt_obj, &prompt)) return nullptr;

  SmallVec<Index, 128> out_tokens;
  Status st = self->model.generate(Span<const Index>(prompt.data(), prompt.size()),
                                   static_cast<Index>(max_new), &out_tokens,
                                   static_cast<f32>(temp));
  if (!st.is_ok()) return set_python_error(st);

  PyObject* lst = PyList_New(out_tokens.size());
  if (!lst) return nullptr;
  for (Index i = 0; i < out_tokens.size(); ++i) {
    PyList_SET_ITEM(lst, i, PyLong_FromLongLong(out_tokens[i]));
  }
  return lst;
}

PyObject* PyBitNetTransformer_get_embedding_weight(PyBitNetTransformer* self, void*) noexcept {
  return wrap_tensor(self->model.tok_embeddings().weight().view());
}

PyGetSetDef PyBitNetTransformer_getset[] = {
    {"embedding_weight", reinterpret_cast<getter>(PyBitNetTransformer_get_embedding_weight),
     nullptr, "Token embedding weight Tensor of shape (vocab_size, dim).", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef PyBitNetTransformer_methods[] = {
    {"forward", L3_PY_CFUNC(PyBitNetTransformer_forward),
     METH_VARARGS | METH_KEYWORDS,
     "forward(tokens: Sequence[int]) -> Tensor\nRun full BitNet Transformer forward pass returning logits (seq_len, vocab_size)."},
    {"generate", L3_PY_CFUNC(PyBitNetTransformer_generate),
     METH_VARARGS | METH_KEYWORDS,
     "generate(prompt: Sequence[int], max_new_tokens: int, temperature: float = 0.0) -> list[int]\nAutoregressively generate tokens with KV-caching."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyBitNetTransformerType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.BitNetTransformer",
    sizeof(PyBitNetTransformer),
    0,
    reinterpret_cast<destructor>(PyBitNetTransformer_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    reinterpret_cast<ternaryfunc>(PyBitNetTransformer_forward),
    nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Complete BitNet b1.58 Decoder Transformer with GQA and SwiGLU FFN.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyBitNetTransformer_methods, nullptr, PyBitNetTransformer_getset, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyBitNetTransformer_init),
    nullptr, PyType_GenericNew,
};

// =============================================================================
// 5. Zero-Copy .l3 Checkpoint I/O (ModelWriter & ModelReader)
// =============================================================================
struct PyModelWriter {
  PyObject_HEAD
  io::ModelWriter writer;
  PyObject* kept_names;
};

void PyModelWriter_dealloc(PyModelWriter* self) noexcept {
  Py_XDECREF(self->kept_names);
  self->writer.~ModelWriter();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyModelWriter_init(PyModelWriter* self, PyObject*, PyObject*) noexcept {
  new (&self->writer) io::ModelWriter();
  self->kept_names = PyList_New(0);
  return self->kept_names ? 0 : -1;
}

PyObject* PyModelWriter_add_tensor(PyModelWriter* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"name", "tensor", nullptr};
  PyObject* name_obj = nullptr;
  PyObject* t_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "UO", const_cast<char**>(kwlist),
                                   &name_obj, &t_obj)) {
    return nullptr;
  }
  Tensor* t = nullptr;
  if (!check_tensor(t_obj, &t)) return nullptr;
  const char* name_cstr = PyUnicode_AsUTF8(name_obj);
  if (!name_cstr) return nullptr;
  if (PyList_Append(self->kept_names, name_obj) < 0) return nullptr;

  Status st = self->writer.add_tensor(name_cstr, *t);
  if (!st.is_ok()) return set_python_error(st);
  Py_RETURN_NONE;
}

PyObject* PyModelWriter_save(PyModelWriter* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"filepath", nullptr};
  const char* filepath = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", const_cast<char**>(kwlist), &filepath)) {
    return nullptr;
  }
  Status st = self->writer.save(filepath);
  if (!st.is_ok()) return set_python_error(st);
  Py_RETURN_NONE;
}

PyMethodDef PyModelWriter_methods[] = {
    {"add_tensor", L3_PY_CFUNC(PyModelWriter_add_tensor),
     METH_VARARGS | METH_KEYWORDS,
     "add_tensor(name: str, tensor: Tensor) -> None\nRegister a named tensor for 64-byte aligned serialization."},
    {"save", L3_PY_CFUNC(PyModelWriter_save),
     METH_VARARGS | METH_KEYWORDS,
     "save(filepath: str) -> None\nWrite all registered tensors to a 64-byte aligned .l3 binary file."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyModelWriterType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.ModelWriter",
    sizeof(PyModelWriter),
    0,
    reinterpret_cast<destructor>(PyModelWriter_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "64-byte cache-line aligned .l3 model checkpoint writer.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyModelWriter_methods, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyModelWriter_init),
    nullptr, PyType_GenericNew,
};

struct PyModelReader {
  PyObject_HEAD
  io::ModelReader reader;
};

void PyModelReader_dealloc(PyModelReader* self) noexcept {
  self->reader.~ModelReader();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyModelReader_init(PyModelReader* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"filepath", "prefetch", nullptr};
  const char* filepath = nullptr;
  int prefetch = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|p", const_cast<char**>(kwlist),
                                   &filepath, &prefetch)) {
    return -1;
  }
  new (&self->reader) io::ModelReader();
  auto res = io::ModelReader::open(filepath, prefetch != 0);
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->reader = res.take_value();
  return 0;
}

PyObject* PyModelReader_open_static(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  PyModelReader* self = PyObject_New(PyModelReader, &PyModelReaderType);
  if (!self) return nullptr;
  if (PyModelReader_init(self, args, kwds) < 0) {
    Py_DECREF(self);
    return nullptr;
  }
  return reinterpret_cast<PyObject*>(self);
}

PyObject* PyModelReader_has_tensor(PyModelReader* self, PyObject* args) noexcept {
  const char* name = nullptr;
  Py_ssize_t len = 0;
  if (!PyArg_ParseTuple(args, "s#", &name, &len)) return nullptr;
  return PyBool_FromLong(self->reader.has_tensor(StrView(name, static_cast<Size>(len))));
}

PyObject* PyModelReader_get_tensor(PyModelReader* self, PyObject* args) noexcept {
  const char* name = nullptr;
  Py_ssize_t len = 0;
  if (!PyArg_ParseTuple(args, "s#", &name, &len)) return nullptr;
  return wrap_result_tensor(self->reader.get_tensor(StrView(name, static_cast<Size>(len))));
}

PyObject* PyModelReader_get_num_tensors(PyModelReader* self, void*) noexcept {
  return PyLong_FromLongLong(self->reader.num_tensors());
}

PyGetSetDef PyModelReader_getset[] = {
    {"num_tensors", reinterpret_cast<getter>(PyModelReader_get_num_tensors), nullptr,
     "Number of tensors stored in the memory-mapped .l3 checkpoint.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef PyModelReader_methods[] = {
    {"open", L3_PY_CFUNC(PyModelReader_open_static),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     "open(filepath: str, prefetch: bool = False) -> ModelReader\nMemory-map a .l3 checkpoint file with zero heap copies."},
    {"has_tensor", L3_PY_CFUNC(PyModelReader_has_tensor), METH_VARARGS,
     "has_tensor(name: str) -> bool\nReturn True if tensor exists in the checkpoint."},
    {"get_tensor", L3_PY_CFUNC(PyModelReader_get_tensor), METH_VARARGS,
     "get_tensor(name: str) -> Tensor\nReturn a zero-copy Tensor backed directly by the OS mmap."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyModelReaderType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.ModelReader",
    sizeof(PyModelReader),
    0,
    reinterpret_cast<destructor>(PyModelReader_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Zero-copy memory-mapped .l3 checkpoint reader.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyModelReader_methods, nullptr, PyModelReader_getset, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyModelReader_init),
    nullptr, PyType_GenericNew,
};

// =============================================================================
// 6. Optimizers (SGD & AdamW)
// =============================================================================
bool parse_tensor_list(PyObject* seq_obj, SmallVec<Tensor, 32>* out) noexcept {
  PyObject* seq = PySequence_Fast(seq_obj, "params must be a sequence of log3.Tensor objects");
  if (!seq) return false;
  const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  out->clear();
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
    Tensor* t = nullptr;
    if (!check_tensor(item, &t)) {
      Py_DECREF(seq);
      return false;
    }
    if (!out->push_back(t->view()).is_ok()) {
      Py_DECREF(seq);
      PyErr_NoMemory();
      return false;
    }
  }
  Py_DECREF(seq);
  return true;
}

struct PySGD {
  PyObject_HEAD
  optim::SGD opt;
};

void PySGD_dealloc(PySGD* self) noexcept {
  self->opt.~SGD();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PySGD_init(PySGD* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"params", "lr", "momentum", "weight_decay", nullptr};
  PyObject* params_obj = nullptr;
  double lr = 1e-2, momentum = 0.0, wd = 0.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ddd", const_cast<char**>(kwlist),
                                   &params_obj, &lr, &momentum, &wd)) {
    return -1;
  }
  SmallVec<Tensor, 32> params;
  if (!parse_tensor_list(params_obj, &params)) return -1;
  new (&self->opt) optim::SGD();
  auto res = optim::SGD::create(Span<const Tensor>(params.data(), params.size()),
                                static_cast<f32>(lr), static_cast<f32>(momentum),
                                static_cast<f32>(wd));
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->opt = res.take_value();
  return 0;
}

PyObject* PySGD_zero_grad(PySGD* self, PyObject*) noexcept {
  self->opt.zero_grad();
  Py_RETURN_NONE;
}

PyObject* PySGD_step(PySGD* self, PyObject*) noexcept {
  Status st = self->opt.step();
  if (!st.is_ok()) return set_python_error(st);
  Py_RETURN_NONE;
}

PyObject* PySGD_get_lr(PySGD* self, void*) noexcept {
  return PyFloat_FromDouble(static_cast<double>(self->opt.lr()));
}

int PySGD_set_lr(PySGD* self, PyObject* val, void*) noexcept {
  double v = PyFloat_AsDouble(val);
  if (PyErr_Occurred()) return -1;
  self->opt.set_lr(static_cast<f32>(v));
  return 0;
}

PyGetSetDef PySGD_getset[] = {
    {"lr", reinterpret_cast<getter>(PySGD_get_lr), reinterpret_cast<setter>(PySGD_set_lr),
     "Learning rate.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef PySGD_methods[] = {
    {"zero_grad", L3_PY_CFUNC(PySGD_zero_grad), METH_NOARGS,
     "zero_grad() -> None\nZero all tracked parameter gradients."},
    {"step", L3_PY_CFUNC(PySGD_step), METH_NOARGS,
     "step() -> None\nPerform a single SGD parameter update step."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PySGDType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.SGD",
    sizeof(PySGD),
    0,
    reinterpret_cast<destructor>(PySGD_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Stochastic Gradient Descent optimizer with optional momentum and weight decay.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PySGD_methods, nullptr, PySGD_getset, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PySGD_init),
    nullptr, PyType_GenericNew,
};

struct PyAdamW {
  PyObject_HEAD
  optim::AdamW opt;
};

void PyAdamW_dealloc(PyAdamW* self) noexcept {
  self->opt.~AdamW();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int PyAdamW_init(PyAdamW* self, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"params", "lr", "beta1", "beta2", "eps", "weight_decay", nullptr};
  PyObject* params_obj = nullptr;
  double lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, wd = 0.01;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ddddd", const_cast<char**>(kwlist),
                                   &params_obj, &lr, &beta1, &beta2, &eps, &wd)) {
    return -1;
  }
  SmallVec<Tensor, 32> params;
  if (!parse_tensor_list(params_obj, &params)) return -1;
  new (&self->opt) optim::AdamW();
  auto res = optim::AdamW::create(Span<const Tensor>(params.data(), params.size()),
                                  static_cast<f32>(lr), static_cast<f32>(beta1),
                                  static_cast<f32>(beta2), static_cast<f32>(eps),
                                  static_cast<f32>(wd));
  if (!res.is_ok()) {
    set_python_error(res.status());
    return -1;
  }
  self->opt = res.take_value();
  return 0;
}

PyObject* PyAdamW_zero_grad(PyAdamW* self, PyObject*) noexcept {
  self->opt.zero_grad();
  Py_RETURN_NONE;
}

PyObject* PyAdamW_step(PyAdamW* self, PyObject*) noexcept {
  Status st = self->opt.step();
  if (!st.is_ok()) return set_python_error(st);
  Py_RETURN_NONE;
}

PyObject* PyAdamW_get_lr(PyAdamW* self, void*) noexcept {
  return PyFloat_FromDouble(static_cast<double>(self->opt.lr()));
}

int PyAdamW_set_lr(PyAdamW* self, PyObject* val, void*) noexcept {
  double v = PyFloat_AsDouble(val);
  if (PyErr_Occurred()) return -1;
  self->opt.set_lr(static_cast<f32>(v));
  return 0;
}

PyObject* PyAdamW_get_step_count(PyAdamW* self, void*) noexcept {
  return PyLong_FromLongLong(self->opt.step_count());
}

PyGetSetDef PyAdamW_getset[] = {
    {"lr", reinterpret_cast<getter>(PyAdamW_get_lr), reinterpret_cast<setter>(PyAdamW_set_lr),
     "Learning rate.", nullptr},
    {"step_count", reinterpret_cast<getter>(PyAdamW_get_step_count), nullptr,
     "Number of completed optimization steps.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef PyAdamW_methods[] = {
    {"zero_grad", L3_PY_CFUNC(PyAdamW_zero_grad), METH_NOARGS,
     "zero_grad() -> None\nZero all tracked parameter gradients."},
    {"step", L3_PY_CFUNC(PyAdamW_step), METH_NOARGS,
     "step() -> None\nPerform a bias-corrected AdamW update step."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyAdamWType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "log3._C.AdamW",
    sizeof(PyAdamW),
    0,
    reinterpret_cast<destructor>(PyAdamW_dealloc),
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Decoupled Weight Decay AdamW optimizer.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    PyAdamW_methods, nullptr, PyAdamW_getset, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(PyAdamW_init),
    nullptr, PyType_GenericNew,
};

// =============================================================================
// 7. Module-Level Functions (Core, Tensor Factories, SIMD Kernels, Autograd, Optim)
// =============================================================================

PyObject* mod_init_runtime(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"min_level", nullptr};
  int min_level = static_cast<int>(LogLevel::kInfo);
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", const_cast<char**>(kwlist), &min_level)) {
    return nullptr;
  }
  if (min_level < 0) min_level = 0;
  if (min_level > 6) min_level = 6;
  init_runtime(static_cast<LogLevel>(min_level));
  Py_RETURN_NONE;
}

PyObject* mod_set_log_level(PyObject*, PyObject* args) noexcept {
  int level = 2;
  if (!PyArg_ParseTuple(args, "i", &level)) return nullptr;
  if (level < 0) level = 0;
  if (level > 6) level = 6;
  set_log_level(static_cast<LogLevel>(level));
  Py_RETURN_NONE;
}

PyObject* mod_log_level(PyObject*, PyObject*) noexcept {
  return PyLong_FromLong(static_cast<long>(log_level()));
}

PyObject* mod_version_string(PyObject*, PyObject*) noexcept {
  return PyUnicode_FromString(version_string());
}

PyObject* mod_version_hex(PyObject*, PyObject*) noexcept {
  return PyLong_FromUnsignedLong(version_hex());
}

PyObject* mod_platform_name(PyObject*, PyObject*) noexcept {
  return PyUnicode_FromString(platform_name());
}

PyObject* mod_arch_name(PyObject*, PyObject*) noexcept {
  return PyUnicode_FromString(arch_name());
}

PyObject* mod_cpu_features(PyObject*, PyObject*) noexcept {
  const CpuFeatures& c = cpu_features();
  return Py_BuildValue(
      "{s:i,s:i,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:n,s:n,s:n,s:n}",
      "logical_cores", c.logical_cores,
      "physical_cores", c.physical_cores,
      "has_sse2", c.has_sse2 ? Py_True : Py_False,
      "has_avx", c.has_avx ? Py_True : Py_False,
      "has_avx2", c.has_avx2 ? Py_True : Py_False,
      "has_fma", c.has_fma ? Py_True : Py_False,
      "has_avx512f", c.has_avx512f ? Py_True : Py_False,
      "has_neon", c.has_neon ? Py_True : Py_False,
      "has_neon_dotprod", c.has_neon_dotprod ? Py_True : Py_False,
      "l1d_cache_size", static_cast<Py_ssize_t>(c.l1d_cache_size),
      "l2_cache_size", static_cast<Py_ssize_t>(c.l2_cache_size),
      "l3_cache_size", static_cast<Py_ssize_t>(c.l3_cache_size),
      "cache_line_size", static_cast<Py_ssize_t>(c.cache_line_size));
}

PyObject* mod_memory_stats(PyObject*, PyObject*) noexcept {
  const MemoryStats m = memory_stats();
  return Py_BuildValue(
      "{s:K,s:K,s:K,s:K}",
      "current_bytes", static_cast<unsigned long long>(m.current_bytes),
      "peak_bytes", static_cast<unsigned long long>(m.peak_bytes),
      "total_allocations", static_cast<unsigned long long>(m.total_allocations),
      "total_deallocations", static_cast<unsigned long long>(m.total_deallocations));
}

PyObject* mod_reset_peak_memory(PyObject*, PyObject*) noexcept {
  reset_peak_memory();
  Py_RETURN_NONE;
}

// --- Tensor Factory Functions ---
PyObject* mod_empty(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"shape", "dtype", "requires_grad", nullptr};
  PyObject* shape_obj = nullptr;
  PyObject* dtype_obj = nullptr;
  int req_grad = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|Op", const_cast<char**>(kwlist),
                                   &shape_obj, &dtype_obj, &req_grad)) {
    return nullptr;
  }
  Shape s;
  if (!parse_shape_obj(shape_obj, &s)) return nullptr;
  DType dt = DType::kFloat32;
  if (!parse_dtype_obj(dtype_obj, &dt)) return nullptr;
  auto res = Tensor::empty(s, dt);
  if (!res.is_ok()) return set_python_error(res.status());
  Tensor t = res.take_value();
  if (req_grad) {
    Status st = t.set_requires_grad(true);
    if (!st.is_ok()) return set_python_error(st);
  }
  return wrap_tensor(std::move(t));
}

PyObject* mod_zeros(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"shape", "dtype", "requires_grad", nullptr};
  PyObject* shape_obj = nullptr;
  PyObject* dtype_obj = nullptr;
  int req_grad = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|Op", const_cast<char**>(kwlist),
                                   &shape_obj, &dtype_obj, &req_grad)) {
    return nullptr;
  }
  Shape s;
  if (!parse_shape_obj(shape_obj, &s)) return nullptr;
  DType dt = DType::kFloat32;
  if (!parse_dtype_obj(dtype_obj, &dt)) return nullptr;
  auto res = Tensor::zeros(s, dt);
  if (!res.is_ok()) return set_python_error(res.status());
  Tensor t = res.take_value();
  if (req_grad) {
    Status st = t.set_requires_grad(true);
    if (!st.is_ok()) return set_python_error(st);
  }
  return wrap_tensor(std::move(t));
}

PyObject* mod_ones(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"shape", "dtype", "requires_grad", nullptr};
  PyObject* shape_obj = nullptr;
  PyObject* dtype_obj = nullptr;
  int req_grad = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|Op", const_cast<char**>(kwlist),
                                   &shape_obj, &dtype_obj, &req_grad)) {
    return nullptr;
  }
  Shape s;
  if (!parse_shape_obj(shape_obj, &s)) return nullptr;
  DType dt = DType::kFloat32;
  if (!parse_dtype_obj(dtype_obj, &dt)) return nullptr;
  auto res = Tensor::full(s, 1.0f, dt);
  if (!res.is_ok()) return set_python_error(res.status());
  Tensor t = res.take_value();
  if (req_grad) {
    Status st = t.set_requires_grad(true);
    if (!st.is_ok()) return set_python_error(st);
  }
  return wrap_tensor(std::move(t));
}

PyObject* mod_full(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"shape", "fill_value", "dtype", "requires_grad", nullptr};
  PyObject* shape_obj = nullptr;
  double val = 0.0;
  PyObject* dtype_obj = nullptr;
  int req_grad = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Od|Op", const_cast<char**>(kwlist),
                                   &shape_obj, &val, &dtype_obj, &req_grad)) {
    return nullptr;
  }
  Shape s;
  if (!parse_shape_obj(shape_obj, &s)) return nullptr;
  DType dt = DType::kFloat32;
  if (!parse_dtype_obj(dtype_obj, &dt)) return nullptr;
  auto res = Tensor::full(s, static_cast<f32>(val), dt);
  if (!res.is_ok()) return set_python_error(res.status());
  Tensor t = res.take_value();
  if (req_grad) {
    Status st = t.set_requires_grad(true);
    if (!st.is_ok()) return set_python_error(st);
  }
  return wrap_tensor(std::move(t));
}

PyObject* mod_from_list(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"values", "shape", "dtype", "requires_grad", nullptr};
  PyObject* vals_obj = nullptr;
  PyObject* shape_obj = nullptr;
  PyObject* dtype_obj = nullptr;
  int req_grad = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|Op", const_cast<char**>(kwlist),
                                   &vals_obj, &shape_obj, &dtype_obj, &req_grad)) {
    return nullptr;
  }
  Shape s;
  if (!parse_shape_obj(shape_obj, &s)) return nullptr;
  DType dt = DType::kFloat32;
  if (!parse_dtype_obj(dtype_obj, &dt)) return nullptr;

  PyObject* seq = PySequence_Fast(vals_obj, "values must be a flat sequence of numbers");
  if (!seq) return nullptr;
  const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  if (static_cast<Index>(n) != s.numel()) {
    Py_DECREF(seq);
    PyErr_Format(PyExc_ValueError, "values length (%zd) does not match shape numel (%lld)",
                 n, static_cast<long long>(s.numel()));
    return nullptr;
  }

  auto res = Tensor::empty(s, dt);
  if (!res.is_ok()) {
    Py_DECREF(seq);
    return set_python_error(res.status());
  }
  Tensor t = res.take_value();

  if (dt == DType::kFloat32) {
    f32* p = t.mutable_data_ptr<f32>();
    for (Py_ssize_t i = 0; i < n; ++i) {
      p[i] = static_cast<f32>(PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i)));
    }
  } else if (dt == DType::kInt32) {
    i32* p = t.mutable_data_ptr<i32>();
    for (Py_ssize_t i = 0; i < n; ++i) {
      p[i] = static_cast<i32>(PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i)));
    }
  } else if (dt == DType::kInt64) {
    i64* p = t.mutable_data_ptr<i64>();
    for (Py_ssize_t i = 0; i < n; ++i) {
      p[i] = static_cast<i64>(PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, i)));
    }
  } else {
    Py_DECREF(seq);
    PyErr_Format(PyExc_NotImplementedError, "from_list does not support dtype %s",
                 dtype_name(dt));
    return nullptr;
  }
  Py_DECREF(seq);
  if (PyErr_Occurred()) return nullptr;
  if (req_grad) {
    Status st = t.set_requires_grad(true);
    if (!st.is_ok()) return set_python_error(st);
  }
  return wrap_tensor(std::move(t));
}

PyObject* mod_from_buffer(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"buffer", "shape", "dtype", "requires_grad", nullptr};
  PyObject* buf_obj = nullptr;
  PyObject* shape_obj = Py_None;
  PyObject* dtype_obj = nullptr;
  int req_grad = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp", const_cast<char**>(kwlist),
                                   &buf_obj, &shape_obj, &dtype_obj, &req_grad)) {
    return nullptr;
  }

  Py_buffer view;
  if (PyObject_GetBuffer(buf_obj, &view, PyBUF_SIMPLE) < 0) {
    return nullptr;
  }

  DType dt = DType::kFloat32;
  if (!parse_dtype_obj(dtype_obj, &dt)) {
    PyBuffer_Release(&view);
    return nullptr;
  }

  Shape s;
  if (shape_obj != nullptr && shape_obj != Py_None) {
    if (!parse_shape_obj(shape_obj, &s)) {
      PyBuffer_Release(&view);
      return nullptr;
    }
  } else {
    const Size elem_size = dtype_info(dt).is_subbyte ? 0 : static_cast<Size>(dtype_info(dt).block_bytes);
    if (elem_size == 0 || view.len % static_cast<Py_ssize_t>(elem_size) != 0) {
      PyBuffer_Release(&view);
      PyErr_SetString(PyExc_ValueError, "Explicit shape is required for sub-byte or unaligned buffers.");
      return nullptr;
    }
    const Index count = static_cast<Index>(view.len / static_cast<Py_ssize_t>(elem_size));
    auto s_res = Shape::from_dims({count});
    s = s_res.take_value();
  }

  auto t_res = Tensor::empty(s, dt);
  if (!t_res.is_ok()) {
    PyBuffer_Release(&view);
    return set_python_error(t_res.status());
  }
  Tensor t = t_res.take_value();
  if (static_cast<Size>(view.len) != t.nbytes()) {
    PyBuffer_Release(&view);
    PyErr_Format(PyExc_ValueError, "Buffer byte length (%zd) != Tensor nbytes (%zu)",
                 view.len, t.nbytes());
    return nullptr;
  }
  mem_copy(t.raw_mutable_data(), view.buf, t.nbytes());
  PyBuffer_Release(&view);

  if (req_grad) {
    Status st = t.set_requires_grad(true);
    if (!st.is_ok()) return set_python_error(st);
  }
  return wrap_tensor(std::move(t));
}

// --- Low-Level SIMD & BitNet Kernels ---
PyObject* mod_quantize_f32_to_t2(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"weights_f32", nullptr};
  PyObject* w_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &w_obj)) {
    return nullptr;
  }
  Tensor* w = nullptr;
  if (!check_tensor(w_obj, &w)) return nullptr;
  return wrap_result_tensor(kernels::quantize_f32_to_t2(*w));
}

PyObject* mod_dequantize_t2_to_f32(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"weights_t2", nullptr};
  PyObject* w_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &w_obj)) {
    return nullptr;
  }
  Tensor* w = nullptr;
  if (!check_tensor(w_obj, &w)) return nullptr;
  if (!w->is_valid() || w->dtype() != DType::kTernary2 || !w->is_contiguous()) {
    PyErr_SetString(PyExc_ValueError, "dequantize_t2_to_f32 requires a contiguous kTernary2 tensor.");
    return nullptr;
  }
  auto out_res = Tensor::empty(w->shape(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  kernels::dequantize_t2_to_f32_raw(w->raw_data(), out.mutable_data_ptr<f32>(),
                                    w->numel(), w->quant_scale());
  return wrap_tensor(std::move(out));
}

PyObject* mod_bitnet_linear_t2_f32(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"x", "w_t2", nullptr};
  PyObject *x_obj = nullptr, *w_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist), &x_obj, &w_obj)) {
    return nullptr;
  }
  Tensor *x = nullptr, *w = nullptr;
  if (!check_tensor(x_obj, &x) || !check_tensor(w_obj, &w)) return nullptr;
  if (x->rank() != 2 || w->rank() != 2) {
    PyErr_SetString(PyExc_ValueError, "bitnet_linear_t2_f32 expects 2D tensors x(M, K) and w_t2(N, K).");
    return nullptr;
  }
  auto out_res = Tensor::empty({x->dim(0), w->dim(0)}, DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  Status st = kernels::bitnet_linear_t2_f32(*x, *w, &out);
  if (!st.is_ok()) return set_python_error(st);
  return wrap_tensor(std::move(out));
}

PyObject* mod_dot_f32(PyObject*, PyObject* args) noexcept {
  PyObject *a_obj = nullptr, *b_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return nullptr;
  Tensor *a = nullptr, *b = nullptr;
  if (!check_tensor(a_obj, &a) || !check_tensor(b_obj, &b)) return nullptr;
  if (!a->is_contiguous() || !b->is_contiguous() || a->dtype() != DType::kFloat32 ||
      b->dtype() != DType::kFloat32 || a->numel() != b->numel()) {
    PyErr_SetString(PyExc_ValueError, "dot_f32 requires two contiguous f32 tensors with equal numel.");
    return nullptr;
  }
  f32 d = kernels::dot_f32(a->data_ptr<f32>(), b->data_ptr<f32>(), a->numel());
  return PyFloat_FromDouble(static_cast<double>(d));
}

PyObject* mod_add_f32(PyObject*, PyObject* args) noexcept {
  PyObject *a_obj = nullptr, *b_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return nullptr;
  Tensor *a = nullptr, *b = nullptr;
  if (!check_tensor(a_obj, &a) || !check_tensor(b_obj, &b)) return nullptr;
  if (!a->is_contiguous() || !b->is_contiguous() || a->dtype() != DType::kFloat32 ||
      b->dtype() != DType::kFloat32 || a->shape() != b->shape()) {
    PyErr_SetString(PyExc_ValueError, "add_f32 requires two contiguous f32 tensors of identical shape.");
    return nullptr;
  }
  auto out_res = Tensor::empty(a->shape(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  kernels::add_f32(a->data_ptr<f32>(), b->data_ptr<f32>(),
                   out.mutable_data_ptr<f32>(), a->numel());
  return wrap_tensor(std::move(out));
}

PyObject* mod_mul_f32(PyObject*, PyObject* args) noexcept {
  PyObject *a_obj = nullptr, *b_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return nullptr;
  Tensor *a = nullptr, *b = nullptr;
  if (!check_tensor(a_obj, &a) || !check_tensor(b_obj, &b)) return nullptr;
  if (!a->is_contiguous() || !b->is_contiguous() || a->dtype() != DType::kFloat32 ||
      b->dtype() != DType::kFloat32 || a->shape() != b->shape()) {
    PyErr_SetString(PyExc_ValueError, "mul_f32 requires two contiguous f32 tensors of identical shape.");
    return nullptr;
  }
  auto out_res = Tensor::empty(a->shape(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  kernels::mul_f32(a->data_ptr<f32>(), b->data_ptr<f32>(),
                   out.mutable_data_ptr<f32>(), a->numel());
  return wrap_tensor(std::move(out));
}

PyObject* mod_silu_f32(PyObject*, PyObject* args) noexcept {
  PyObject* x_obj = nullptr;
  if (!PyArg_ParseTuple(args, "O", &x_obj)) return nullptr;
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;
  if (!x->is_contiguous() || x->dtype() != DType::kFloat32) {
    PyErr_SetString(PyExc_ValueError, "silu_f32 requires a contiguous f32 Tensor.");
    return nullptr;
  }
  auto out_res = Tensor::empty(x->shape(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  kernels::silu_f32(x->data_ptr<f32>(), out.mutable_data_ptr<f32>(), x->numel());
  return wrap_tensor(std::move(out));
}

PyObject* mod_rmsnorm_f32(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"x", "weight", "eps", nullptr};
  PyObject *x_obj = nullptr, *w_obj = nullptr;
  double eps = 1e-5;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|d", const_cast<char**>(kwlist),
                                   &x_obj, &w_obj, &eps)) {
    return nullptr;
  }
  Tensor *x = nullptr, *w = nullptr;
  if (!check_tensor(x_obj, &x) || !check_tensor(w_obj, &w)) return nullptr;
  if (!x->is_contiguous() || !w->is_contiguous() || x->dtype() != DType::kFloat32 ||
      w->dtype() != DType::kFloat32 || x->rank() < 1 || w->rank() != 1 ||
      x->dim(x->rank() - 1) != w->dim(0)) {
    PyErr_SetString(PyExc_ValueError, "rmsnorm_f32 requires contiguous f32 x(..., D) and weight(D).");
    return nullptr;
  }
  auto out_res = Tensor::empty(x->shape(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  const Index cols = w->dim(0);
  const Index rows = x->numel() / cols;
  for (Index r = 0; r < rows; ++r) {
    kernels::rmsnorm_f32(x->data_ptr<f32>() + r * cols, w->data_ptr<f32>(),
                         out.mutable_data_ptr<f32>() + r * cols, cols,
                         static_cast<f32>(eps));
  }
  return wrap_tensor(std::move(out));
}

PyObject* mod_softmax_f32(PyObject*, PyObject* args) noexcept {
  PyObject* x_obj = nullptr;
  if (!PyArg_ParseTuple(args, "O", &x_obj)) return nullptr;
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;
  if (!x->is_contiguous() || x->dtype() != DType::kFloat32 || x->rank() < 1) {
    PyErr_SetString(PyExc_ValueError, "softmax_f32 requires a contiguous f32 Tensor.");
    return nullptr;
  }
  auto out_res = Tensor::empty(x->shape(), DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  const Index cols = x->dim(x->rank() - 1);
  const Index rows = x->numel() / cols;
  for (Index r = 0; r < rows; ++r) {
    kernels::softmax_f32(x->data_ptr<f32>() + r * cols,
                         out.mutable_data_ptr<f32>() + r * cols, cols);
  }
  return wrap_tensor(std::move(out));
}

PyObject* mod_rope_f32_inplace(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"qk", "start_pos", "theta_base", nullptr};
  PyObject* qk_obj = nullptr;
  long long start_pos = 0;
  double theta = 10000.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|Ld", const_cast<char**>(kwlist),
                                   &qk_obj, &start_pos, &theta)) {
    return nullptr;
  }
  Tensor* qk = nullptr;
  if (!check_tensor(qk_obj, &qk)) return nullptr;
  auto rope_res = nn::RotaryEmbedding::create(qk->dim(qk->rank() - 1), static_cast<f32>(theta));
  if (!rope_res.is_ok()) return set_python_error(rope_res.status());
  nn::RotaryEmbedding rope = rope_res.take_value();
  Status st = rope.apply(qk, static_cast<Index>(start_pos));
  if (!st.is_ok()) return set_python_error(st);
  Py_INCREF(qk_obj);
  return qk_obj;
}

PyObject* mod_matmul_f32(PyObject*, PyObject* args) noexcept {
  PyObject *a_obj = nullptr, *b_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return nullptr;
  Tensor *a = nullptr, *b = nullptr;
  if (!check_tensor(a_obj, &a) || !check_tensor(b_obj, &b)) return nullptr;
  if (a->rank() != 2 || b->rank() != 2) {
    PyErr_SetString(PyExc_ValueError, "matmul_f32 requires 2D tensors A(M, K) and B(K, N).");
    return nullptr;
  }
  auto out_res = Tensor::empty({a->dim(0), b->dim(1)}, DType::kFloat32);
  if (!out_res.is_ok()) return set_python_error(out_res.status());
  Tensor out = out_res.take_value();
  Status st = kernels::matmul_f32(*a, *b, &out);
  if (!st.is_ok()) return set_python_error(st);
  return wrap_tensor(std::move(out));
}

// --- Reverse-Mode Autograd & BitNet STE QAT Operations ---
PyObject* mod_is_grad_enabled(PyObject*, PyObject*) noexcept {
  return PyBool_FromLong(autograd::GradMode::is_enabled());
}

PyObject* mod_set_grad_enabled(PyObject*, PyObject* args) noexcept {
  int enabled = 1;
  if (!PyArg_ParseTuple(args, "p", &enabled)) return nullptr;
  autograd::GradMode::set_enabled(enabled != 0);
  Py_RETURN_NONE;
}

PyObject* mod_add(PyObject*, PyObject* args) noexcept {
  PyObject *a_obj = nullptr, *b_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return nullptr;
  Tensor *a = nullptr, *b = nullptr;
  if (!check_tensor(a_obj, &a) || !check_tensor(b_obj, &b)) return nullptr;
  return wrap_result_tensor(autograd::add(*a, *b));
}

PyObject* mod_sub(PyObject*, PyObject* args) noexcept {
  PyObject *a_obj = nullptr, *b_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return nullptr;
  Tensor *a = nullptr, *b = nullptr;
  if (!check_tensor(a_obj, &a) || !check_tensor(b_obj, &b)) return nullptr;
  return wrap_result_tensor(autograd::sub(*a, *b));
}

PyObject* mod_mul(PyObject*, PyObject* args) noexcept {
  PyObject *a_obj = nullptr, *b_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return nullptr;
  Tensor *a = nullptr, *b = nullptr;
  if (!check_tensor(a_obj, &a) || !check_tensor(b_obj, &b)) return nullptr;
  return wrap_result_tensor(autograd::mul(*a, *b));
}

PyObject* mod_silu(PyObject*, PyObject* args) noexcept {
  PyObject* x_obj = nullptr;
  if (!PyArg_ParseTuple(args, "O", &x_obj)) return nullptr;
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;
  return wrap_result_tensor(autograd::silu(*x));
}

PyObject* mod_relu(PyObject*, PyObject* args) noexcept {
  PyObject* x_obj = nullptr;
  if (!PyArg_ParseTuple(args, "O", &x_obj)) return nullptr;
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;
  return wrap_result_tensor(autograd::relu(*x));
}

PyObject* mod_rmsnorm(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"x", "weight", "eps", nullptr};
  PyObject *x_obj = nullptr, *w_obj = nullptr;
  double eps = 1e-5;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|d", const_cast<char**>(kwlist),
                                   &x_obj, &w_obj, &eps)) {
    return nullptr;
  }
  Tensor *x = nullptr, *w = nullptr;
  if (!check_tensor(x_obj, &x) || !check_tensor(w_obj, &w)) return nullptr;
  return wrap_result_tensor(autograd::rmsnorm(*x, *w, static_cast<f32>(eps)));
}

PyObject* mod_matmul(PyObject*, PyObject* args) noexcept {
  PyObject *a_obj = nullptr, *b_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return nullptr;
  Tensor *a = nullptr, *b = nullptr;
  if (!check_tensor(a_obj, &a) || !check_tensor(b_obj, &b)) return nullptr;
  return wrap_result_tensor(autograd::matmul(*a, *b));
}

PyObject* mod_linear(PyObject*, PyObject* args) noexcept {
  PyObject *x_obj = nullptr, *w_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &x_obj, &w_obj)) return nullptr;
  Tensor *x = nullptr, *w = nullptr;
  if (!check_tensor(x_obj, &x) || !check_tensor(w_obj, &w)) return nullptr;
  return wrap_result_tensor(autograd::linear(*x, *w));
}

PyObject* mod_bitnet_linear_ste(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"x", "w_master_f32", nullptr};
  PyObject *x_obj = nullptr, *w_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist),
                                   &x_obj, &w_obj)) {
    return nullptr;
  }
  Tensor *x = nullptr, *w = nullptr;
  if (!check_tensor(x_obj, &x) || !check_tensor(w_obj, &w)) return nullptr;
  return wrap_result_tensor(autograd::bitnet_linear_ste(*x, *w));
}

PyObject* mod_sum(PyObject*, PyObject* args) noexcept {
  PyObject* x_obj = nullptr;
  if (!PyArg_ParseTuple(args, "O", &x_obj)) return nullptr;
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;
  return wrap_result_tensor(autograd::sum(*x));
}

PyObject* mod_mean(PyObject*, PyObject* args) noexcept {
  PyObject* x_obj = nullptr;
  if (!PyArg_ParseTuple(args, "O", &x_obj)) return nullptr;
  Tensor* x = nullptr;
  if (!check_tensor(x_obj, &x)) return nullptr;
  return wrap_result_tensor(autograd::mean(*x));
}

PyObject* mod_mse_loss(PyObject*, PyObject* args) noexcept {
  PyObject *p_obj = nullptr, *t_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &p_obj, &t_obj)) return nullptr;
  Tensor *p = nullptr, *t = nullptr;
  if (!check_tensor(p_obj, &p) || !check_tensor(t_obj, &t)) return nullptr;
  return wrap_result_tensor(autograd::mse_loss(*p, *t));
}

PyObject* mod_cross_entropy_loss(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"logits", "targets", nullptr};
  PyObject *logits_obj = nullptr, *targets_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist),
                                   &logits_obj, &targets_obj)) {
    return nullptr;
  }
  Tensor* logits = nullptr;
  if (!check_tensor(logits_obj, &logits)) return nullptr;
  SmallVec<Index, 128> targets;
  if (!parse_index_vec(targets_obj, &targets)) return nullptr;
  return wrap_result_tensor(
      autograd::cross_entropy_loss(*logits, Span<const Index>(targets.data(), targets.size())));
}

PyObject* mod_clip_grad_norm(PyObject*, PyObject* args, PyObject* kwds) noexcept {
  static const char* kwlist[] = {"params", "max_norm", nullptr};
  PyObject* params_obj = nullptr;
  double max_norm = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Od", const_cast<char**>(kwlist),
                                   &params_obj, &max_norm)) {
    return nullptr;
  }
  SmallVec<Tensor, 32> params;
  if (!parse_tensor_list(params_obj, &params)) return nullptr;
  auto res = optim::clip_grad_norm(Span<Tensor>(params.data(), params.size()),
                                   static_cast<f32>(max_norm));
  if (!res.is_ok()) return set_python_error(res.status());
  return PyFloat_FromDouble(static_cast<double>(res.value()));
}

// =============================================================================
// Module Method Table
// =============================================================================
PyMethodDef Log3ModuleMethods[] = {
    // Runtime & Diagnostics
    {"init_runtime", L3_PY_CFUNC(mod_init_runtime),
     METH_VARARGS | METH_KEYWORDS,
     "init_runtime(min_level: int = 2) -> None\nInitialize log3 runtime logging, CPU SIMD detector, and thread pool."},
    {"set_log_level", mod_set_log_level, METH_VARARGS,
     "set_log_level(level: int) -> None\nSet minimum log level (0=TRACE..5=FATAL, 6=OFF)."},
    {"log_level", mod_log_level, METH_NOARGS,
     "log_level() -> int\nReturn current runtime log level."},
    {"version_string", mod_version_string, METH_NOARGS,
     "version_string() -> str\nReturn semantic version string ('0.4.0')."},
    {"version_hex", mod_version_hex, METH_NOARGS,
     "version_hex() -> int\nReturn packed 24-bit version integer."},
    {"platform_name", mod_platform_name, METH_NOARGS,
     "platform_name() -> str\nReturn target operating system name."},
    {"arch_name", mod_arch_name, METH_NOARGS,
     "arch_name() -> str\nReturn target CPU architecture name."},
    {"cpu_features", mod_cpu_features, METH_NOARGS,
     "cpu_features() -> dict[str, int | bool]\nQuery hardware SIMD capabilities and cache sizes."},
    {"memory_stats", mod_memory_stats, METH_NOARGS,
     "memory_stats() -> dict[str, int]\nReturn live and peak 64-byte aligned memory allocation counters."},
    {"reset_peak_memory", mod_reset_peak_memory, METH_NOARGS,
     "reset_peak_memory() -> None\nReset peak memory watermark to current live bytes."},

    // Tensor Factories
    {"empty", L3_PY_CFUNC(mod_empty), METH_VARARGS | METH_KEYWORDS,
     "empty(shape: Sequence[int], dtype: str = 'f32', requires_grad: bool = False) -> Tensor\nAllocate an uninitialized 64-byte aligned Tensor."},
    {"zeros", L3_PY_CFUNC(mod_zeros), METH_VARARGS | METH_KEYWORDS,
     "zeros(shape: Sequence[int], dtype: str = 'f32', requires_grad: bool = False) -> Tensor\nAllocate a zero-initialized 64-byte aligned Tensor."},
    {"ones", L3_PY_CFUNC(mod_ones), METH_VARARGS | METH_KEYWORDS,
     "ones(shape: Sequence[int], dtype: str = 'f32', requires_grad: bool = False) -> Tensor\nAllocate a 1-initialized 64-byte aligned Tensor."},
    {"full", L3_PY_CFUNC(mod_full), METH_VARARGS | METH_KEYWORDS,
     "full(shape: Sequence[int], fill_value: float, dtype: str = 'f32', requires_grad: bool = False) -> Tensor\nAllocate a Tensor filled with fill_value."},
    {"from_list", L3_PY_CFUNC(mod_from_list), METH_VARARGS | METH_KEYWORDS,
     "from_list(values: Sequence[float], shape: Sequence[int], dtype: str = 'f32', requires_grad: bool = False) -> Tensor\nConstruct a Tensor from a flat Python list/sequence."},
    {"from_buffer", L3_PY_CFUNC(mod_from_buffer), METH_VARARGS | METH_KEYWORDS,
     "from_buffer(buffer: Buffer, shape: Optional[Sequence[int]] = None, dtype: str = 'f32', requires_grad: bool = False) -> Tensor\nConstruct a Tensor from any Python buffer (NumPy array, bytes, memoryview)."},

    // Low-Level SIMD & BitNet Kernels
    {"quantize_f32_to_t2", L3_PY_CFUNC(mod_quantize_f32_to_t2),
     METH_VARARGS | METH_KEYWORDS,
     "quantize_f32_to_t2(weights_f32: Tensor) -> Tensor\nQuantize FP32 weights into 2-bit packed kTernary2 format (4 weights/byte) using absmean scaling."},
    {"dequantize_t2_to_f32", L3_PY_CFUNC(mod_dequantize_t2_to_f32),
     METH_VARARGS | METH_KEYWORDS,
     "dequantize_t2_to_f32(weights_t2: Tensor) -> Tensor\nUnpack 2-bit kTernary2 weights back to FP32 multiplied by quant_scale."},
    {"bitnet_linear_t2_f32", L3_PY_CFUNC(mod_bitnet_linear_t2_f32),
     METH_VARARGS | METH_KEYWORDS,
     "bitnet_linear_t2_f32(x: Tensor, w_t2: Tensor) -> Tensor\nExecute multi-threaded SIMD BitNet b1.58 ternary projection."},
    {"dot_f32", mod_dot_f32, METH_VARARGS,
     "dot_f32(a: Tensor, b: Tensor) -> float\nCompute SIMD dot product of two contiguous f32 tensors."},
    {"add_f32", mod_add_f32, METH_VARARGS,
     "add_f32(a: Tensor, b: Tensor) -> Tensor\nCompute SIMD elementwise addition of two f32 tensors."},
    {"mul_f32", mod_mul_f32, METH_VARARGS,
     "mul_f32(a: Tensor, b: Tensor) -> Tensor\nCompute SIMD elementwise multiplication of two f32 tensors."},
    {"silu_f32", mod_silu_f32, METH_VARARGS,
     "silu_f32(x: Tensor) -> Tensor\nCompute SIMD SiLU activation x * sigmoid(x)."},
    {"rmsnorm_f32", L3_PY_CFUNC(mod_rmsnorm_f32),
     METH_VARARGS | METH_KEYWORDS,
     "rmsnorm_f32(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor\nCompute SIMD Root Mean Square Layer Normalization."},
    {"softmax_f32", mod_softmax_f32, METH_VARARGS,
     "softmax_f32(x: Tensor) -> Tensor\nCompute row-wise numerically stable softmax over the last axis."},
    {"rope_f32_inplace", L3_PY_CFUNC(mod_rope_f32_inplace),
     METH_VARARGS | METH_KEYWORDS,
     "rope_f32_inplace(qk: Tensor, start_pos: int = 0, theta_base: float = 10000.0) -> Tensor\nApply Rotary Position Embedding in-place."},
    {"matmul_f32", mod_matmul_f32, METH_VARARGS,
     "matmul_f32(a: Tensor, b: Tensor) -> Tensor\nCompute multi-threaded SIMD matrix multiplication A(M, K) @ B(K, N)."},

    // Autograd & BitNet STE QAT Operations
    {"is_grad_enabled", mod_is_grad_enabled, METH_NOARGS,
     "is_grad_enabled() -> bool\nReturn True if reverse-mode autograd graph recording is active."},
    {"set_grad_enabled", mod_set_grad_enabled, METH_VARARGS,
     "set_grad_enabled(enabled: bool) -> None\nEnable or disable reverse-mode autograd graph recording."},
    {"add", mod_add, METH_VARARGS,
     "add(a: Tensor, b: Tensor) -> Tensor\nDifferentiable elementwise addition."},
    {"sub", mod_sub, METH_VARARGS,
     "sub(a: Tensor, b: Tensor) -> Tensor\nDifferentiable elementwise subtraction."},
    {"mul", mod_mul, METH_VARARGS,
     "mul(a: Tensor, b: Tensor) -> Tensor\nDifferentiable elementwise Hadamard multiplication."},
    {"silu", mod_silu, METH_VARARGS,
     "silu(x: Tensor) -> Tensor\nDifferentiable SiLU (Swish) activation x * sigmoid(x)."},
    {"relu", mod_relu, METH_VARARGS,
     "relu(x: Tensor) -> Tensor\nDifferentiable ReLU activation max(0, x)."},
    {"rmsnorm", L3_PY_CFUNC(mod_rmsnorm), METH_VARARGS | METH_KEYWORDS,
     "rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor\nDifferentiable Root Mean Square Layer Normalization."},
    {"matmul", mod_matmul, METH_VARARGS,
     "matmul(a: Tensor, b: Tensor) -> Tensor\nDifferentiable 2D matrix multiplication A(M, K) @ B(K, N)."},
    {"linear", mod_linear, METH_VARARGS,
     "linear(x: Tensor, weight: Tensor) -> Tensor\nDifferentiable dense linear projection x(M, K) @ weight(N, K)^T."},
    {"bitnet_linear_ste", L3_PY_CFUNC(mod_bitnet_linear_ste),
     METH_VARARGS | METH_KEYWORDS,
     "bitnet_linear_ste(x: Tensor, w_master_f32: Tensor) -> Tensor\nBitNet b1.58 Straight-Through Estimator (STE) linear projection for Quantization-Aware Training."},
    {"sum", mod_sum, METH_VARARGS,
     "sum(x: Tensor) -> Tensor\nDifferentiable reduction sum to a 0-D scalar Tensor."},
    {"mean", mod_mean, METH_VARARGS,
     "mean(x: Tensor) -> Tensor\nDifferentiable reduction mean to a 0-D scalar Tensor."},
    {"mse_loss", mod_mse_loss, METH_VARARGS,
     "mse_loss(pred: Tensor, target: Tensor) -> Tensor\nDifferentiable Mean Squared Error loss."},
    {"cross_entropy_loss", L3_PY_CFUNC(mod_cross_entropy_loss),
     METH_VARARGS | METH_KEYWORDS,
     "cross_entropy_loss(logits: Tensor, targets: Sequence[int]) -> Tensor\nDifferentiable fused log-softmax Cross-Entropy loss."},
    {"clip_grad_norm", L3_PY_CFUNC(mod_clip_grad_norm),
     METH_VARARGS | METH_KEYWORDS,
     "clip_grad_norm(params: Sequence[Tensor], max_norm: float) -> float\nClip global L2 gradient norm across parameters in-place and return unclipped norm."},
    {nullptr, nullptr, 0, nullptr},
};

struct PyModuleDef Log3ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "log3._C",
    "Native C++17 core extension for log3 (BitNet b1.58, SIMD Kernels, Zero-Copy .l3 I/O, and Reverse-Mode Autograd).",
    -1,
    Log3ModuleMethods,
    nullptr, nullptr, nullptr, nullptr,
};

bool register_type(PyObject* m, const char* name, PyTypeObject* type) noexcept {
  if (PyType_Ready(type) < 0) return false;
  Py_INCREF(type);
  if (PyModule_AddObject(m, name, reinterpret_cast<PyObject*>(type)) < 0) {
    Py_DECREF(type);
    return false;
  }
  return true;
}

}  // namespace

PyMODINIT_FUNC PyInit__C(void) {
  PyObject* m = PyModule_Create(&Log3ModuleDef);
  if (!m) return nullptr;

  if (!register_type(m, "Tensor", &PyTensorType)) return nullptr;
  if (!register_type(m, "Rng", &PyRngType)) return nullptr;
  if (!register_type(m, "NoGradGuard", &PyNoGradGuardType)) return nullptr;
  if (!register_type(m, "RMSNorm", &PyRMSNormType)) return nullptr;
  if (!register_type(m, "BitLinear", &PyBitLinearType)) return nullptr;
  if (!register_type(m, "Linear", &PyLinearType)) return nullptr;
  if (!register_type(m, "Embedding", &PyEmbeddingType)) return nullptr;
  if (!register_type(m, "RotaryEmbedding", &PyRotaryEmbeddingType)) return nullptr;
  if (!register_type(m, "KVCache", &PyKVCacheType)) return nullptr;
  if (!register_type(m, "GroupedQueryAttention", &PyGQAType)) return nullptr;
  if (!register_type(m, "BitNetFeedForward", &PyBitNetFFNType)) return nullptr;
  if (!register_type(m, "TransformerBlock", &PyTransformerBlockType)) return nullptr;
  if (!register_type(m, "BitNetTransformer", &PyBitNetTransformerType)) return nullptr;
  if (!register_type(m, "ModelWriter", &PyModelWriterType)) return nullptr;
  if (!register_type(m, "ModelReader", &PyModelReaderType)) return nullptr;
  if (!register_type(m, "SGD", &PySGDType)) return nullptr;
  if (!register_type(m, "AdamW", &PyAdamWType)) return nullptr;

  // Export LogLevel constants
  PyModule_AddIntConstant(m, "LOG_TRACE", static_cast<long>(l3::LogLevel::kTrace));
  PyModule_AddIntConstant(m, "LOG_DEBUG", static_cast<long>(l3::LogLevel::kDebug));
  PyModule_AddIntConstant(m, "LOG_INFO", static_cast<long>(l3::LogLevel::kInfo));
  PyModule_AddIntConstant(m, "LOG_WARN", static_cast<long>(l3::LogLevel::kWarn));
  PyModule_AddIntConstant(m, "LOG_ERROR", static_cast<long>(l3::LogLevel::kError));
  PyModule_AddIntConstant(m, "LOG_FATAL", static_cast<long>(l3::LogLevel::kFatal));
  PyModule_AddIntConstant(m, "LOG_OFF", static_cast<long>(l3::LogLevel::kOff));

  return m;
}
