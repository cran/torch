#pragma once
#define LANTERN_HEADERS_ONLY
#include <string>
#include <memory>

#include "lantern/lantern.h"

#include <RcppCommon.h>

class XPtrTorch
{
private:
  std::shared_ptr<void> ptr;
public:
  XPtrTorch(void * x, std::function<void(void*)> deleter = [](void*){}) :
    XPtrTorch(std::shared_ptr<void>(x, deleter)) {};
  explicit XPtrTorch(std::shared_ptr<void> x) : ptr(x) {}
  void* get() const
  {
    return ptr.get();
  }
  std::shared_ptr<void> get_shared() const {
    return ptr;
  }
};

class XPtrTorchIndexTensor: public XPtrTorch {
public:
  XPtrTorchIndexTensor (): XPtrTorch{NULL} {}
  XPtrTorchIndexTensor (void* x) : XPtrTorch(x, lantern_Tensor_delete) {}
  explicit XPtrTorchIndexTensor (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchIndexTensor (const XPtrTorchIndexTensor& x): XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchIndexTensor (SEXP x);
  operator SEXP () const;
};

std::function<void(void*)> tensor_deleter (void* x);

class XPtrTorchTensor : public XPtrTorch {
public:
  // TODO: we should make this explicit at some point, but not currently
  // possible because we rely on it in too many places.
  XPtrTorchTensor () : XPtrTorch{NULL} {}
  XPtrTorchTensor (void* x) : XPtrTorch(x, lantern_Tensor_delete) {}
  explicit XPtrTorchTensor (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchTensor (const XPtrTorchTensor& x): XPtrTorch(x.get_shared()) {}
  XPtrTorchTensor (XPtrTorchIndexTensor x): XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchTensor (SEXP x);
  operator SEXP () const;
};

class XPtrTorchScriptModule : public XPtrTorch {
public:
  // TODO rename lantern_JITModule_delete 
  XPtrTorchScriptModule (void* x) : XPtrTorch(x, lantern_JITModule_delete) {}
  explicit XPtrTorchScriptModule (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchScriptModule (const XPtrTorchScriptModule& x): XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchScriptModule (SEXP x);
  operator SEXP () const;
};

class XPtrTorchScriptMethod : public XPtrTorch {
public:
  XPtrTorchScriptMethod (void* x) : XPtrTorch(x, lantern_jit_ScriptMethod_delete) {}
  XPtrTorchScriptMethod (void* x, std::function<void(void* x)> deleter) : XPtrTorch(x, deleter) {}
  explicit XPtrTorchScriptMethod (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchScriptMethod (const XPtrTorchScriptMethod& x): XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchScriptMethod (SEXP x);
  operator SEXP () const;
};

class XPtrTorchOptionalTensor : public XPtrTorch {
public:
  XPtrTorchOptionalTensor (void* x) : XPtrTorch(x, lantern_optional_tensor_delete) {}
  explicit XPtrTorchOptionalTensor (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchOptionalTensor (const XPtrTorchOptionalTensor& x): XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchOptionalTensor (SEXP x);
  operator SEXP () const;
};

class XPtrTorchTensorList : public XPtrTorch {
public:
  XPtrTorchTensorList (void* x) : XPtrTorch(x, lantern_TensorList_delete) {}
  explicit XPtrTorchTensorList (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchTensorList (const XPtrTorchTensorList& x) : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchTensorList (SEXP x);
  operator SEXP () const;
};

class XPtrTorchOptionalTensorList : public XPtrTorch {
public:
  XPtrTorchOptionalTensorList (void* x) : XPtrTorch(x, lantern_OptionalTensorList_delete) {}
  explicit XPtrTorchOptionalTensorList (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchOptionalTensorList (const XPtrTorchTensorList& x) : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchOptionalTensorList (SEXP x);
  operator SEXP () const;
};

class XPtrTorchIndexTensorList: public XPtrTorch {
public:
  XPtrTorchIndexTensorList (void* x) : XPtrTorch(x, lantern_TensorList_delete) {}
  explicit XPtrTorchIndexTensorList (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchIndexTensorList (const XPtrTorchIndexTensorList& x) : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchIndexTensorList (SEXP x);
};

class XPtrTorchOptionalIndexTensorList: public XPtrTorch {
public:
  XPtrTorchOptionalIndexTensorList (void* x) : XPtrTorch(x, lantern_OptionalTensorList_delete) {}
  explicit XPtrTorchOptionalIndexTensorList (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchOptionalIndexTensorList (const XPtrTorchOptionalIndexTensorList& x) : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchOptionalIndexTensorList (SEXP x);
};

class XPtrTorchScalarType : public XPtrTorch {
public:
  XPtrTorchScalarType (void* x) : XPtrTorch (x, lantern_ScalarType_delete) {}
  explicit XPtrTorchScalarType (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchScalarType (const XPtrTorchScalarType& x) : XPtrTorch(x.get_shared()) {}
  operator SEXP () const;
};

class XPtrTorchScalar : public XPtrTorch {
public:
  XPtrTorchScalar () : XPtrTorch{NULL} {}
  XPtrTorchScalar (void* x) : XPtrTorch(x, lantern_Scalar_delete) {}
  explicit XPtrTorchScalar (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchScalar (const XPtrTorchScalar& x) : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchScalar (SEXP x);
  operator SEXP () const;
};

class XPtrTorchTensorOptions : public XPtrTorch {
public:
  XPtrTorchTensorOptions (void* x) : XPtrTorch(x, lantern_TensorOptions_delete) {};
  explicit XPtrTorchTensorOptions (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchTensorOptions (const XPtrTorchTensorOptions& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchTensorOptions (SEXP x);
  operator SEXP () const;
};

class XPtrTorchDevice : public XPtrTorch {
public:
  XPtrTorchDevice (void* x) : XPtrTorch(x, lantern_Device_delete) {}
  explicit XPtrTorchDevice (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchDevice (const XPtrTorchDevice& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchDevice (SEXP x);
  operator SEXP () const;
};

class XPtrTorchOptionalDevice : public XPtrTorch {
public:
  XPtrTorchOptionalDevice (void* x) : XPtrTorch(x, lantern_optional_device_delete) {}
  explicit XPtrTorchOptionalDevice (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchOptionalDevice (const XPtrTorchOptionalDevice& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchOptionalDevice (SEXP x);
  operator SEXP () const;
};

class XPtrTorchDtype : public XPtrTorch {
public:
  XPtrTorchDtype () : XPtrTorch{NULL} {}
  XPtrTorchDtype (void* x) : XPtrTorch(x, lantern_Dtype_delete) {}
  explicit XPtrTorchDtype (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchDtype (const XPtrTorchDtype& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchDtype (SEXP x);
  operator SEXP () const;
};

class XPtrTorchDimname : public XPtrTorch {
public:
  XPtrTorchDimname (void* x) : XPtrTorch(x, lantern_Dimname_delete) {}
  explicit XPtrTorchDimname (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchDimname (const XPtrTorchDimname& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchDimname (SEXP x);
  explicit XPtrTorchDimname (const std::string& x) : 
    XPtrTorch(lantern_Dimname(x.c_str()), lantern_Dimname_delete) {};
  operator SEXP () const;
};

class XPtrTorchDimnameList : public XPtrTorch {
public:
  XPtrTorchDimnameList (void* x) : XPtrTorch(x, lantern_DimnameList_delete) {}
  explicit XPtrTorchDimnameList (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchDimnameList (const XPtrTorchDimnameList& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchDimnameList (SEXP x);
  operator SEXP () const;
};

class XPtrTorchjit_named_parameter_list : public XPtrTorch {
public:
  XPtrTorchjit_named_parameter_list (void* x) : XPtrTorch(x, lantern_jit_named_parameter_list_delete) {}
  explicit XPtrTorchjit_named_parameter_list (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchjit_named_parameter_list (const XPtrTorchjit_named_parameter_list& x) : XPtrTorch(x.get_shared()) {};
  operator SEXP () const;
};

class XPtrTorchjit_named_buffer_list : public XPtrTorch {
public:
  XPtrTorchjit_named_buffer_list (void* x) : XPtrTorch(x, lantern_jit_named_buffer_list_delete) {}
  explicit XPtrTorchjit_named_buffer_list (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchjit_named_buffer_list (const XPtrTorchjit_named_buffer_list& x) : XPtrTorch(x.get_shared()) {};
  operator SEXP () const;
};

class XPtrTorchjit_named_module_list : public XPtrTorch {
public:
  XPtrTorchjit_named_module_list (void* x) : XPtrTorch(x, lantern_jit_named_module_list_delete) {}
  explicit XPtrTorchjit_named_module_list (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchjit_named_module_list (const XPtrTorchjit_named_module_list& x) : XPtrTorch(x.get_shared()) {};
  operator SEXP () const;
};

class XPtrTorchGenerator : public XPtrTorch {
public:
  XPtrTorchGenerator (void* x) : XPtrTorch(x, lantern_Generator_delete) {}
  explicit XPtrTorchGenerator (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchGenerator (const XPtrTorchGenerator& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchGenerator (SEXP x);
  operator SEXP () const;
};

class XPtrTorchMemoryFormat : public XPtrTorch {
public:
  XPtrTorchMemoryFormat () : XPtrTorch{NULL} {};
  XPtrTorchMemoryFormat (void* x) : XPtrTorch(x, lantern_MemoryFormat_delete) {}
  explicit XPtrTorchMemoryFormat (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchMemoryFormat (const XPtrTorchMemoryFormat& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchMemoryFormat (SEXP x);
  operator SEXP () const;
};

class XPtrTorchIntArrayRef : public XPtrTorch {
public:
  XPtrTorchIntArrayRef () : XPtrTorch{NULL} {};
  XPtrTorchIntArrayRef (void* x) : XPtrTorch(x, lantern_vector_int64_t_delete) {}
  explicit XPtrTorchIntArrayRef (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchIntArrayRef (const XPtrTorchIntArrayRef& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchIntArrayRef (SEXP x);
  //operator SEXP () const;
};

class XPtrTorchFunctionPtr : public XPtrTorch {
public:
  XPtrTorchFunctionPtr (void* x) : XPtrTorch(x, lantern_FunctionPtr_delete) {}
  XPtrTorchFunctionPtr (void* x, std::function<void(void*)> deleter) : XPtrTorch(x, deleter) {}
  explicit XPtrTorchFunctionPtr (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchFunctionPtr (const XPtrTorchFunctionPtr& x) : XPtrTorch(x.get_shared()) {};
};

class XPtrTorchIndexIntArrayRef : public XPtrTorch {
public:
  XPtrTorchIndexIntArrayRef () : XPtrTorch{NULL} {};
  XPtrTorchIndexIntArrayRef (void* x) : XPtrTorch(x, lantern_vector_int64_t_delete) {}
  explicit XPtrTorchIndexIntArrayRef (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchIndexIntArrayRef (const XPtrTorchIndexIntArrayRef& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchIndexIntArrayRef (SEXP x);
};

class XPtrTorchOptionalIntArrayRef {
public:
  std::shared_ptr<void> ptr;
  std::vector<int64_t> data;
  bool is_null;
  
  XPtrTorchOptionalIntArrayRef () {};
  explicit XPtrTorchOptionalIntArrayRef (SEXP x);
  
  XPtrTorchOptionalIntArrayRef (std::vector<int64_t> data_, bool is_null_) {
    data = data_;
    ptr = std::shared_ptr<void>(
      lantern_optional_vector_int64_t(data.data(), data.size(), is_null_),
      lantern_optional_vector_int64_t_delete
    );
    is_null = is_null_;
  }
  
  XPtrTorchOptionalIntArrayRef (const XPtrTorchOptionalIntArrayRef& x ) : 
    XPtrTorchOptionalIntArrayRef (x.data, x.is_null) {}
  
  void* get() const
  {
    return ptr.get();
  }
};  

class XPtrTorchOptionalIndexIntArrayRef : public XPtrTorchOptionalIntArrayRef {
public:
  explicit XPtrTorchOptionalIndexIntArrayRef (SEXP x);
};

class XPtrTorchint64_t2 {
public:
  std::shared_ptr<void> ptr;
  explicit XPtrTorchint64_t2 (SEXP x_);
  void* get () {
    return ptr.get();
  }
};

class XPtrTorchoptional_int64_t2 {
public:
  std::shared_ptr<void> ptr;
  explicit XPtrTorchoptional_int64_t2 (SEXP x_);
  void* get () {
    return ptr.get();
  }
};

class XPtrTorchindex_int64_t {
public:
  std::shared_ptr<void> ptr;
  explicit XPtrTorchindex_int64_t (SEXP x_);
  void* get () {
    return ptr.get();
  }
};

class XPtrTorchoptional_index_int64_t {
public:
  std::shared_ptr<void> ptr;
  explicit XPtrTorchoptional_index_int64_t (SEXP x_);
  void* get () {
    return ptr.get();
  }
};

class XPtrTorchvector_string : public XPtrTorch {
public:
  XPtrTorchvector_string (void * x) : XPtrTorch(x, lantern_vector_string_delete) {}
  operator SEXP () const;
};

class XPtrTorchstring : public XPtrTorch {
public:
  XPtrTorchstring (void * x) : XPtrTorch(x, lantern_string_delete) {}
  XPtrTorchstring (SEXP x);
  XPtrTorchstring (const XPtrTorchstring& x) : XPtrTorch(x.get_shared()) {};
  XPtrTorchstring (std::string x) : XPtrTorchstring(lantern_string_new(x.c_str())) {};
  operator SEXP () const;
};

class XPtrTorchStack : public XPtrTorch {
public:
  XPtrTorchStack (void * x) : XPtrTorch(x, lantern_Stack_delete) {}
  XPtrTorchStack (void * x, std::function<void(void*)> deleter) : XPtrTorch(x, deleter) {}
  explicit XPtrTorchStack (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchStack (SEXP x);
  operator SEXP () const;
};

class XPtrTorchIValue : public XPtrTorch {
public:
  XPtrTorchIValue (void * x) : XPtrTorch (x, lantern_IValue_delete) {}
  explicit XPtrTorchIValue (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchIValue (const XPtrTorchIValue& x) : XPtrTorch(x.get_shared()) {};
  XPtrTorchIValue (SEXP x);
  operator SEXP () const;
};

class XPtrTorchTuple : public XPtrTorch {
public:
  XPtrTorchTuple (void * x) : XPtrTorch (x, lantern_jit_Tuple_delete) {}
  XPtrTorchTuple (const XPtrTorchTuple& x) : XPtrTorch(x.get_shared()) {};
  XPtrTorchTuple (SEXP x);
  operator SEXP () const;
};

class XPtrTorchvector_bool : public XPtrTorch {
public:
  XPtrTorchvector_bool (void * x) : XPtrTorch(x, lantern_vector_bool_delete) {}
  operator SEXP() const;
  XPtrTorchvector_bool (SEXP x);
};

class XPtrTorchvector_Scalar : public XPtrTorch {
public:
  XPtrTorchvector_Scalar (void * x) : XPtrTorch(x, lantern_vector_Scalar_delete) {}
  operator SEXP() const;
  XPtrTorchvector_Scalar (SEXP x);
  XPtrTorchvector_Scalar (const XPtrTorchvector_Scalar& x) : XPtrTorch(x.get_shared()) {};
};

class XPtrTorchvector_int64_t : public XPtrTorch {
public:
  XPtrTorchvector_int64_t (void* x) : XPtrTorch(x, lantern_vector_int64_t2_delete) {}
  operator SEXP() const;
  XPtrTorchvector_int64_t (SEXP x);
};

class XPtrTorchvector_double : public XPtrTorch {
public:
  XPtrTorchvector_double (void* x) : XPtrTorch(x, lantern_vector_double_delete) {}
  operator SEXP() const;
  XPtrTorchvector_double (SEXP x);
};

class XPtrTorchTensorDict : public XPtrTorch {
public:
  XPtrTorchTensorDict (void* x) : XPtrTorch(x, lantern_jit_TensorDict_delete) {}
  XPtrTorchTensorDict (SEXP x);
};

class XPtrTorchGenericDict : public XPtrTorch {
public:
  XPtrTorchGenericDict (void* x) : XPtrTorch(x, lantern_jit_GenericDict_delete) {}
  operator SEXP() const;
};

class XPtrTorchGenericList : public XPtrTorch {
public:
  XPtrTorchGenericList (void* x) : XPtrTorch(x, lantern_jit_GenericList_delete) {}
  operator SEXP() const;
};

class XPtrTorchvector_IValue : public XPtrTorch {
public:
  XPtrTorchvector_IValue (void* x) : XPtrTorch(x, lantern_jit_vector_IValue_delete) {}
  operator SEXP() const;
};

class XPtrTorchNamedTupleHelper : public XPtrTorch {
public:
  XPtrTorchNamedTupleHelper (void* x) : XPtrTorch(x, lantern_NamedTupleHelper_delete) {}
  XPtrTorchNamedTupleHelper (SEXP x);
  operator SEXP() const;
};

class XPtrTorchCompilationUnit : public XPtrTorch {
public:
  XPtrTorchCompilationUnit (void * x) : XPtrTorch(x, lantern_CompilationUnit_delete) {}
  XPtrTorchCompilationUnit (SEXP x);
  explicit XPtrTorchCompilationUnit (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchCompilationUnit (const XPtrTorchCompilationUnit& x) : XPtrTorch(x.get_shared()) {};
  operator SEXP () const;
};

#include <Rcpp.h>

class XPtrTorchQScheme : public XPtrTorch {
public:
  XPtrTorchQScheme (void* x) : XPtrTorch (x, lantern_QScheme_delete) {}
};

class XPtrTorchdouble : public XPtrTorch {
public:
  XPtrTorchdouble (void* x) : XPtrTorch(x, lantern_double_delete) {}
};

class XPtrTorchvariable_list : public XPtrTorch {
public:
  XPtrTorchvariable_list (void* x) : XPtrTorch(x, lantern_variable_list_delete) {}
};

class XPtrTorchint64_t : public XPtrTorch {
public:
  XPtrTorchint64_t (void* x) : XPtrTorch(x, lantern_int64_t_delete) {}
};

class XPtrTorchbool : public XPtrTorch {
public:
  XPtrTorchbool (void* x) : XPtrTorch(x, lantern_bool_delete) {}
};


class XPtrTorchLayout : public XPtrTorch {
public:
  XPtrTorchLayout (void* x) : XPtrTorch(x, lantern_Layout_delete) {}
};

class XPtrTorchTensorIndex : public XPtrTorch {
public:
  XPtrTorchTensorIndex (void* x) : XPtrTorch(x, lantern_TensorIndex_delete) {}
};

class XPtrTorchoptional_int64_t : public XPtrTorch {
public:
  XPtrTorchoptional_int64_t (void* x) : XPtrTorch(x, lantern_optional_int64_t_delete) {}
};

class XPtrTorchSlice : public XPtrTorch {
public:
  XPtrTorchSlice (void* x) : XPtrTorch(x, lantern_Slice_delete) {}
};

class XPtrTorchPackedSequence : public XPtrTorch {
public:
  XPtrTorchPackedSequence (void * x) : XPtrTorch(x, lantern_PackedSequence_delete) {}
};

class XPtrTorchStorage : public XPtrTorch {
public:
  XPtrTorchStorage (void * x) : XPtrTorch(x, lantern_Storage_delete) {}
};

class XPtrTorchJITModule : public XPtrTorch {
public:
  XPtrTorchJITModule (void * x) : XPtrTorch(x, lantern_JITModule_delete) {}
};

class XPtrTorchTraceableFunction : public XPtrTorch {
public:
  XPtrTorchTraceableFunction (void * x) : XPtrTorch(x, lantern_TraceableFunction_delete) {}
};

class XPtrTorchvector_void : public XPtrTorch {
public:
  XPtrTorchvector_void (void * x) : XPtrTorch(x, lantern_vector_void_delete) {}
};

template<class T>
class nullable {
public:
  T x;
  bool is_null = false;
  nullable (Rcpp::Nullable<T> x) {
    if (x.isNotNull()) {
      this->x = Rcpp::as<T>(x);
    } else {
      this->is_null = true;
    }
  };
  void* get () {
    if (this->is_null)
      return nullptr;
    else
      return &(this->x);
  }
  T get_value () {
    if (this->is_null)
      return NULL;
    else
      return this->x;
  }
};

template<class T>
class nullableVector {
public:
  T x = {0};
  bool is_null = false;
  nullableVector (Rcpp::Nullable<T> x) {
    if (x.isNotNull()) {
      this->x = Rcpp::as<T>(x);
    } else {
      this->is_null = true;
    }
  };
  void* get () {
    if (this->is_null)
      return nullptr;
    else
      return &(this->x);
  }
  T get_value () {
    if (this->is_null)
      return NULL;
    else
      return this->x;
  }
};
