//  Copyright (c) 2018-2019
//  Mohammad Ashar Khan
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.

#ifndef BOOST_UBLAS_TENSOR_CAST_MACRO
#define BOOST_UBLAS_TENSOR_CAST_MACRO

#include <boost/yap/yap.hpp>
#include <type_traits>

// todo(@coder3101): Casting always return a new tensor with std::vector<T> as
// internal container. Make this return the same container as passed in
// Tensor::array_type.

// todo(@coder3101): Also Lazy cast that takes tensor_expression

#define BOOST_UBLAS_EAGER_TENSOR_CAST(func_name, cast_name)                    \
  template <class new_type, typename Tensor> auto func_name(Tensor &e) {       \
    if constexpr (std::is_same<new_type, typename Tensor::value_type>::value)  \
      return e;                                                                \
    tensor<new_type, typename Tensor::layout_type,                             \
           std::vector<new_type, std::allocator<new_type>>>                    \
        result;                                                                \
    result.data_.resize(e.extents().product());                                \
    result.strides_ = e.strides();                                             \
    result.extents_ = e.extents();                                             \
    for (auto i = 0u; i < e.extents().product(); i++)                          \
      result.data_[i] = cast_name<new_type>(e.data_[i]);                       \
    return result;                                                             \
  }                                                                            \
  template <class new_type, typename Tensor> auto func_name(Tensor &&e) {      \
    if constexpr (std::is_same<new_type, typename Tensor::value_type>::value)  \
      return e;                                                                \
    using namespace boost::hana::literals;                                     \
    tensor<new_type, typename Tensor::layout_type,                             \
           std::vector<new_type, std::allocator<new_type>>>                    \
        result;                                                                \
    result.data_.resize(e.extents().product());                                \
    result.strides_ = std::move(e.strides());                                  \
    result.extents_ = std::move(e.extents());                                  \
    for (auto i = 0u; i < result.extents_.product(); i++)                      \
      result.data_[i] = cast_name<new_type>(e.data_[i]);                       \
    return result;                                                             \
  }

#endif
