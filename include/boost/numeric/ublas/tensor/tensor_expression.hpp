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
//

#ifndef BOOST_UBLAS_TENSOR_YAP_EXPRESSIONS_HPP
#define BOOST_UBLAS_TENSOR_YAP_EXPRESSIONS_HPP

#include "expression_transforms.hpp"
#include "extents.hpp"
#include "strides.hpp"
#include <boost/config.hpp>
#include <boost/yap/print.hpp>
#include <boost/yap/yap.hpp>

namespace boost::numeric::ublas::detail {

/**
 * @brief A YAP expression for tensor type.
 *
 * @tparam Kind the Kind of Operation represented
 *
 * @tparam Tuple the operands to which operation is performed.
 */
template <boost::yap::expr_kind Kind, typename Tuple> struct tensor_expression {
  const static boost::yap::expr_kind kind = Kind;

  Tuple elements;
  /**
   * @brief Evaluates the tensor_expression lazily. one index at a time.
   *
   * @param i the Index to evaluate
   *
   * @return the Evaluated Value of the expression at ith index.
   */
  BOOST_UBLAS_INLINE decltype(auto) operator()(size_t i) {
    auto nth = boost::yap::transform(*this, transforms::at_index{i});
    return boost::yap::evaluate(nth);
  }
  //  todo (coder3101) : Make eval() and eval_to() based on device.

  /**
   * @brief Completely evaluated this expression and returns a tensor.
   *
   * @tparam T the data type to use for resulting tensor.
   *
   * @tparam F the storage format to use for resulting tensor.
   *
   * @tparam A the array type to use for resulting tensor.
   *
   * @return The tensor which contains the values of evaluated expresssion.
   */
  template <class T, class F = ::boost::numeric::ublas::first_order,
            class A = std::vector<T, std::allocator<T>>>
  auto eval() const {
    ::boost::numeric::ublas::tensor<T, F, A> result;
    auto shape_expr = boost::yap::transform(*this, transforms::get_extents{});
    result.extents_ = shape_expr;
    result.strides_ = basic_strides<std::size_t, F>{shape_expr};
    result.data_.resize(shape_expr.product());
#pragma omp parallel for
    for (auto i = 0u; i < shape_expr.product(); i++)
      result.data_[i] = this->operator()(i);
    return result;
  }
  /**
   * @brief Completely evaluates this expression and fills values into target.
   *
   * @tparam T the data-type of target tensor (deduced)
   *
   * @tparam F the format-type of target tensor (deduced)
   *
   * @tparam A the  array type of target tensor (deduced)
   *
   * @param[out] target the resulting tensor.
   */

  template <class T, class F, class A>
  void eval_to(::boost::numeric::ublas::tensor<T, F, A> &target) {
    auto shape_expr = boost::yap::transform(*this, transforms::get_extents{});
    target.data_.resize(shape_expr.product());
    target.extents_ = shape_expr;
    target.strides_ = basic_strides<std::size_t, F>{shape_expr};

    for (auto i = 0u; i < shape_expr.product(); i++)
      target.data_[i] = this->operator()(i);
  }

  /**
   * @brief Implictly converts this tensor_expression to a bool type.
   *
   * @note This conversion throws a runtime_error if the expression does not
   * contain any logical operator.
   */
  operator bool() {
    boost::numeric::ublas::detail::transforms::expr_has_logical_operator e;
    boost::yap::transform(*this, e);
    if (!e.status) {
      throw std::runtime_error(
          "Cannot convert the tensor expression to bool type. Only expression "
          "with at least one-logical operator are convertible");
    } else {
      auto shape_expr = boost::yap::transform(*this, transforms::get_extents{});
      for (auto i = 0u; i < shape_expr.product(); i++)
        if (!this->operator()(i))
          return false;
    }
    return true;
  }
};
} // namespace boost::numeric::ublas::detail

#endif
