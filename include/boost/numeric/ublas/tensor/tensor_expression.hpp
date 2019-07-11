//  Copyright (c) 2019-2020
//  Mohammad Ashar Khan, ashar786khan@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google in producing this work
//  which started as a Google Summer of Code project.

#ifndef BOOST_UBLAS_TENSOR_YAP_EXPRESSIONS_HPP
#define BOOST_UBLAS_TENSOR_YAP_EXPRESSIONS_HPP

#include "expression_transforms.hpp"
#include "extents.hpp"
#include "lambda_traits.hpp"
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
  const static ::boost::yap::expr_kind kind = Kind;

  Tuple elements;
  /**
   * @brief Evaluates the tensor_expression lazily. one index at a time.
   *
   * @param i the Index to evaluate
   *
   * @return the Evaluated Value of the expression at ith index.
   */
  BOOST_UBLAS_INLINE decltype(auto) operator()(size_t i) {
    auto nth = ::boost::yap::transform(*this, transforms::at_index{i});
#ifndef BOOST_UBLAS_NO_EXPRESSION_OPTIMIZATION
    auto optimized =
        ::boost::yap::transform(nth, transforms::apply_distributive_law{});
    return ::boost::yap::evaluate(optimized);
#else
    return ::boost::yap::evaluate(nth);
#endif
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
  auto eval() {
    ::boost::numeric::ublas::tensor<T, F, A> result;
    auto shape_expr = ::boost::yap::transform(*this, transforms::get_extents{});
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
    auto shape_expr = ::boost::yap::transform(*this, transforms::get_extents{});
    target.data_.resize(shape_expr.product());
    target.extents_ = shape_expr;
    target.strides_ = basic_strides<std::size_t, F>{shape_expr};
#pragma omp parallel for
    for (auto i = 0u; i < shape_expr.product(); i++)
      target.data_[i] = this->operator()(i);
  }

  /**
   * @brief Implicitly converts this tensor_expression to a bool type.
   *
   * @note This conversion throws a runtime_error if the expression does not
   * contain any relational operator. It returns true if tensor on both side are
   * empty.
   */
  operator bool() { // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
    auto meta_transform = transforms::expr_count_relational_operator{};

    std::size_t count = ::boost::yap::transform(*this, meta_transform);
    if (count != 1)
      throw std::runtime_error(
          "A tensor expression is only convertible to "
          "bool if it has exactly one relational operator.");

    if (meta_transform.equal_to_found || meta_transform.not_equal_to_found) {
      auto e = transforms::is_equality_or_non_equality_extent_same{};
      ::boost::yap::transform(*this, e);
      if (!e.status && meta_transform.equal_to_found)
        return false;
      if (!e.status && meta_transform.not_equal_to_found)
        return true;
    }

    auto shape_expr = ::boost::yap::transform(*this, transforms::get_extents{});
    for (auto i = 0u; i < shape_expr.product(); i++)
      if (!(this->operator()(i)))
        return false;
    return true;
  }

  /**
   * @brief This is a transform that applies a lambda to the expression and
   * returns a new expression.
   *
   * @tparam Callable The lambda type (deduced)
   *
   * @param c the lambda object
   *
   * @return the new expression
   *
   * @note This takes only lambdas. Please do not pass a functor.
   *
   * @warning Due to some compile-time checkups the lambda cannot be generic or
   * template.
   */
  template <class Callable> decltype(auto) transform(Callable c) {
    using traits = boost::numeric::ublas::detail::function_traits<Callable>;
    static_assert(
        traits::arity == 1,
        "The lambda passed to transform must take only one parameter.");
    static_assert(
        !std::is_same_v<typename traits::result_type, void>,
        "The lambda passed to transform must not return void");
    return boost::yap::make_expression<tensor_expression,
                                       boost::yap::expr_kind::call>(
        boost::yap::make_terminal<tensor_expression>(c), *this);
  }
};
} // namespace boost::numeric::ublas::detail

#endif
