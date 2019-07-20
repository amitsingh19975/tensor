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
};
} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas {
/**
 * @brief Applies a lambda lazily on an expression.
 *
 * @tparam Expr The type of Expression (deduced)
 *
 * @tparam Callable The type of lambda (deduced)
 *
 * @param expr the expression to which the lambda is applied
 *
 * @param c the Generic Lambda to apply
 *
 * @return the new expression denoting the callable
 *
 * @note You must provide a generic lambda that takes only one argument by
 * const-reference and returns non-void type.
 */
template <class Expr, typename Callable>
decltype(auto) for_each(Expr &&e, Callable c) {

  auto expr =
      boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(
          std::forward<Expr>(e));

//  auto temp1 = boost::yap::transform(expr, boost::numeric::ublas::detail::transforms::at_index{0});
//  auto arg = boost::yap::evaluate(boost::yap::transform(
//      temp1, boost::numeric::ublas::detail::transforms::make_dummy_type_expression{}));
  auto arg = expr(0);

  using arg_t = decltype(arg) const &;
  using ret_t = decltype(c(arg));

  using signature = ret_t(arg_t);

  static_assert(!std::is_same_v<void, ret_t>,
                "Callable must return non-void type");

  static_assert(
      std::is_convertible_v<Callable, std::function<signature>>,
      "Invalid signature for the callable, expression value_type cannot be "
      "converted to callable's formal parameter. You can make Callable a generic "
      "lambda that takes only one argument by const-reference");

  ret_t (*func)(arg_t) = c;

  return boost::yap::make_expression<detail::tensor_expression,
                                     boost::yap::expr_kind::call>(
      boost::yap::make_terminal<detail::tensor_expression>(func),
      std::forward<decltype(expr)>(expr));
}

template <class Expr, typename Callable>
decltype(auto) for_each2(Expr &&e, Callable c) {

//  auto expr =
//      boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(
//          std::forward<Expr>(e));

//  auto temp1 = boost::yap::transform(expr, boost::numeric::ublas::detail::transforms::at_index{0});
//  auto arg = boost::yap::evaluate(boost::yap::transform(
//      temp1, boost::numeric::ublas::detail::transforms::make_dummy_type_expression{}));
 auto arg = e(0);

//  using arg_t = decltype(arg) const &;
//  using ret_t = decltype(c(arg));
//
//  using signature = ret_t(arg_t);
//
//  static_assert(!std::is_same_v<void, ret_t>,
//                "Callable must return non-void type");
//
//  static_assert(
//      std::is_convertible_v<Callable, std::function<signature>>,
//      "Invalid signature for the callable, expression value_type cannot be "
//      "converted to callable's formal parameter. You can make Callable a generic "
//      "lambda that takes only one argument by const-reference");
//
//  ret_t (*func)(arg_t) = c;

  return e;
}

} // namespace boost::numeric::ublas

#endif
