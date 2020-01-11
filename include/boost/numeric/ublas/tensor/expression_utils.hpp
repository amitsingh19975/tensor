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

#ifndef BOOST_UBLAS_EXPRESSION_UTILS_HPP
#define BOOST_UBLAS_EXPRESSION_UTILS_HPP

#include <boost/numeric/ublas/detail/config.hpp>

#include "tensor_expression.hpp"
#include <boost/yap/user_macros.hpp>
#include <boost/yap/yap.hpp>

namespace boost::numeric::ublas::detail {

/**
 * @brief A False type trait for a Tensor-Expression which is non-terminal type
 *
 * @tparam T The Type to check
 */
template <class T> struct is_terminal_type_expr {
  constexpr static bool value = false;
};

/**
 * @brief The True type trait for a Tensor-Expression which is a terminal type
 *
 * @tparam T The type to check
 *
 * @note a type alias `value_type` holds the value wrapped in the terminal node.
 */
template <class T>
struct is_terminal_type_expr<tensor_expression<
    ::boost::yap::expr_kind::terminal, ::boost::hana::tuple<T>>> {
  constexpr static bool value = true;
  using value_type = typename std::remove_reference_t<T>;
};

/**
 * @brief The True type trait for a YAP Expression which is a terminal type
 *
 * @tparam T The type to check
 *
 * @note a type alias `value_type` holds the value wrapped in the terminal node.
 */
template <class T>
struct is_terminal_type_expr<::boost::yap::expression<
    ::boost::yap::expr_kind::terminal, ::boost::hana::tuple<T>>> {
  constexpr static bool value = true;
  using value_type = typename std::remove_reference_t<T>;
};

/**
 * @brief A False type to check the return type of callable passed to
 * `ublas::apply`.
 *
 * @tparam T The Callable type
 */
template <class T> struct function_return;

/**
 * @brief A True type to check the return type of the callable passed to
 * `ublas::apply`.
 *
 * @tparam R The return type (deduced)
 *
 * @tparam A The Argument type (deduced)
 *
 * @note Holds a type alias `type` which is the return type of the callable.
 */
template <class R, class A> struct function_return<R (*)(A)> {
  using type = R;
};

/**
 * @brief Returns a default constructed value of type that the given expression
 * will eval into considering type-promotions (if any) by the language
 * standards.
 *
 * @tparam Expr The Tensor/YAP-Expression type whose evaluated type is expected.
 * (deduced)
 *
 * @param e Expression / Tensor
 *
 * @return Default constructed value of type T, where T is the type of evaluated
 * expression.
 */
template <class Expr> BOOST_UBLAS_INLINE decltype(auto) get_type(Expr &&e) {
  using namespace ::boost::hana::literals;

  auto expr = ::boost::yap::as_expr<tensor_expression>(std::forward<Expr>(e));
  using Expr_t = decltype(expr);

  if constexpr (is_terminal_type_expr<std::remove_reference_t<Expr_t>>::value) {
    using type = typename is_terminal_type_expr<
        std::remove_reference_t<Expr_t>>::value_type;
    if constexpr (::boost::numeric::ublas::detail::is_tensor_v<type> ||
                  ::boost::numeric::ublas::detail::is_vector_v<type> ||
                  ::boost::numeric::ublas::detail::is_matrix_v<type> ||
                  ::boost::numeric::ublas::detail::is_matrix_expression_v<type> ||
                  ::boost::numeric::ublas::detail::is_vector_expression_v<type>) {
      typename type::value_type s{};
      return s;
    } else
      return type{};
  }

  else if constexpr (Expr_t::kind != ::boost::yap::expr_kind::call) {
    if constexpr (Expr_t::kind == ::boost::yap::expr_kind::negate) {
      auto left_t = get_type(::boost::yap::get(expr, 0_c));
      return -left_t;

    } else if constexpr (Expr_t::kind == ::boost::yap::expr_kind::unary_plus) {
      return get_type(::boost::yap::get(expr, 0_c));
    } else {
      auto left_t = get_type(::boost::yap::left(expr));
      auto right_t = get_type(::boost::yap::right(expr));

      return ::boost::yap::evaluate(
          ::boost::yap::make_expression<Expr_t::kind>(left_t, right_t));
    }
  } else {
    using ret_t = typename function_return<std::remove_reference_t<decltype(
        ::boost::yap::value(::boost::yap::get(expr, 0_c)))>>::type;
    return ret_t{};
  }
}

/**
 * @brief Runs a static assert as a function checking each terminal node and
 * failing static assert as soon as a terminal is found that has a ublas
 * expression in it.
 *
 * @tparam Expr The type of the expression (deduced)
 *
 * @param e Expression / Tensor
 *
 * @return void
 */
template <class Expr> constexpr void assert_no_ublas_terminal(Expr &&e) {
  using namespace ::boost::hana::literals;

  auto expr = ::boost::yap::as_expr<tensor_expression>(std::forward<Expr>(e));
  using Expr_t = decltype(expr);

  if constexpr (is_terminal_type_expr<std::remove_reference_t<Expr_t>>::value) {
    using type = typename is_terminal_type_expr<
        std::remove_reference_t<Expr_t>>::value_type;
    static_assert(
        !(::boost::numeric::ublas::detail::is_matrix_expression_v<
              std::remove_reference_t<type>> ||
          ::boost::numeric::ublas::detail::is_vector_expression_v<
              std::remove_reference_t<type>>),
        "This Tensor Expression contains ublas expressions like vector/matrix "
        "expression. Hence this operation cannot be performed.");
  } else if constexpr (::boost::yap::expr_kind::negate == Expr_t::kind ||
                       Expr_t::kind == ::boost::yap::expr_kind::unary_plus ||
                       Expr_t::kind == ::boost::yap::expr_kind::call) {
    assert_no_ublas_terminal(::boost::yap::get(expr, 0_c));
  } else {
    assert_no_ublas_terminal(::boost::yap::left(expr));
    assert_no_ublas_terminal(::boost::yap::right(expr));
  }
}

/**
 * @brief The end condition for terminating the varidiac `ublas::apply_impl`
 * calls
 *
 * @tparam Expr The type of expression (deduced)
 *
 * @tparam Callable The Type of callable (deduced)
 *
 * @param e Expression / Tensor
 *
 * @param c Callable
 *
 * @return a new expression after applying the callable.
 */
template <class Expr, class Callable>
constexpr decltype(auto) apply_impl(Expr &&e, Callable c) {
  auto expr = ::boost::yap::as_expr<tensor_expression>(std::forward<Expr>(e));

  assert_no_ublas_terminal(expr);
  auto arg = get_type(expr);
  using arg_t = decltype(arg) const &;
  using ret_t = std::remove_reference_t<decltype(c(arg))>;

  using signature = ret_t(arg_t);

  static_assert(!std::is_same_v<void, ret_t>,
                "Callable must return non-void type");

  static_assert(
      std::is_convertible_v<Callable, std::function<signature>>,
      "Invalid signature for the last callable, expression value_type cannot "
      "be "
      "converted to callable's formal parameter. You can make Callable a "
      "generic "
      "lambda that takes only one argument by const-reference");

  ret_t (*func)(arg_t) = c;

  return ::boost::yap::make_expression<tensor_expression,
                                       ::boost::yap::expr_kind::call>(
      ::boost::yap::make_terminal<tensor_expression>(func),
      std::forward<decltype(expr)>(expr));
};

/**
 *  @brief Implementation for `ublas::apply`
 *
 * @tparam Expr The type of expression (deduced)
 *
 * @tparam FirstCallable The Type of First callable (deduced)
 *
 * @tparam others The parameter pack of other callables (deduced)
 *
 * @param e Expression / Tensor
 *
 * @param c First Callable
 *
 * @param x Other Callables
 *
 * @return A new expression representing the result of all callable applied into
 * Expression.
 */

template <class Expr, class FirstCallable, class... others>
constexpr decltype(auto) apply_impl(Expr &&e, FirstCallable c, others... x) {
  auto expr = ::boost::yap::as_expr<tensor_expression>(std::forward<Expr>(e));

  assert_no_ublas_expression(expr);
  auto arg = get_type(expr);
  using arg_t = decltype(arg) const &;
  using ret_t = std::remove_reference_t<decltype(c(arg))>;

  using signature = ret_t(arg_t);

  static_assert(!std::is_same_v<void, ret_t>,
                "Callable must return non-void type");

  static_assert(
      std::is_convertible_v<FirstCallable, std::function<signature>>,
      "Invalid signature for the callable, expression value_type cannot "
      "be "
      "converted to callable's formal parameter. You can make Callable a "
      "generic "
      "lambda that takes only one argument by const-reference");

  ret_t (*func)(arg_t) = c;

  auto intermediate_expr =
      ::boost::yap::make_expression<tensor_expression,
                                    ::boost::yap::expr_kind::call>(
          ::boost::yap::make_terminal<tensor_expression>(func),
          std::forward<decltype(expr)>(expr));

  return apply_impl(std::move(intermediate_expr), x...);
}

} // namespace boost::numeric::ublas::detail

#endif // UBLAS_EXPRESSION_UTILS_HPP
