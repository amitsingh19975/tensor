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

#ifndef BOOST_UBLAS_EXPRESSION_OPERATOR_HPP
#define BOOST_UBLAS_EXPRESSION_OPERATOR_HPP

#include "functions.hpp"
#include "multi_index_utility.hpp"
#include "tensor_cast_macros.hpp"
#include "ublas_type_traits.hpp"
#include <boost/yap/user_macros.hpp>

namespace boost::numeric::ublas {

namespace detail {
template <boost::yap::expr_kind K, typename A> struct tensor_expression;

}

BOOST_UBLAS_EAGER_TENSOR_CAST(static_tensor_cast, static_cast)
BOOST_UBLAS_EAGER_TENSOR_CAST(dynamic_tensor_cast, dynamic_cast)
BOOST_UBLAS_EAGER_TENSOR_CAST(reinterpret_tensor_cast, reinterpret_cast)

} // namespace boost::numeric::ublas

// Tensor to expr
BOOST_YAP_USER_UDT_UNARY_OPERATOR(
    negate, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)
BOOST_YAP_USER_UDT_UNARY_OPERATOR(
    unary_plus, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)

BOOST_YAP_USER_UDT_ANY_BINARY_OPERATOR(
    plus, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)

BOOST_YAP_USER_UDT_ANY_BINARY_OPERATOR(
    minus, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)

BOOST_YAP_USER_UDT_ANY_BINARY_OPERATOR(
    multiplies, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)

BOOST_YAP_USER_UDT_ANY_BINARY_OPERATOR(
    divides, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)

BOOST_YAP_USER_UDT_ANY_BINARY_OPERATOR(
    less, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)

BOOST_YAP_USER_UDT_ANY_BINARY_OPERATOR(
    less_equal, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)

BOOST_YAP_USER_UDT_ANY_BINARY_OPERATOR(
    equal_to, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)

BOOST_YAP_USER_UDT_ANY_BINARY_OPERATOR(
    not_equal_to, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)

BOOST_YAP_USER_UDT_ANY_BINARY_OPERATOR(
    greater, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)

BOOST_YAP_USER_UDT_ANY_BINARY_OPERATOR(
    greater_equal, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::is_tensor)

// Expr to Expr
BOOST_YAP_USER_BINARY_OPERATOR(plus,
                               boost::numeric::ublas::detail::tensor_expression,
                               boost::numeric::ublas::detail::tensor_expression)
BOOST_YAP_USER_BINARY_OPERATOR(minus,
                               boost::numeric::ublas::detail::tensor_expression,
                               boost::numeric::ublas::detail::tensor_expression)
BOOST_YAP_USER_BINARY_OPERATOR(multiplies,
                               boost::numeric::ublas::detail::tensor_expression,
                               boost::numeric::ublas::detail::tensor_expression)
BOOST_YAP_USER_BINARY_OPERATOR(divides,
                               boost::numeric::ublas::detail::tensor_expression,
                               boost::numeric::ublas::detail::tensor_expression)
BOOST_YAP_USER_UNARY_OPERATOR(unary_plus,
                              boost::numeric::ublas::detail::tensor_expression,
                              boost::numeric::ublas::detail::tensor_expression)
BOOST_YAP_USER_UNARY_OPERATOR(negate,
                              boost::numeric::ublas::detail::tensor_expression,
                              boost::numeric::ublas::detail::tensor_expression)
BOOST_YAP_USER_BINARY_OPERATOR(equal_to,
                               boost::numeric::ublas::detail::tensor_expression,
                               boost::numeric::ublas::detail::tensor_expression)
BOOST_YAP_USER_BINARY_OPERATOR(less,
                               boost::numeric::ublas::detail::tensor_expression,
                               boost::numeric::ublas::detail::tensor_expression)
BOOST_YAP_USER_BINARY_OPERATOR(not_equal_to,
                               boost::numeric::ublas::detail::tensor_expression,
                               boost::numeric::ublas::detail::tensor_expression)
BOOST_YAP_USER_BINARY_OPERATOR(greater,
                               boost::numeric::ublas::detail::tensor_expression,
                               boost::numeric::ublas::detail::tensor_expression)
BOOST_YAP_USER_BINARY_OPERATOR(greater_equal,
                               boost::numeric::ublas::detail::tensor_expression,
                               boost::numeric::ublas::detail::tensor_expression)
BOOST_YAP_USER_BINARY_OPERATOR(less_equal,
                               boost::numeric::ublas::detail::tensor_expression,
                               boost::numeric::ublas::detail::tensor_expression)
// Tensor Contraction
template <class tensor_type_left, class tuple_type_left,
          class tensor_type_right, class tuple_type_right>
auto operator*(std::pair<tensor_type_left const &, tuple_type_left> lhs,
               std::pair<tensor_type_right const &, tuple_type_right> rhs) {

  using namespace boost::numeric::ublas;

  auto const &tensor_left = lhs.first;
  auto const &tensor_right = rhs.first;

  auto multi_index_left = lhs.second;
  auto multi_index_right = rhs.second;

  static constexpr auto num_equal_ind =
      number_equal_indexes<tuple_type_left, tuple_type_right>::value;

  if constexpr (num_equal_ind == 0) {
    return tensor_left * tensor_right;
  } else if constexpr (num_equal_ind ==
                           std::tuple_size<tuple_type_left>::value &&
                       std::is_same<tuple_type_left, tuple_type_right>::value) {
    return boost::numeric::ublas::inner_prod(tensor_left, tensor_right);
  } else {
    auto array_index_pairs =
        index_position_pairs(multi_index_left, multi_index_right);
    auto index_pairs = array_to_vector(array_index_pairs);
    return boost::numeric::ublas::prod(tensor_left, tensor_right,
                                       index_pairs.first, index_pairs.second);
  }
}

// Assign Operators
template <class T, class F, class V, class Expr>
auto operator+=(boost::numeric::ublas::tensor<T, V, F> &lhs, Expr const &e) {
  decltype(auto) expr =
      boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(e);
  auto shape = boost::yap::transform(
      expr, boost::numeric::ublas::detail::transforms::get_extents{});
  if (shape != lhs.extents()) {
    throw std::runtime_error("Cannot apply operator += with extents " +
                             lhs.extents().to_string() + " and " +
                             shape.to_string());
  }
  auto new_expr = lhs + expr;
  new_expr.eval_to(lhs);
  return lhs;
}

template <class T, class F, class V, class Expr>
auto operator-=(boost::numeric::ublas::tensor<T, V, F> &lhs, Expr const &e) {
  decltype(auto) expr =
      boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(e);
  auto shape = boost::yap::transform(
      expr, boost::numeric::ublas::detail::transforms::get_extents{});
  if (shape != lhs.extents()) {
    throw std::runtime_error("Cannot apply operator -= with extents " +
                             lhs.extents().to_string() + " and " +
                             shape.to_string());
  }
  auto new_expr = lhs - expr;
  new_expr.eval_to(lhs);
  return lhs;
}
template <class T, class F, class V, class Expr>
auto operator*=(boost::numeric::ublas::tensor<T, V, F> &lhs, Expr const &e) {
  decltype(auto) expr =
      boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(e);
  auto shape = boost::yap::transform(
      expr, boost::numeric::ublas::detail::transforms::get_extents{});
  if (shape != lhs.extents()) {
    throw std::runtime_error("Cannot apply operator *= with extents " +
                             lhs.extents().to_string() + " and " +
                             shape.to_string());
  }
  auto new_expr = lhs * expr;
  new_expr.eval_to(lhs);
  return lhs;
}
template <class T, class F, class V, class Expr>
auto operator/=(boost::numeric::ublas::tensor<T, V, F> &lhs, Expr const &e) {
  decltype(auto) expr =
      boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(e);
  auto shape = boost::yap::transform(
      expr, boost::numeric::ublas::detail::transforms::get_extents{});
  if (shape != lhs.extents()) {
    throw std::runtime_error("Cannot apply operator /= with extents " +
                             lhs.extents().to_string() + " and " +
                             shape.to_string());
  }
  auto new_expr = lhs / expr;
  new_expr.eval_to(lhs);
  return lhs;
}

template <boost::yap::expr_kind K, typename Tuple>
bool operator!(boost::numeric::ublas::detail::tensor_expression<K, Tuple>& expr){
  bool result = expr;
  return !result;
}

#endif
