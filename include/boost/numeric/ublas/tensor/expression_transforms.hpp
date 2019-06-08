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

#ifndef BOOST_UBLAS_TENSOR_EXPRESSION_TRANSFORM_HPP
#define BOOST_UBLAS_TENSOR_EXPRESSION_TRANSFORM_HPP

#include <boost/yap/yap.hpp>

namespace boost::numeric::ublas {

template <class size_type> class basic_extents;

namespace detail {

template <boost::yap::expr_kind Kind, typename Tuple> struct tensor_expression;

}

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas::detail::transforms {

struct at_index {
  template <class T, class F, class A>
  decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::terminal>,
             ::boost::numeric::ublas::detail::tensor<T, F, A> &terminal) {
    return boost::yap::make_terminal(std::move(terminal(index)));
  }
  size_t index;
};

struct scalar_extent_type {
  // This is the assumed extent type of a Scalar in the expression. The
  // transform get_extents returns this object if called on a scalar terminal
  // node.
};

// todo(@coder3101): Make a constexpr version of this transform once constexpr
// based extents are ready.

struct get_extents {

  template <class T, class F, class A>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::terminal>,
             ::boost::numeric::ublas::detail::tensor<T, F, A> &terminal) {
    return terminal.extents();
  }

  template <class LExpr, class RExpr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::minus>, LExpr &lexpr,
             RExpr &rexpr) {

    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr),
        *this); // left-side extent
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr),
        *this); // right-side extent

    // If both left and right sides are scalars then the result is also scalar
    // type. Return scalar type extent.
    if constexpr (std::is_same<decltype(left), scalar_extent_type>::value &&
                  std::is_same<decltype(right), scalar_extent_type>::value)
      return scalar_extent_type{};

    // If left side is only scalar, it means right will be a non-scalar. Return
    // right's extent
    else if constexpr (std::is_same<decltype(left), scalar_extent_type>::value)
      return right;

    // If right side is only scalar, it means left will be a non-scalar. Return
    // left's extent
    else if constexpr (std::is_same<decltype(right), scalar_extent_type>::value)
      return left;

    // otherwise both are non-scalars, assert they have same extent for
    // operation and return any one of the two
    else {
      if (left != right)
        throw std::runtime_error("Cannot Subtract Tensor of shapes " +
                                 left.to_string() + " and " +
                                 right.to_string());
      return left;
    }
  }

  template <class LExpr, class RExpr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::multiplies>,
             LExpr &lexpr, RExpr &rexpr) {

    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr), *this);
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr), *this);
    if constexpr (std::is_same<decltype(left), scalar_extent_type>::value &&
                  std::is_same<decltype(right), scalar_extent_type>::value)
      return scalar_extent_type{};
    else if constexpr (std::is_same<decltype(left), scalar_extent_type>::value)
      return right;
    else if constexpr (std::is_same<decltype(right), scalar_extent_type>::value)
      return left;
    else {
      if (left != right)
        throw std::runtime_error("Cannot Multiply Tensor of shapes " +
                                 left.to_string() + " and " +
                                 right.to_string());
      return left;
    }
  }

  template <class LExpr, class RExpr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::divides>,
             LExpr &lexpr, RExpr &rexpr) {

    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr), *this);
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr), *this);
    if constexpr (std::is_same<decltype(left), scalar_extent_type>::value &&
                  std::is_same<decltype(right), scalar_extent_type>::value)
      return scalar_extent_type{};
    else if constexpr (std::is_same<decltype(left), scalar_extent_type>::value)
      return right;
    else if constexpr (std::is_same<decltype(right), scalar_extent_type>::value)
      return left;
    else {
      if (left != right)
        throw std::runtime_error("Cannot Divide Tensor of shapes " +
                                 left.to_string() + " and " +
                                 right.to_string());
      return left;
    }
  }

  template <class LExpr, class RExpr>
  constexpr decltype(auto) operator()(::boost::yap::expr_tag<boost::yap::expr_kind::plus>,
                            LExpr &lexpr, RExpr &rexpr) {
    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr), *this);
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr), *this);

    if constexpr (std::is_same<decltype(left), scalar_extent_type>::value &&
                  std::is_same<decltype(right), scalar_extent_type>::value)
      return scalar_extent_type{};
    else if constexpr (std::is_same<decltype(left), scalar_extent_type>::value)
      return right;
    else if constexpr (std::is_same<decltype(right), scalar_extent_type>::value)
      return left;
    else {
      if (left != right)
        throw std::runtime_error("Cannot Add Tensor of shapes " +
                                 left.to_string() + " and " +
                                 right.to_string());
      return left;
    }
  }

  template <class Expr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::negate>,
             Expr &expr) {
    return boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(expr), *this);
  }

  template <class Expr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::unary_plus>,
             Expr &expr) {
    return boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(expr), *this);
  }

  // Match Scalars so that we could avoid them from expression completely as
  // they do not have extents. return the special type called scalar_extent_type
  // which makes the caller know that this node is extent-less.

  template <typename scalar_t>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
             scalar_t &) {
    return scalar_extent_type{};
  }
};

template <class T, class F = ::boost::numeric::ublas::column_major,
          class A = std::vector<T, std::allocator<T>>>
struct evaluate_ublas_expr {
  template <template <typename...> class outer, class... inner>
  constexpr decltype(auto)
  operator()(boost::yap::expr_tag<boost::yap::expr_kind::terminal>,
             outer<inner...> &expr) {
    if constexpr (std::is_base_of_v<::boost::numeric::ublas::vector_expression<
                                        outer<inner...>>,
                                    outer<inner...>>) {
      return ::boost::numeric::ublas::tensor<T, F, A>{
          std::move(::boost::numeric::ublas::vector<T>{expr})};
    } else if constexpr (std::is_base_of_v<
                             ::boost::numeric::ublas::matrix_expression<
                                 outer<inner...>>,
                             outer<inner...>>) {
      return ::boost::numeric::ublas::tensor<T, F, A>{
          std::move(::boost::numeric::ublas::matrix<T>{expr})};
    } else
      return boost::yap::make_terminal<detail::tensor_expression>(
          std::move(expr));
  }
};

} // namespace boost::numeric::ublas::detail::transforms

#endif
