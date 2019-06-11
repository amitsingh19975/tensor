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

#include "extents.hpp"
#include <boost/yap/yap.hpp>

namespace boost::numeric::ublas {

namespace detail {

template <boost::yap::expr_kind Kind, typename Tuple> struct tensor_expression;

}

template <class T, class F, class A> class tensor;

template <class T, class F, class A> class matrix;

template <class T, class A> class vector;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas::detail::transforms {

struct at_index {
  template <class T, class F, class A>
  decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::terminal>,
             ::boost::numeric::ublas::tensor<T, F, A> const &terminal) {
    return boost::yap::make_terminal(terminal(index));
  }
  template <class T, class F, class A>
  decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::terminal>,
             ::boost::numeric::ublas::matrix<T, F, A> const &terminal) {
    return boost::yap::make_terminal(terminal(index));
  }
  template <class T, class A>
  decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::terminal>,
             ::boost::numeric::ublas::vector<T, A> const &terminal) {
    return boost::yap::make_terminal(terminal(index));
  }
  template <typename Expr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
             boost::numeric::ublas::matrix_expression<Expr> &terminal) {
    return boost::yap::make_terminal(terminal()(index));
  }
  template <typename Expr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
             boost::numeric::ublas::vector_expression<Expr> &terminal) {
    return boost::yap::make_terminal(terminal()(index));
  }
  size_t index;
};

/*
 * We assume that basic_extents<size_t>(1) is a scalar; Every Scalar operand
 * returns this value.
 */

struct get_extents {

  template <class T, class F, class A>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::terminal>,
             ::boost::numeric::ublas::tensor<T, F, A> &terminal) {
    return terminal.extents();
  }
  template <class T, class F, class A>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::terminal>,
             ::boost::numeric::ublas::matrix<T, F, A> &terminal) {
    return boost::numeric::ublas::basic_extents<size_t>{terminal.size1(),
                                                        terminal.size2()};
  }
  template <class T, class A>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::terminal>,
             ::boost::numeric::ublas::vector<T, A> &terminal) {
    return boost::numeric::ublas::basic_extents<size_t>{terminal.size(), 1};
  }

  template <class LExpr, class RExpr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::minus>, LExpr &lexpr,
             RExpr &rexpr) {

    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr), *this);
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr), *this);
    if (left.is_free_scalar() && right.is_free_scalar())
      return basic_extents<size_t>{1};
    else if (left.is_free_scalar())
      return right;
    else if (right.is_free_scalar())
      return left;
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
    if (left.is_free_scalar() && right.is_free_scalar())
      return basic_extents<size_t>{1};
    else if (left.is_free_scalar())
      return right;
    else if (right.is_free_scalar())
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
    if (left.is_free_scalar() && right.is_free_scalar())
      return basic_extents<size_t>{1};
    else if (left.is_free_scalar())
      return right;
    else if (right.is_free_scalar())
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
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::plus>, LExpr &lexpr,
             RExpr &rexpr) {
    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr), *this);
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr), *this);

    if (left.is_free_scalar() && right.is_free_scalar())
      return basic_extents<size_t>{1};
    else if (left.is_free_scalar())
      return right;
    else if (right.is_free_scalar())
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

  template <class LExpr, class RExpr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::equal_to>,
             LExpr &lexpr, RExpr &rexpr) {
    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr), *this);
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr), *this);

    if (left.is_free_scalar() && right.is_free_scalar())
      return basic_extents<size_t>{1};
    else if (left.is_free_scalar())
      return right;
    else if (right.is_free_scalar())
      return left;
    else {
      if (left != right)
        throw std::runtime_error("Cannot perform == on tensor of shapes " +
                                 left.to_string() + " and " +
                                 right.to_string());
      return left;
    }
  }

  template <class LExpr, class RExpr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::not_equal_to>,
             LExpr &lexpr, RExpr &rexpr) {
    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr), *this);
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr), *this);

    if (left.is_free_scalar() && right.is_free_scalar())
      return basic_extents<size_t>{1};
    else if (left.is_free_scalar())
      return right;
    else if (right.is_free_scalar())
      return left;
    else {
      if (left != right)
        throw std::runtime_error("Cannot perform != on tensor of shapes " +
                                 left.to_string() + " and " +
                                 right.to_string());
      return left;
    }
  }
  template <class LExpr, class RExpr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::less>, LExpr &lexpr,
             RExpr &rexpr) {
    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr), *this);
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr), *this);

    if (left.is_free_scalar() && right.is_free_scalar())
      return basic_extents<size_t>{1};
    else if (left.is_free_scalar())
      return right;
    else if (right.is_free_scalar())
      return left;
    else {
      if (left != right)
        throw std::runtime_error("Cannot perform < on tensor of shapes " +
                                 left.to_string() + " and " +
                                 right.to_string());
      return left;
    }
  }
  template <class LExpr, class RExpr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::greater>,
             LExpr &lexpr, RExpr &rexpr) {
    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr), *this);
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr), *this);

    if (left.is_free_scalar() && right.is_free_scalar())
      return basic_extents<size_t>{1};
    else if (left.is_free_scalar())
      return right;
    else if (right.is_free_scalar())
      return left;
    else {
      if (left != right)
        throw std::runtime_error("Cannot perform > on tensor of shapes " +
                                 left.to_string() + " and " +
                                 right.to_string());
      return left;
    }
  }
  template <class LExpr, class RExpr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::greater_equal>,
             LExpr &lexpr, RExpr &rexpr) {
    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr), *this);
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr), *this);

    if (left.is_free_scalar() && right.is_free_scalar())
      return basic_extents<size_t>{1};
    else if (left.is_free_scalar())
      return right;
    else if (right.is_free_scalar())
      return left;
    else {
      if (left != right)
        throw std::runtime_error("Cannot perform >= on tensor of shapes " +
                                 left.to_string() + " and " +
                                 right.to_string());
      return left;
    }
  }
  template <class LExpr, class RExpr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::less_equal>,
             LExpr &lexpr, RExpr &rexpr) {
    auto left = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(lexpr), *this);
    auto right = boost::yap::transform(
        boost::yap::as_expr<detail::tensor_expression>(rexpr), *this);

    if (left.is_free_scalar() && right.is_free_scalar())
      return basic_extents<size_t>{1};
    else if (left.is_free_scalar())
      return right;
    else if (right.is_free_scalar())
      return left;
    else {
      if (left != right)
        throw std::runtime_error("Cannot perform <= on tensor of shapes " +
                                 left.to_string() + " and " +
                                 right.to_string());
      return left;
    }
  }

  // Match Scalars so that we could avoid them from expression completely as
  // they do not have extents. return the special type called
  // scalar_extent_type which makes the caller know that this node is
  // extent-less. Otherwise match

  template <typename Expr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
             Expr &terminal) {
    if constexpr (std::is_base_of_v<
                      boost::numeric::ublas::vector_expression<Expr>, Expr>) {
      return basic_extents<size_t>{terminal.size(), 1};
    }
    if constexpr (std::is_base_of_v<
                      boost::numeric::ublas::matrix_expression<Expr>, Expr>) {
      return basic_extents<size_t>{terminal.size1(), terminal.size2()};
    }
    return basic_extents<size_t>{1};
  }
};

struct expr_has_logical_operator {

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::equal_to>, Expr1 &,
             Expr2 &) {
    status = true;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::not_equal_to>,
             Expr1 &, Expr2 &) {
    status = true;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::less>, Expr1 &,
             Expr2 &) {
    status = true;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::less_equal>, Expr1 &,
             Expr2 &) {
    status = true;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::greater>, Expr1 &,
             Expr2 &) {
    status = true;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::greater_equal>,
             Expr1 &, Expr2 &) {
    status = true;
  }

  bool status = false;
};

} // namespace boost::numeric::ublas::detail::transforms

#endif
