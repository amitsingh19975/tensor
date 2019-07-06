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
#include "ublas_type_traits.hpp"
#include <boost/yap/yap.hpp>

namespace boost::numeric::ublas {

namespace detail {

template <boost::yap::expr_kind Kind, typename Tuple> struct tensor_expression;

}

template <class T, class F, class A> class tensor;

template <class T, class F, class A> class matrix;

template <class T, class A> class vector;

} // namespace boost::numeric::ublas

/**
 * @brief This namespace contains all the transforms of YAP expression.
 */
namespace boost::numeric::ublas::detail::transforms {

/**
 * @brief A transform that extracts ith index value from terminal types.
 *
 * @note This transform when applied to an tensor_expression with terminal nodes
 * as tensor returns a new expression with terminal nodes as ith values of
 * tensor.
 *
 */
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

/**
 * @brief This transform returns the extent of the expression.
 *
 * @note If any extent inconsistency is found, this transform throws a runtime
 * exception. Also note that for vector the returned extent is `{vec.size, 1}`
 * and for matrix it is `{mat.size1, mat.size2}`.
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
    if constexpr (boost::numeric::ublas::is_vector_expression_v<Expr>) {
      return basic_extents<size_t>{terminal.size(), 1};
    } else if constexpr (boost::numeric::ublas::is_matrix_expression_v<Expr>) {
      return basic_extents<size_t>{terminal.size1(), terminal.size2()};
    } else
      return basic_extents<size_t>{1};
  }
};

/**
 * @brief A stateful transform that that sets status to true if the expression
 * to which it is applied has at-least one relational operator in it.
 *
 * @note If this transform sets status to true then only the expression can be
 * implicitly converted to bool type.
 *
 * @deprecated Please use `expr_count_relational_operator`.
 */
struct [[deprecated(
    "This stateless transform has been replaced with "
    "expr_count_relational_operator")]] expr_has_relational_operator {

  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<boost::yap::expr_kind::equal_to>, Expr1 &,
      Expr2 &) {
    status = true;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<boost::yap::expr_kind::not_equal_to>, Expr1 &,
      Expr2 &) {
    status = true;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<boost::yap::expr_kind::less>, Expr1 &, Expr2 &) {
    status = true;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<boost::yap::expr_kind::less_equal>, Expr1 &,
      Expr2 &) {
    status = true;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<boost::yap::expr_kind::greater>, Expr1 &,
      Expr2 &) {
    status = true;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<boost::yap::expr_kind::greater_equal>, Expr1 &,
      Expr2 &) {
    status = true;
  }

  bool status = false;
};

/**
 * @brief This transform counts the number of the relational operators that
 * appeared in an expression.
 */
struct expr_count_relational_operator {

  constexpr expr_count_relational_operator() = default;

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::equal_to>, Expr1 &e1,
             Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e1), *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e2), *this);
    return left + right + 1u;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::not_equal_to>,
             Expr1 &e1, Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e1), *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e2), *this);
    return left + right + 1u;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::less>, Expr1 &e1,
             Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e1), *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e2), *this);
    return left + right + 1u;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::less_equal>,
             Expr1 &e1, Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e1), *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e2), *this);
    return left + right + 1u;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::greater>, Expr1 &e1,
             Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e1), *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e2), *this);
    return left + right + 1u;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::greater_equal>,
             Expr1 &e1, Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e1), *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e2), *this);
    return left + right + 1u;
  }

  template <class Expr>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::terminal>, Expr &) {
    return 0;
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::plus>, Expr1 &e1,
             Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e1), *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e2), *this);
    return left + right;
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::minus>, Expr1 &e1,
             Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e1), *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e2), *this);
    return left + right;
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::multiplies>,
             Expr1 &e1, Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e1), *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e2), *this);
    return left + right;
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::divides>, Expr1 &e1,
             Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e1), *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<detail::tensor_expression>(e2), *this);
    return left + right;
  }
};

/**
 * @brief A stateful transform that sets the status to true if expression has
 * `==` operator in it.
 */

struct expr_has_equal_to_operator {
  constexpr expr_has_equal_to_operator() = default;

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::equal_to>, Expr1 &e1,
             Expr2 &e2) {
    status = true;
  }
  bool status = false;
};

/**
 * @brief A stateful transform that sets the status to true if expression has
 * `!=` operator in it.
 */
struct expr_has_not_equal_operator {
  constexpr expr_has_not_equal_operator() = default;

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::not_equal_to>,
             Expr1 &e1, Expr2 &e2) {
    status = true;
  }
  bool status = false;
};

/**
 * @brief If an expression has only one relational operator which is `==` or
 * `!=`, this transformed is called and results whether the left and right side
 * operands/expression have same extents. This is a stateful transform
 *
 * @note If called with an expression that has multiple relational operator, a
 * compile time error from YAP is thrown.
 */
struct is_equality_or_non_equality_extent_same {
  constexpr is_equality_or_non_equality_extent_same() = default;

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::not_equal_to>,
             Expr1 &e1, Expr2 &e2) {
    auto left =
        ::boost::yap::transform(::boost::yap::as_expr(e1), get_extents{});
    auto right =
        ::boost::yap::transform(::boost::yap::as_expr(e2), get_extents{});
    status = left.is_free_scalar() || right.is_free_scalar() || left == right;
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(::boost::yap::expr_tag<boost::yap::expr_kind::equal_to>, Expr1 &e1,
             Expr2 &e2) {
    auto left =
        ::boost::yap::transform(::boost::yap::as_expr(e1), get_extents{});
    auto right =
        ::boost::yap::transform(::boost::yap::as_expr(e2), get_extents{});
    status = left.is_free_scalar() || right.is_free_scalar() || left == right;
  }

  bool status = false;
};

struct apply_distributive_law {
  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(boost::yap::expr_tag<boost::yap::expr_kind::plus>, Expr1 &&e1,
             Expr2 &&e2) {
    return boost::yap::make_expression<
        boost::numeric::ublas::detail::tensor_expression,
        boost::yap::expr_kind::plus>(std::forward<Expr1>(e1),
                                     std::forward<Expr2>(e2));
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(boost::yap::expr_tag<boost::yap::expr_kind::minus>, Expr1 &&e1,
             Expr2 &&e2) {
    return boost::yap::make_expression<
        boost::numeric::ublas::detail::tensor_expression,
        boost::yap::expr_kind::minus>(std::forward<Expr1>(e1),
                                     std::forward<Expr2>(e2));
  }
};

} // namespace boost::numeric::ublas::detail::transforms

#endif
