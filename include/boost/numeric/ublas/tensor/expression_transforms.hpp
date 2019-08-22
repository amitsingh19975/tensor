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

#ifndef BOOST_UBLAS_TENSOR_EXPRESSION_TRANSFORM_HPP
#define BOOST_UBLAS_TENSOR_EXPRESSION_TRANSFORM_HPP

#include <boost/numeric/ublas/detail/config.hpp>

#include <boost/type_traits/has_multiplies.hpp>
#include <boost/yap/yap.hpp>
#include "expression_transforms_traits.hpp"
#include "extents.hpp"
#include "ublas_type_traits.hpp"

namespace boost::numeric::ublas {

namespace detail {

template <boost::yap::expr_kind Kind, typename Tuple>
struct tensor_expression;

}

template <class T, class F, class A>
class tensor;

template <class T, class F, class A>
class matrix;

template <class T, class A>
class vector;

}  // namespace boost::numeric::ublas

namespace boost::numeric::ublas::detail::transforms {

/**
 * @brief A transform that extracts ith index value from terminal types.
 *
 * @note This transform when applied to an tensor_expression with terminal nodes
 * as tensor returns a new expression with terminal nodes as ith values of
 * tensor.
 *
 * @todo(coder3101): Constexpr-ify these transforms.
 *
 */
struct at_index {
  template <class T, class F, class A>
  BOOST_UBLAS_INLINE decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      ::boost::numeric::ublas::tensor<T, F, A> const &terminal) {
    return ::boost::yap::make_terminal(terminal(index));
  }
  template <class T, class F, class A>
  BOOST_UBLAS_INLINE decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      ::boost::numeric::ublas::matrix<T, F, A> const &terminal) {
    return ::boost::yap::make_terminal(terminal(index));
  }
  template <class T, class A>
  BOOST_UBLAS_INLINE decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      ::boost::numeric::ublas::vector<T, A> const &terminal) {
    return ::boost::yap::make_terminal(terminal(index));
  }
  template <typename Expr>
  BOOST_UBLAS_INLINE decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      ::boost::numeric::ublas::matrix_expression<Expr> &terminal) {
    return ::boost::yap::make_terminal(terminal()(index));
  }
  template <typename Expr>
  BOOST_UBLAS_INLINE decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      ::boost::numeric::ublas::vector_expression<Expr> &terminal) {
    return ::boost::yap::make_terminal(terminal()(index));
  }
  template <class Optimized, class Unoptimized>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      std::variant<Optimized, Unoptimized> &variant) {
    auto visitor_lambda = [this](auto &&subexpr) {
      return boost::yap::transform(
          boost::yap::as_expr<detail::tensor_expression>(
              std::forward<decltype(subexpr)>(subexpr)),
          *this);
    };

    using return_optimized_t = decltype(
        std::declval<decltype(visitor_lambda)>()(std::declval<Optimized>()));

    using return_un_optimized_t = decltype(
        std::declval<decltype(visitor_lambda)>()(std::declval<Unoptimized>()));

    return boost::yap::make_terminal(std::visit(
        [&](auto &&e)
            -> std::variant<return_optimized_t, return_un_optimized_t> {
          return visitor_lambda(std::forward<decltype(e)>(e));
        },
        variant));
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
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      ::boost::numeric::ublas::tensor<T, F, A> &terminal) {
    return terminal.extents();
  }
  // This overload must be existing for const-refences
  // YAP takes everything by reference (non-const)
  // All our transforms do the same thing, and operators
  // Since, contractions are marked const, they pass const-referenced
  // tensor-terminal. It is important to have this overload for it.
  // We can omit for other terminal types because we know only
  // contractions are defined for tensor terminals.
  template <class T, class F, class A>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      ::boost::numeric::ublas::tensor<T, F, A> const &terminal) {
    return terminal.extents();
  }

  template <class T, class F, class A>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      ::boost::numeric::ublas::matrix<T, F, A> &terminal) {
    return ::boost::numeric::ublas::basic_extents<size_t>{terminal.size1(),
                                                          terminal.size2()};
  }
  template <class T, class A>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      ::boost::numeric::ublas::vector<T, A> &terminal) {
    return ::boost::numeric::ublas::basic_extents<size_t>{terminal.size(), 1};
  }

  template <class Optimized, class Unoptimized>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      std::variant<Optimized, Unoptimized> &variant) {
    return std::visit(
        [this](auto &&subexpr) {
          return boost::yap::transform(std::forward<decltype(subexpr)>(subexpr),
                                       *this);
        },
        variant);
  }

  template <class LExpr, class RExpr>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::minus>, LExpr &lexpr,
      RExpr &rexpr) {
    auto left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(lexpr),
        *this);
    auto right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(rexpr),
        *this);
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
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::multiplies>, LExpr &lexpr,
      RExpr &rexpr) {
    auto left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(lexpr),
        *this);
    auto right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(rexpr),
        *this);
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
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::divides>, LExpr &lexpr,
      RExpr &rexpr) {
    auto left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(lexpr),
        *this);
    auto right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(rexpr),
        *this);
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
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::plus>, LExpr &lexpr,
      RExpr &rexpr) {
    auto left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(lexpr),
        *this);
    auto right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(rexpr),
        *this);

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
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::negate>, Expr &expr) {
    return ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(expr),
        *this);
  }

  template <class Expr>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::unary_plus>, Expr &expr) {
    return ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(expr),
        *this);
  }

  template <class LExpr, class RExpr>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::equal_to>, LExpr &lexpr,
      RExpr &rexpr) {
    auto left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(lexpr),
        *this);
    auto right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(rexpr),
        *this);

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
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::not_equal_to>,
      LExpr &lexpr, RExpr &rexpr) {
    auto left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(lexpr),
        *this);
    auto right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(rexpr),
        *this);

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
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::less>, LExpr &lexpr,
      RExpr &rexpr) {
    auto left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(lexpr),
        *this);
    auto right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(rexpr),
        *this);

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
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::greater>, LExpr &lexpr,
      RExpr &rexpr) {
    auto left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(lexpr),
        *this);
    auto right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(rexpr),
        *this);

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
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::greater_equal>,
      LExpr &lexpr, RExpr &rexpr) {
    auto left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(lexpr),
        *this);
    auto right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(rexpr),
        *this);

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
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::less_equal>, LExpr &lexpr,
      RExpr &rexpr) {
    auto left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(lexpr),
        *this);
    auto right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(rexpr),
        *this);

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

  template <class Func, class Arg>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::call>, Func f, Arg &e2) {
    auto right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    return right;
  }

  // Match Scalars so that we could avoid them from expression completely as
  // they do not have extents. return the special type called
  // scalar_extent_type which makes the caller know that this node is
  // extent-less. Otherwise match

  template <typename Expr>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      Expr &terminal) {
    if constexpr (::boost::numeric::ublas::is_vector_expression_v<Expr>) {
      return basic_extents<size_t>{terminal.size(), 1};
    } else if constexpr (::boost::numeric::ublas::is_matrix_expression_v<
                             Expr>) {
      return basic_extents<size_t>{terminal.size1(), terminal.size2()};
    } else
      return basic_extents<size_t>{1};
  }
};

/**
 * @brief This transform counts the number of the relational operators that
 * appeared in an expression.
 */
struct expr_count_relational_operator {
  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::equal_to>, Expr1 &e1,
      Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e1),
        *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    equal_to_found = true;
    return left + right + 1u;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::not_equal_to>, Expr1 &e1,
      Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e1),
        *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    not_equal_to_found = true;
    return left + right + 1u;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::less>, Expr1 &e1,
      Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e1),
        *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    return left + right + 1u;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::less_equal>, Expr1 &e1,
      Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e1),
        *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    return left + right + 1u;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::greater>, Expr1 &e1,
      Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e1),
        *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    return left + right + 1u;
  }
  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::greater_equal>, Expr1 &e1,
      Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e1),
        *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    return left + right + 1u;
  }

  template <class Expr>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>, Expr &) {
    return 0;
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::plus>, Expr1 &e1,
      Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e1),
        *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    return left + right;
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::minus>, Expr1 &e1,
      Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e1),
        *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    return left + right;
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::multiplies>, Expr1 &e1,
      Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e1),
        *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    return left + right;
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::divides>, Expr1 &e1,
      Expr2 &e2) {
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e1),
        *this);
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    return left + right;
  }
  template <class Expr1>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::negate>, Expr1 &e1) {
    using namespace ::boost::hana::literals;
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(
            ::boost::yap::get(e1, 0_c)),
        *this);
    return left;
  }

  template <class Expr1>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::unary_plus>, Expr1 &e1) {
    using namespace ::boost::hana::literals;
    std::size_t left = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(
            ::boost::yap::get(e1, 0_c)),
        *this);
    return left;
  }

  template <class Func, class Arg>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::call>, Func f, Arg &e2) {
    std::size_t right = ::boost::yap::transform(
        ::boost::yap::as_expr<
            ::boost::numeric::ublas::detail::tensor_expression>(e2),
        *this);
    return right;
  }

  template <class Optimized, class Unoptimized>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      std::variant<Optimized, Unoptimized> &variant) {
    return std::visit(
        [this](auto &&subexpr) {
          return boost::yap::transform(std::forward<decltype(subexpr)>(subexpr),
                                       *this);
        },
        variant);
  }

  bool equal_to_found = false;
  bool not_equal_to_found = false;
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
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::not_equal_to>, Expr1 &e1,
      Expr2 &e2) {
    auto left =
        ::boost::yap::transform(::boost::yap::as_expr(e1), get_extents{});
    auto right =
        ::boost::yap::transform(::boost::yap::as_expr(e2), get_extents{});
    status = left.is_free_scalar() || right.is_free_scalar() || left == right;
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::equal_to>, Expr1 &e1,
      Expr2 &e2) {
    auto left =
        ::boost::yap::transform(::boost::yap::as_expr(e1), get_extents{});
    auto right =
        ::boost::yap::transform(::boost::yap::as_expr(e2), get_extents{});
    status = left.is_free_scalar() || right.is_free_scalar() || left == right;
  }

  template <class Optimized, class Unoptimized>
  constexpr decltype(auto) operator()(
      ::boost::yap::expr_tag<::boost::yap::expr_kind::terminal>,
      std::variant<Optimized, Unoptimized> &variant) {
    return std::visit(
        [this](auto &&subexpr) {
          return boost::yap::transform(std::forward<decltype(subexpr)>(subexpr),
                                       *this);
        },
        variant);
  }

  bool status = false;
};

struct evaluate_with_variant {
  template <class termA, class termB>
  decltype(auto) operator()(
      boost::yap::expr_tag<boost::yap::expr_kind::terminal>,
      std::variant<termA, termB> &v) {
    return std::visit(
        [this](auto &&subexpr) {
          return boost::yap::transform(
              boost::yap::as_expr(std::forward<decltype(subexpr)>(subexpr)),
              *this);
        },
        v);
  }

#define BINARY_MAKE_RECURSE_OVERLOAD(op_name, symbol)                    \
  template <class Lexpr, class Rexpr>                                    \
  decltype(auto) operator()(                                             \
      boost::yap::expr_tag<boost::yap::expr_kind::op_name>, Lexpr &&lhs, \
      Rexpr &&rhs) {                                                     \
    return boost::yap::transform(                                        \
        boost::yap::as_expr(std::forward<Lexpr>(lhs)), *this)            \
        symbol boost::yap::transform(                                    \
            boost::yap::as_expr(std::forward<Rexpr>(rhs)), *this);       \
  }

  BINARY_MAKE_RECURSE_OVERLOAD(plus, +);
  BINARY_MAKE_RECURSE_OVERLOAD(minus, -);
  BINARY_MAKE_RECURSE_OVERLOAD(multiplies, *);
  BINARY_MAKE_RECURSE_OVERLOAD(divides, /);

  BINARY_MAKE_RECURSE_OVERLOAD(equal_to, ==);
  BINARY_MAKE_RECURSE_OVERLOAD(not_equal_to, !=);
  BINARY_MAKE_RECURSE_OVERLOAD(greater, >);
  BINARY_MAKE_RECURSE_OVERLOAD(greater_equal, >=);
  BINARY_MAKE_RECURSE_OVERLOAD(less, <);
  BINARY_MAKE_RECURSE_OVERLOAD(less_equal, <=);

  template <class T>
  auto operator()(boost::yap::expr_tag<boost::yap::expr_kind::terminal>,
                  T &&t) {
    return std::forward<T>(t);
  }

  // Todo(coder3101): Implement for call operator and Unary has been left off.
};
}  // namespace boost::numeric::ublas::detail::transforms

#endif
