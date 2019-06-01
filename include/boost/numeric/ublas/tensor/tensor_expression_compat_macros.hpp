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

#ifndef BOOST_UBLAS_TENSOR_EXPRESSION_COMPAT_HPP
#define BOOST_UBLAS_TENSOR_EXPRESSION_COMPAT_HPP

#include <boost/yap/yap.hpp>

namespace boost::numeric::ublas {

template <class T, class F, class A> class tensor;

namespace detail {
template <boost::yap::expr_kind Kind, typename type> class tensor_expression;
}

} // namespace boost::numeric::ublas
/**
 * Let me explain the hack I have used to integrate the ublas expression with
 * yap expression. We evaluate the ublas expression to build a tensor out of the
 * matrix/vector expression, this tensor is then used to build the yap
 * expression
 */

// todo(coder3101): In future write a intermediate special terminal node in YAP
// that can lazily extract values from ublas expressions. This however forces us
// to write all the transforms again.

#define BOOST_UBLAS_TENSOR_OPERATOR_COMPATIBILITY_WITH(expr_name, op_symbol)   \
  template <boost::yap::expr_kind K, typename args, class derived_type>        \
  decltype(auto) operator op_symbol(                                           \
      boost::numeric::ublas::detail::tensor_expression<K, args> &t_expr,       \
      boost::numeric::ublas::expr_name<derived_type> &v_expr) {                \
    return t_expr op_symbol boost::numeric::ublas::tensor{v_expr};             \
  }                                                                            \
  template <boost::yap::expr_kind K, typename args, class derived_type>        \
  decltype(auto) operator op_symbol(                                           \
      boost::numeric::ublas::expr_name<derived_type> &v_expr,                  \
      boost::numeric::ublas::detail::tensor_expression<K, args> &t_expr) {     \
    return boost::numeric::ublas::tensor{v_expr} op_symbol t_expr;             \
  }

#endif // BOOST_UBLAS_TENSOR_EXPRESSION_COMPAT_HPP
