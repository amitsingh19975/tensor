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

#include <boost/yap/yap.hpp>

namespace boost
{
namespace numeric
{
namespace ublas
{

template <class element_type, class storage_format, class storage_type>
class tensor;

template <class E>
class matrix_expression;

template <class E>
class vector_expression;

namespace detail
{
template <boost::yap::expr_kind K, typename A> /* A : Hana Arguments */
class tensor_expression;
}

} // namespace ublas
} // namespace numeric
} // namespace boost

// #define BINARY_TENSOR_TENSOR_OPERATOR_SET(YAP_OP_NAME, SYMBOL)                                        \
//     template <boost::yap::expr_kind A, typename HanaA,                                                \
//               boost::yap::expr_kind B, typename HanaB>                                                \
//     decltype(auto) operator SYMBOL(boost::numeric::ublas::detail::tensor_expression<A, HanaA> &lexpr, \
//                                    boost::numeric::ublas::detail::tensor_expression<B, HanaB> &rexpr) \
//     {                                                                                                 \
//         auto new_expr = boost::yap::make_expression<boost::numeric::ublas::detail::tensor_expression, YAP_OP_NAME> \
//         (boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(lexpr),                \
//          boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(rexpr));               \
//         return new_expr;                                                                            \
//     }

// //BINARY_TENSOR_TENSOR_OPERATOR_SET(boost::yap::expr_kind::plus, +);
// BINARY_TENSOR_TENSOR_OPERATOR_SET(boost::yap::expr_kind::minus, -);
// BINARY_TENSOR_TENSOR_OPERATOR_SET(boost::yap::expr_kind::multiplies, *);
// BINARY_TENSOR_TENSOR_OPERATOR_SET(boost::yap::expr_kind::divides, /);

// template <boost::yap::expr_kind a, typename b, boost::yap::expr_kind c, typename d>
// auto operator+(boost::numeric::ublas::detail::tensor_expression<a,b> &l, boost::numeric::ublas::detail::tensor_expression<c,d>&r){
//     return boost::yap::make_expression<boost::numeric::ublas::detail::tensor_expression, boost::yap::expr_kind::plus>(boost::yap::as_expr(l), boost::yap::as_expr(r));
// }

BOOST_YAP_USER_BINARY_OPERATOR(plus, boost::numeric::ublas::detail::tensor_expression, boost::numeric::ublas::detail::tensor_expression);
#endif