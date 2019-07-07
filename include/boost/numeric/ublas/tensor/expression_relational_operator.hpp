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

#ifndef BOOST_UBLAS_EXPRESSION_RELATIONAL_OPERATOR_HPP
#define BOOST_UBLAS_EXPRESSION_RELATIONAL_OPERATOR_HPP

#include "ublas_type_traits.hpp"
#include <boost/yap/user_macros.hpp>

namespace boost::numeric::ublas::detail {
template <boost::yap::expr_kind K, typename A> struct tensor_expression;

} // namespace boost::numeric::ublas::detail

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

#endif // UBLAS_EXPRESSION_RELATIONAL_OPERATOR_HPP
