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

#include "tensor_cast_macros.hpp"
#include "tensor_expression_compat_macros.hpp"
#include <boost/yap/user_macros.hpp>
#include <boost/yap/yap.hpp>
#include <type_traits>

namespace boost::numeric::ublas {

template <class E> class matrix_expression;

template <class E> class vector_expression;

namespace detail {
template <boost::yap::expr_kind K, typename A> /* A : Hana Arguments */
struct tensor_expression;
}

// tensor-casts
BOOST_UBLAS_EAGER_TENSOR_CAST(static_tensor_cast, static_cast);

BOOST_UBLAS_EAGER_TENSOR_CAST(dynamic_tensor_cast, dynamic_cast);

BOOST_UBLAS_EAGER_TENSOR_CAST(reinterpret_tensor_cast, reinterpret_cast);

} // namespace boost::numeric::ublas

// Binary tensor-tensor operator
BOOST_YAP_USER_BINARY_OPERATOR(
    plus, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::detail::tensor_expression);

BOOST_YAP_USER_BINARY_OPERATOR(
    minus, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::detail::tensor_expression);

BOOST_YAP_USER_BINARY_OPERATOR(
    multiplies, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::detail::tensor_expression);

BOOST_YAP_USER_BINARY_OPERATOR(
    divides, boost::numeric::ublas::detail::tensor_expression,
    boost::numeric::ublas::detail::tensor_expression);

// Unary tensor operator
BOOST_YAP_USER_UNARY_OPERATOR(negate,
                              boost::numeric::ublas::detail::tensor_expression,
                              boost::numeric::ublas::detail::tensor_expression);

BOOST_YAP_USER_UNARY_OPERATOR(unary_plus,
                              boost::numeric::ublas::detail::tensor_expression,
                              boost::numeric::ublas::detail::tensor_expression);

// tensor-vector, tensor-matrix expression compatibility. This is a hack please
// check tensor_expression_compat.hpp for more information.

BOOST_UBLAS_TENSOR_OPERATOR_COMPATIBILITY_WITH(matrix_expression, +);
BOOST_UBLAS_TENSOR_OPERATOR_COMPATIBILITY_WITH(matrix_expression, -);
BOOST_UBLAS_TENSOR_OPERATOR_COMPATIBILITY_WITH(matrix_expression, /);
BOOST_UBLAS_TENSOR_OPERATOR_COMPATIBILITY_WITH(matrix_expression, *);

BOOST_UBLAS_TENSOR_OPERATOR_COMPATIBILITY_WITH(vector_expression, +);
BOOST_UBLAS_TENSOR_OPERATOR_COMPATIBILITY_WITH(vector_expression, -);
BOOST_UBLAS_TENSOR_OPERATOR_COMPATIBILITY_WITH(vector_expression, /);
BOOST_UBLAS_TENSOR_OPERATOR_COMPATIBILITY_WITH(vector_expression, *);

// todo(coder3101): Add Tensor Contraction Support

// Tensor Contraction
// template <class tensor_type_left, class tuple_type_left,
//           class tensor_type_right, class tuple_type_right>
// auto operator*(std::pair<tensor_type_left const &, tuple_type_left> lhs,
//                std::pair<tensor_type_right const &, tuple_type_right> rhs) {
//   using namespace boost::numeric::ublas;

//   auto const &tensor_left = lhs.first;
//   auto const &tensor_right = rhs.first;

//   auto multi_index_left = lhs.second;
//   auto multi_index_right = rhs.second;

//   static constexpr auto num_equal_ind =
//       number_equal_indexes<tuple_type_left, tuple_type_right>::value;

//   if constexpr (num_equal_ind == 0) {
//     return tensor_left * tensor_right;
//   } else if constexpr (num_equal_ind ==
//                            std::tuple_size<tuple_type_left>::value &&
//                        std::is_same<tuple_type_left,
//                        tuple_type_right>::value) {
//     return boost::numeric::ublas::inner_prod(tensor_left, tensor_right);
//   } else {
//     auto array_index_pairs =
//         index_position_pairs(multi_index_left, multi_index_right);
//     auto index_pairs = array_to_vector(array_index_pairs);
//     return boost::numeric::ublas::prod(tensor_left, tensor_right,
//                                        index_pairs.first,
//                                        index_pairs.second);
//   }
// }

#endif
