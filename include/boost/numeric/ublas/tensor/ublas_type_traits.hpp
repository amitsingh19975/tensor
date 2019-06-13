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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_HPP

#include <boost/yap/yap.hpp>
#include <type_traits>

namespace boost::numeric::ublas {

// Forward declare classes
template <class T, class F, class A> class tensor;
template <class T, class F, class A> class matrix;
template <class T, class A> class vector;
namespace detail {
template <boost::yap::expr_kind K, typename Tuple> class tensor_expression;
}

/**
 * @brief static constexpr `value` is resolved to true if template type is a
 * tensor.
 *
 * @tparam T the type to check for if its tensor or not.
 */
template <class T> struct is_tensor { static constexpr bool value = false; };
template <class T, class F, class A> struct is_tensor<tensor<T, F, A>> {
  static constexpr bool value = true;
};

/**
 * @brief static constexpr `value` is resolved to true if template type is a
 * matrix.
 *
 * @tparam T the type to check for if its matrix or not.
 */
template <class T> struct is_matrix { static constexpr bool value = false; };
template <class T, class F, class A> struct is_matrix<matrix<T, F, A>> {
  static constexpr bool value = true;
};

/**
 * @brief static constexpr `value` is resolved to true if template type is a
 * vector.
 *
 * @tparam T the type to check for if its vector or not.
 */
template <class T> struct is_vector { static constexpr bool value = false; };
template <class T, class A> struct is_vector<vector<T, A>> {
  static constexpr bool value = true;
};

/**
 * @brief static constexpr `value` is resolved to true if template type is a
 * Boost.YAP tensor_expression.
 *
 * @tparam T the type to check for if its tensor_expression or not.
 */
template <class T> struct is_tensor_expression {
  static constexpr bool value = false;
};
template <boost::yap::expr_kind K, typename Tuple>
struct is_tensor_expression<detail::tensor_expression<K, Tuple>> {
  static constexpr bool value = true;
};

/**
 * @brief static constexpr `value` is resolved to true if template type is a
 * ublas matrix_expression.
 *
 * @tparam T the type to check for if its matrix_expression or not.
 */
template <class T> struct is_matrix_expression {
  static constexpr bool value = std::is_base_of_v<matrix_expression<T>, T>;
};

/**
 * @brief static constexpr `value` is resolved to true if template type is a
 * Boost.YAP vector_expression.
 *
 * @tparam T the type to check for if its vector_expression or not.
 */
template <class T> struct is_vector_expression {
  static constexpr bool value = std::is_base_of_v<vector_expression<T>, T>;
};

/**
 * @brief true is T type is a tensor
 *
 * @tparam T the type to test
 */
template <class T> constexpr bool is_tensor_v = is_tensor<T>::value;

/**
 * @brief true is T type is a matrix
 *
 * @tparam T the type to test
 */
template <class T> constexpr bool is_matrix_v = is_matrix<T>::value;

/**
 * @brief true is T type is a vector
 *
 * @tparam T the type to test
 */
template <class T> constexpr bool is_vector_v = is_vector<T>::value;

/**
 * @brief true is T type is a tensor_expression
 *
 * @tparam T the type to test
 */
template <class T>
constexpr bool is_tensor_expression_v = is_tensor_expression<T>::value;

/**
 * @brief true is T type is a matrix_expression
 *
 * @tparam T the type to test
 */
template <class T>
constexpr bool is_matrix_expression_v = is_matrix_expression<T>::value;

/**
 * @brief true is T type is a vector_expression
 *
 * @tparam T the type to test
 */
template <class T>
constexpr bool is_vector_expression_v = is_vector_expression<T>::value;

} // namespace boost::numeric::ublas

#endif // BOOST_UBLAS_TENSOR_TYPE_TRAITS_HPP
