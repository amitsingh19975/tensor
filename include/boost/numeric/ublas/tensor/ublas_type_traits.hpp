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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_HPP

#include <boost/yap/yap.hpp>
#include <type_traits>

namespace boost::numeric::ublas {

// Forward declare classes
template <class T, class F, class A> class tensor;
template <class T, class F, class A> class matrix;
template <class T, class A> class vector;
template <class derived_type> class matrix_expression;
template <class derived_type> class vector_expression;
namespace detail {
template <boost::yap::expr_kind K, typename Tuple> struct tensor_expression;
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
  static constexpr bool value = std::is_base_of<matrix_expression<T>, T>::value;
};

/**
 * @brief static constexpr `value` is resolved to true if template type is a
 * Boost.YAP vector_expression.
 *
 * @tparam T the type to check for if its vector_expression or not.
 */
template <class T> struct is_vector_expression {
  static constexpr bool value = std::is_base_of<vector_expression<T>, T>::value;
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

/**
 * @brief static constexpr `value` is resolved to true if template type is a
 * either a ublas::matrix, ublas::vector or ublas::tensor.
 *
 * @tparam T the type to check
 */
template <class T> struct is_ublas_type {
  constexpr static bool value =
      is_tensor_v<T> || is_matrix_v<T> || is_vector_v<T>;
};

/**
 * @brief static constexpr `value` is resolved to true if template type is a
 * either a ublas::matrix_expression, ublas::vector_expression or
 * ublas::tensor_expression.
 *
 * @tparam T the type to check
 */
template <class T> struct is_ublas_expression {
  constexpr static bool value = is_tensor_expression_v<T> ||
                                is_matrix_expression_v<T> ||
                                is_vector_expression_v<T>;
};

/**
 * @brief true is T type is a any ublas type
 *
 * @tparam T the type to test
 */
template <class T> constexpr bool is_ublas_type_v = is_ublas_type<T>::value;

/**
 * @brief true is T type is a ublas expression
 *
 * @tparam T the type to test
 */
template <class T>
constexpr bool is_ublas_expression_v = is_ublas_expression<T>::value;

} // namespace boost::numeric::ublas

#endif // BOOST_UBLAS_TENSOR_TYPE_TRAITS_HPP
