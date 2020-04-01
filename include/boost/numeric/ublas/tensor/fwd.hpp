//
//  Copyright (c) 2018-2019, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_FWD_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_FWD_HPP_

#include <cstddef>
#include <string>

namespace boost::numeric::ublas {

template <class T, T... E> struct basic_static_extents;
template <class T, size_t R> struct basic_fixed_rank_extents;

template <class T> class basic_extents;

template <class E> constexpr bool valid(E const &e);

template <class E> constexpr bool is_scalar(E const &e);

template <class E> constexpr bool is_vector(E const &e);

template <class E> constexpr bool is_matrix(E const &e);

template <class E> constexpr bool is_tensor(E const &e);

template <class E> auto squeeze(E const &e);

template <class E> constexpr auto product(E const &e);

template <class E> std::string to_string(E const &e);

template <class T, class L> class basic_strides;

template<class T, size_t N, class L> class basic_fixed_rank_strides;

/** @brief Forward declaration of basic_static_strides for specialization
 *
 * @code basic_static_strides<basic_static_extents<4,1,2,3,4>, first_order> s @endcode
 * @code basic_static_strides<basic_static_extents<4,1,2,3,4>, last_order> s @endcode
 *
 * @tparam ExtentType type of basic_static_extents
 * @tparam Layout either first_order or last_order
 *
 */
template <class ExtentType, class Layout> struct basic_static_strides;

/** @brief Type trait for selecting basic_static_strides or basic_stride based on the
 * type of extents
 *
 * @tparam E type of basic_extents or basic_static_extents
 *
 * @tparam Layout either first_order or last_order
 *
 */

template <class E, class Layout> struct stride_type;

template <class T, class E, class F, class A> class tensor;

template <class T, class F, class A> class matrix;

template <class T, class A> class vector;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas::detail {

/** @brief stores the extents
 *
 * tparam R of type size_t which stands for Rank
 * tparam S of type basic_shape
 *
 */

template <class E> struct is_extents;

template <class E> struct is_static_rank;

template <class E> struct is_dynamic;

template <class E> struct is_dynamic_rank;

template <class E> struct is_static;

} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas::storage {
namespace dense_tensor {

    template <typename T, typename E, typename A, typename = void>
    struct default_storage;

}

} // namespace boost::numeric::ublas::storage

#endif
