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

#ifndef BOOST_UBLAS_TENSOR_META_FUNCTIONS_HPP
#define BOOST_UBLAS_TENSOR_META_FUNCTIONS_HPP

#include <boost/numeric/ublas/tensor/fwd.hpp>
#include <array>
#include <type_traits>
#include <vector>

namespace boost::numeric::ublas::detail {

template <class E>
struct is_extents_impl : std::integral_constant<bool, false> {};

template <class T, size_t... E>
struct is_extents_impl<basic_static_extents<T, E...>> : std::true_type {};

template <class T, size_t R>
struct is_extents_impl<basic_fixed_rank_extents<T, R>> : std::true_type {};

template <class T> 
struct is_extents_impl<basic_extents<T>> : std::true_type {};

template <class E> struct is_extents {
  static constexpr bool value =
      is_extents_impl<typename std::decay<E>::type>::value;
};

template <class E>
struct is_stride_impl : std::integral_constant<bool, false> {};

template <class ExtentsType, class Layout>
struct is_stride_impl< static_strides<ExtentsType,Layout> > : std::true_type {};

template <class T, class Layout> 
struct is_stride_impl< basic_strides<T,Layout> > : std::true_type {};

template <class T, size_t N, class Layout> 
struct is_stride_impl< basic_fixed_rank_strides<T, N, Layout> > : std::true_type {};

template <class E> struct is_strides {
  static constexpr bool value =
      is_stride_impl<typename std::decay<E>::type>::value;
};

template <class E>
struct is_static_extents_impl : std::integral_constant<bool, false> {};

template <class T, size_t... E>
struct is_static_extents_impl<basic_static_extents<T, E...>>
    : std::integral_constant<bool, true> {};

template <class E> struct is_static_extents {
  static constexpr bool value =
      is_static_extents_impl<typename std::decay<E>::type>::value;
};

/** @brief type trait for checks if basic_extents_impl or not
 *
 * @tparam E of any type
 *
 **/
template <class E>
struct is_basic_extents_impl_impl : std::integral_constant<bool, false> {};

/** @brief is_extents_impl specialization
 *
 * @tparam R of size_t type
 * @tparam S of basic_shape type
 *
 **/
template <size_t R, size_t... E>
struct is_basic_extents_impl_impl<basic_extents_impl<R, E...>>
    : std::integral_constant<bool, true> {};

template <class E> struct is_basic_extents_impl {
  static constexpr bool value =
      is_basic_extents_impl_impl<typename std::decay<E>::type>::value;
};

/** @brief Checks if the extents is dynamic
 *
 * @tparam E of type basic_extents or basic_static_extents
 *
 */
template <class E> struct is_dynamic : std::integral_constant<bool, false> {};

/** @brief Partial Specialization of is_dynamic_extents with basic_extens
 *
 * @tparam T of any integer type
 *
 */
template <class T>
struct is_dynamic<basic_extents<T>> : std::integral_constant<bool, true> {};

template <class T, size_t R>
struct is_dynamic<basic_fixed_rank_extents<T,R>> : std::integral_constant<bool, true> {};

/** @brief Checks if the extents has dynamic rank
 *
 * @tparam E of type basic_extents or basic_static_extents
 *
 */
template <class E> 
struct is_dynamic_rank : std::integral_constant<bool, true> {};

template <class T, size_t... E>
struct is_dynamic_rank<basic_static_extents<T, E...>> : std::integral_constant<bool, false> {};

template <class T, size_t R>
struct is_dynamic_rank<basic_fixed_rank_extents<T,R>> : std::integral_constant<bool, false> {};


/** @brief Checks if the extents has static rank
 *
 * @tparam E of type basic_extents or basic_static_extents
 *
 */
template <class E> 
struct is_static_rank : std::integral_constant<bool, false> {};

template <class T>
struct is_static_rank<basic_extents<T>> : std::integral_constant<bool, false> {};

template <class T, size_t... E>
struct is_static_rank<basic_static_extents<T, E...>> : std::integral_constant<bool, true> {};
template <class T, size_t R>

struct is_static_rank<basic_fixed_rank_extents<T,R>> : std::integral_constant<bool, true> {};


/** @brief Checks if the extents is static
 *
 * @tparam E of type basic_extents or basic_static_extents
 *
 */
template <class E> struct is_static {
  static constexpr bool value = is_static_extents<E>::value;
};

template <> struct product_helper_impl<> {
  static constexpr size_t value = 1;
};

template <size_t E, size_t... R> struct product_helper_impl<E, R...> {
  static constexpr size_t value = E * product_helper_impl<R...>::value;
};

/** @brief removes the const and refernece
 *
 * @tparam T any type
 *
 **/
template <class T> struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

// empty struct for tagging valid iterator
struct iterator_tag {};

// empty struct for tagging invalid iterator
struct invalid_iterator_tag {};

/** @brief checks and gives back the appropriate tag
 *
 * @tparam I type of iterator
 *
 **/
template <class I>
using iterator_tag_t = std::conditional_t<
    std::is_same<typename std::iterator_traits<I>::iterator_category,
                 std::output_iterator_tag>::value,
    invalid_iterator_tag,
    std::conditional_t<
        std::numeric_limits<typename remove_cvref<
            typename std::iterator_traits<I>::reference>::type>::is_integer,
        iterator_tag, invalid_iterator_tag>>;

/** @brief checks if given type is iterator or not
 *
 * @tparam T any type
 *
 **/
template <class T, class = void> struct is_iterator {
  static constexpr bool value = false;
};

/** @brief Partial specialization for is_iterator
 *
 * @tparam T any type
 *
 **/
template <class T>
struct is_iterator<
    T, typename std::enable_if_t<!std::is_same<
           typename std::iterator_traits<T>::value_type, void>::value>> {
  static constexpr bool value = true;
};

template <typename T> struct is_stl_array : std::false_type {};
template <typename T, std::size_t N>
struct is_stl_array<std::array<T, N>> : std::true_type {};

} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas {

template <typename T, size_t... E>
struct detail::product_helper<basic_static_extents<T,E...>> {
  static constexpr T value = detail::product_helper_impl<E...>::value;
};
template <typename V, typename E, typename F, typename A>
struct tensor_mode_result {
  using type = tensor<V, E, F, A >;
};

template <typename V, typename E, typename F, typename A>
using tensor_mode_result_t = typename tensor_mode_result<V, E, F, A>::type;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas::storage::detail
{

template <typename T>
struct is_tensor_storage
{
  static constexpr bool value = std::is_base_of<tensor_storage, T>::value;
};

template <typename T>
struct is_sparse_storage
{
  static constexpr bool value = std::is_base_of<sparse_storage, T>::value;
};

template <typename T>
struct is_band_storage
{
  static constexpr bool value = std::is_base_of<band_storage, T>::value;
};

template <typename T>
struct is_dense_storage
{
  static constexpr bool value = std::is_base_of<dense_storage, T>::value;
};

} // namespace boost::numeric::ublas::storage::detail

namespace boost::numeric::ublas::detail{
  
  template<typename T>
  class has_resize_member{                                                    
    using yes_type = char;
    using no_type = long;
    template <typename U> static yes_type test(decltype(&U::resize));
    template <typename U> static no_type  test(...);
  public:                                                                    
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes_type);
  };

  template<typename T>
  class has_assign_member{                                                    
    using yes_type = char;
    using no_type = long;
    template <typename U> static yes_type test(decltype(&U::assign));
    template <typename U> static no_type  test(...);
  public:                                                                    
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes_type);
  };

} // namespace boost::numeric::ublas::detail


#endif