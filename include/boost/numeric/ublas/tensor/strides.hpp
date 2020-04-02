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
/// \file strides.hpp Definition for the basic_strides template class

#ifndef _BOOST_UBLAS_TENSOR_STRIDES_HPP_
#define _BOOST_UBLAS_TENSOR_STRIDES_HPP_

#include <algorithm>
#include <boost/numeric/ublas/tensor/dynamic_strides.hpp>
#include <boost/numeric/ublas/tensor/static_strides.hpp>

namespace boost::numeric::ublas {

  template <class LStrides, class RStrides,
    std::enable_if_t<detail::is_strides<LStrides>::value && detail::is_strides<RStrides>::value, int> = 0
  >
  constexpr bool operator==(LStrides const &lhs, RStrides const &rhs) {
    
    static_assert(detail::is_strides<LStrides>::value && detail::is_strides<RStrides>::value,
      "boost::numeric::ublas::operator==() : invalid type, type should be a strides");

    if( lhs.size() != rhs.size() ){
        return false;
    }else{
        return std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }
  }

  template <class LStrides, class RStrides,
    std::enable_if_t<detail::is_strides<LStrides>::value && detail::is_strides<RStrides>::value, int> = 0
  >
  constexpr bool operator!=(LStrides const &lhs, RStrides const &rhs) {
    static_assert(detail::is_strides<LStrides>::value && detail::is_strides<RStrides>::value,
      "boost::numeric::ublas::operator!=() : invalid type, type should be a strides");
    return !(lhs == rhs);
  }

  template <class Strides,
    std::enable_if_t<detail::is_strides<Strides>::value, int> = 0
  >
  std::ostream& operator<<(std::ostream& os, Strides const& s){
    static_assert(detail::is_strides<Strides>::value,
      "boost::numeric::ublas::operator<<() : invalid type, type should be a strides");
    return os<<to_string(s);
  }



  template <class Layout, class T> struct strides;

  /** @brief Partial Specialization of strides for basic_static_extents
   *
   *
   * @tparam Layout either first_order or last_order
   *
   * @tparam R rank of extents
   *
   * @tparam Extents parameter pack of extents
   *
   */
  template <class Layout, class T, T... Extents>
  struct strides<basic_static_extents<T, Extents...>, Layout>
  {
    using type = basic_static_strides<basic_static_extents<T, Extents...>, Layout>;
  };

  /** @brief Partial Specialization of strides for basic_extents
   *
   *
   * @tparam Layout either first_order or last_order
   *
   * @tparam T extents type
   *
   */
  template <class Layout, class T>
  struct strides<basic_extents<T>, Layout>
  {
    using type = basic_strides<T, Layout>;
  };

  /** @brief Partial Specialization of strides for basic_fixed_rank_strides
   *
   *
   * @tparam Layout either first_order or last_order
   *
   * @tparam T extents type
   *
   */
  template <class Layout, std::size_t N, class T>
  struct strides<basic_fixed_rank_extents<T,N>, Layout>
  {
    using type = basic_fixed_rank_strides<T, N, Layout>;
  };

  /** @brief type alias of result of strides::type
   *
   * @tparam E extents type either basic_extents or basic_static_extents
   *
   * @tparam Layout either first_order or last_order
   *
   */
  template <class E, class Layout>
  using strides_t = typename strides<E, Layout>::type;

} // namespace boost::numeric::ublas

#endif
