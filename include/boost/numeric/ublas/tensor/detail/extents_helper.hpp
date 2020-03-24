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

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_HELPER_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_HELPER_HPP_

#include <boost/numeric/ublas/detail/config.hpp>

namespace boost::numeric::ublas::detail
{

template <size_t R, size_t...> struct basic_extents_impl;

/** @brief basic_extents_impl specialization for basic_shape
 *
 * tparam R of type size_t which stands for Rank
 *
 */
template <size_t R>
struct basic_extents_impl<R>
{
  // aliases the basic_extents_impl
  using next = basic_extents_impl;

  // stores the rank
  static constexpr size_t Rank = 0;

  // stores the both static
  static constexpr size_t N = 1;

  // used for stride
  size_t Step{1};

  /**
   * @returns extent at a given index
   **/
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto at(int) const noexcept { return size_t{1}; }

  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto &step(int) noexcept { return this->Step; }
  
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto &step(int) const noexcept { return this->Step; }
  /**
   * @returns extent at a given index
   **/
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto operator[](int) const noexcept { return at(0); }
  /**
   * @returns product of extents
   **/
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto product(int) const noexcept { return size_t{1}; }
  /**
   * @returns product of extents
   **/
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto product() const noexcept { return size_t{1}; }

  //@returns true if empty otherwise false
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto empty() const noexcept { return true; }

  /**
   * @returns Rank of the extents
   **/
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto size() const noexcept { return 0u; }

  // default constructor
  constexpr basic_extents_impl() noexcept = default;
  // copy constructor
  constexpr basic_extents_impl(basic_extents_impl const &other) noexcept = default;
  // copy assignment operator
  constexpr basic_extents_impl &
  operator=(basic_extents_impl const &other) noexcept = default;

  ~basic_extents_impl() = default;
};

template <size_t R, size_t SE, size_t... E>
struct basic_extents_impl<R, SE, E...>
    : basic_extents_impl<R + 1, E...>
{

  static_assert(SE != 0, "boost::numeric::ublas::detail::basic_extents_impl : extent can not be 0");

  using next = basic_extents_impl<R + 1, E...>;  

  // stores the rank
  static constexpr size_t Rank = 1 + next::Rank;

  // stores the static extent
  static constexpr size_t N = SE;

  // used for stride
  size_t Step{1};

  /**
   * @param k index of extent
   * @returns extent at given index
   **/
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto at(int k) const noexcept { return k == R ? N : next::at(k); }

  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto &step(int k) noexcept
  {
    return ( k == R ? Step : next::step(k) );
  }

  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto &step(int k) const noexcept
  {
    return ( k == R ? Step : next::step(k) );
  }

  /**
   * @param k index of extent
   * @returns extent at a given index
   **/
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto operator[](int k) const noexcept { return at(k); }

  /**
   * @param k index of extent
   * @returns product of extents
   **/
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto product(int k) const noexcept
  {
    return k == R ? N * next::product() : next::product(k);
  }

  /**
   * @returns product of extents
   **/
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto product() const noexcept { return N * next::product(); }

  /**
   * @returns Rank of the extents
   **/
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto size() const noexcept { return size_t(Rank); }

  //@returns true if empty otherwise false
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto empty() const noexcept { return false; }

  // default constructor
  constexpr basic_extents_impl() noexcept : next() {}
  // copy constructor
  constexpr basic_extents_impl(basic_extents_impl const &other) noexcept = default;
  // copy assignment operator
  constexpr basic_extents_impl &
  operator=(basic_extents_impl const &other) noexcept = default;

  ~basic_extents_impl() = default;
};

// /**
//    * @tparam IndexType type of index
//    * @tparam Args parameter pack of indices with different types
//    * @param idx index of extent
//    * @param args parameter pack of indices
//    * @returns true if in bound or false if not
//    **/
// template <size_t depth, class E, class IndexType, class... Args>
// constexpr bool in_bounds(E const &e, IndexType const &idx, Args... args)
// {
//   if constexpr (sizeof...(args) == 0)
//   {
//     return 0 <= idx && idx < e.at(depth);
//   }
//   else
//   {
//     return 0 <= idx && idx < e.at(depth) && in_bounds<depth + 1>(e, args...);
//   }
// }

// /**@returns true if nothing is passed*/
// template <class E>
// constexpr bool in_bounds(E const &e)
// {
//   return true;
// }

} // namespace boost::numeric::ublas::detail

#endif