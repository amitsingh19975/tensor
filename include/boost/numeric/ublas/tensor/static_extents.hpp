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

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_STATIC_EXTENTS_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_STATIC_EXTENTS_HPP_

// #include <boost/numeric/ublas/tensor/fwd.hpp>
#include <boost/numeric/ublas/tensor/detail/extents_helper.hpp>
#include <array>
#include <initializer_list>
#include <vector>

namespace boost::numeric::ublas {

template <class ExtentsType, size_t... E> struct basic_static_extents;
template <class ExtentsType> class basic_extents;

/** @brief Template class for storing tensor extents for compile time.
 *
 * @code basic_static_extents<1,2,3,4> t @endcode
 * @tparam E parameter pack of extents
 *
 */
template <class ExtentsType, size_t... E>
struct basic_static_extents
    : detail::basic_extents_impl<0, E...> {

  static constexpr auto Rank = sizeof...(E);
  
  using parent_type     = detail::basic_extents_impl<0, E...>;
  using base_type       = std::array<ExtentsType,Rank>;
	using value_type      = typename base_type::value_type;
	using const_reference = typename base_type::const_reference;
	using reference       = typename base_type::reference;
	using const_pointer   = typename base_type::const_pointer;
	using const_iterator  = typename base_type::const_iterator;
	using size_type       = typename base_type::size_type;

  //@returns the rank of basic_static_extents
  [[nodiscard]] BOOST_UBLAS_INLINE 
  static constexpr auto size() noexcept { return parent_type::Rank; }
  
  //@returns the rank of basic_static_extents
  [[nodiscard]] BOOST_UBLAS_INLINE 
  static constexpr auto rank() noexcept { return parent_type::Rank; }

  /**
   * @param k pos of extent
   * @returns the element at given pos
   */
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr value_type at(size_type k) const{ 
    if ( k >= Rank ){
      throw std::out_of_range("boost::numeric::ublas::basic_static_extents::at: out of bound");
    }
    return parent_type::at(k); 
  }

  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr value_type operator[](size_type k) const noexcept{ 
    return parent_type::at(k); 
  }

  // default constructor
  constexpr basic_static_extents() = default;
  
  constexpr basic_static_extents( detail::basic_extents_impl<0, E...> )
    : parent_type()
  {}
 
  // default copy constructor
  constexpr basic_static_extents(basic_static_extents const&) = default;
  constexpr basic_static_extents& 
  operator=(basic_static_extents const&) = default;
 
  // default assign constructor
  constexpr basic_static_extents(basic_static_extents&&) = default;
  constexpr basic_static_extents& 
  operator=(basic_static_extents&&) = default;

  /** @brief Returns the std::vector containing extents */
  [[nodiscard]] BOOST_UBLAS_INLINE
  auto to_vector() const {
    std::vector<value_type> temp(Rank);
    for (auto i = size_type(0); i < Rank; i++) {
      temp[i] = parent_type::at(i);
    }
    return temp;
  }

  /** @brief Returns the std::vector containing extents */
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto base() const {
    return this->to_array();
  }

  /** @brief Returns the std::array containing extents */
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto to_array() const {
    base_type temp;
    for (auto i = size_type(0); i < Rank; i++) {
      temp[i] = parent_type::at(i);
    }
    return temp;
  }

  /** @brief Returns the basic_extents containing extents */
  [[nodiscard]] BOOST_UBLAS_INLINE
  auto to_dynamic_extents() const {
    return basic_extents<value_type>(this->to_vector());
  }

  /** @brief Checks if extents is empty or not
   *
   * @returns true if rank is 0 else false
   *
   */
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto empty() const noexcept { return Rank == size_type{0}; }

  constexpr value_type back() const noexcept{
    return at( Rank - 1 );
  }

  /** @brief Returns true if both extents are equal else false */
  template <size_t... rhs>
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto operator==(basic_static_extents<ExtentsType, rhs...> const &other) const {
    if (Rank != other.size()) {
      return false;
    }
    for (auto i = 0u; i < Rank; i++) {
      if (other.at(i) != parent_type::at(i))
        return false;
    }
    return true;
  }

  /** @brief Returns false if both extents are equal else true */
  template <size_t... rhs>
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto operator!=(basic_static_extents<ExtentsType, rhs...> const &other) const {
    return !(*this == other);
  }

  ~basic_static_extents() = default;
};

template<size_t... E>
using static_extents = basic_static_extents<size_t,E...>;

template<size_t... E>
basic_static_extents(detail::basic_extents_impl<0, E...>) -> basic_static_extents<size_t,E...>;

} // namespace boost::numeric::ublas

#endif
