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

#include <boost/numeric/ublas/detail/config.hpp>
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
struct basic_static_extents{

  static constexpr auto Rank = sizeof...(E);
  
  using base_type       = std::array<ExtentsType,Rank>;
	using value_type      = typename base_type::value_type;
	using const_reference = typename base_type::const_reference;
	using reference       = typename base_type::reference;
	using const_pointer   = typename base_type::const_pointer;
	using const_iterator  = typename base_type::const_iterator;
	using size_type       = typename base_type::size_type;

  static_assert( std::numeric_limits<value_type>::is_integer, "Static error in basic_static_extents: type must be of type integer.");
	static_assert(!std::numeric_limits<value_type>::is_signed,  "Static error in basic_static_extents: type must be of type unsigned integer.");

  //@returns the rank of basic_static_extents
  [[nodiscard]] BOOST_UBLAS_INLINE 
  static constexpr size_type size() noexcept { return Rank; }
  
  //@returns the rank of basic_static_extents
  [[nodiscard]] BOOST_UBLAS_INLINE 
  constexpr size_type rank() noexcept { return Rank; }

  /**
   * @param k pos of extent
   * @returns the element at given pos
   */
  [[nodiscard]] BOOST_UBLAS_INLINE
  static constexpr value_type const& at(size_type k){
    return m_data.at(k); 
  }

  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr value_type const& operator[](size_type k) const noexcept{ 
    return m_data[k]; 
  }

  // default constructor
  constexpr basic_static_extents() = default;
 
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
    std::copy(begin(), end(), temp.begin());
    return temp;
  }

  /** @brief Returns ref to the std::array containing extents */
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto const& base() const noexcept{
    return m_data;
  }

  /** @brief Returns pointer to the std::array containing extents */
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr const_pointer data() const noexcept{
    return m_data.data();
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
  constexpr auto empty() const noexcept { return m_data.empty(); }

  constexpr value_type back() const noexcept{
    return m_data.back();
  }

  /** @brief Returns true if both extents are equal else false */
  template <size_t... RE>
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto operator==(basic_static_extents<ExtentsType, RE...> const &rhs) const {
    if constexpr( Rank != basic_static_extents<ExtentsType, RE...>::Rank ){
      return false;
    }else{
      for(auto i = size_type(0); i < Rank; ++i){
        if( m_data[i] != rhs[i] ){
          return false;
        }
      }
      return true;
    }
  }

  /** @brief Returns false if both extents are equal else true */
  template <size_t... RE>
  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr auto operator!=(basic_static_extents<ExtentsType, RE...> const &rhs) const {
    return !(*this == rhs);
  }

  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr const_iterator begin() const noexcept{
    return m_data.begin();
  }

  [[nodiscard]] BOOST_UBLAS_INLINE
  constexpr const_iterator end() const noexcept{
    return m_data.end();
  }

  ~basic_static_extents() = default;

// private:
  template<typename T, size_t...>
  friend struct basic_static_extents;

// private:
  static constexpr base_type const m_data{E...};
};

template<size_t... E>
using static_extents = basic_static_extents<size_t,E...>;

} // namespace boost::numeric::ublas

#endif
