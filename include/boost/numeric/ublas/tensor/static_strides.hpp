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

#ifndef BOOST_UBLAS_TENSOR_STATIC_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_STATIC_STRIDES_HPP

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <boost/numeric/ublas/tensor/detail/static_strides_helper.hpp>

namespace boost::numeric::ublas
{

using first_order = column_major;
using last_order = row_major;

template <class E, class L> struct basic_static_strides;

/** @brief Partial Specialization for first_order or column_major
 *
 * @code basic_static_strides<basic_static_extents<4,1,2,3,4>, first_order> s @endcode
 *
 * @tparam R rank of basic_static_extents
 * @tparam Extents paramerter pack of extents
 *
 */
template <class Layout, class T, T... Extents>
struct basic_static_strides<basic_static_extents<T,Extents...>, Layout>
{

  static constexpr size_t const Rank = sizeof...(Extents);

  using layout_type     = Layout;
  using extents_type    = basic_static_extents<T,Extents...>;
  using base_type       = std::array<T, Rank>;
  using value_type      = typename base_type::value_type;
  using reference       = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using size_type       = typename base_type::size_type;
  using const_pointer   = typename base_type::const_pointer;
  using const_iterator  = typename base_type::const_iterator;

  /**
   * @param k pos of extent
   * @returns the element at given pos
   */
  [[nodiscard]] inline 
  constexpr auto at(size_type k) const 
  {
    return m_data.at(k);
  }

  [[nodiscard]] inline 
  constexpr auto operator[](size_type k) const { return m_data[k]; }

  //@returns the rank of basic_static_extents
  [[nodiscard]] inline 
  constexpr auto rank() const noexcept { return Rank; }

  //@returns the rank of basic_static_extents
  [[nodiscard]] inline 
  constexpr auto size() const noexcept { return Rank; }

	value_type back () const{
		return m_data.back();
	}

  // default constructor
  constexpr basic_static_strides(){
    		if( Rank == 0 )
          return;

        static_assert( static_traits::is_valid_v<extents_type>, "Error in boost::numeric::ublas::basic_static_strides() : shape is not valid.");		

        if constexpr( static_traits::is_vector_v<extents_type> || static_traits::is_scalar_v<extents_type> ){
          return;
        }else{
          static_assert(Rank >= 2, "Error in boost::numeric::ublas::basic_static_strides() : size of strides must be greater or equal 2.");
        }

  }

  constexpr basic_static_strides(extents_type const&) {};

  // default copy constructor
  constexpr basic_static_strides(basic_static_strides const &other) noexcept = default;
  // default assign constructor
  constexpr basic_static_strides &
  operator=(basic_static_strides const &other) noexcept = default;

   /** @brief Returns ref to the std::array containing extents */
  [[nodiscard]] inline
  constexpr auto const& base() const noexcept{
    return m_data;
  }

  /** @brief Returns pointer to the std::array containing extents */
  [[nodiscard]] inline
  constexpr const_pointer data() const noexcept{
    return m_data.data();
  }

  [[nodiscard]] inline
  constexpr const_iterator begin() const noexcept{
    return m_data.begin();
  }

  [[nodiscard]] inline
  constexpr const_iterator end() const noexcept{
    return m_data.end();
  }

  [[nodiscard]] inline
  constexpr bool empty() const noexcept{
    return m_data.empty();
  }

  template<class OtherE>
  constexpr bool operator==(basic_static_strides<OtherE,layout_type> const& rhs) const noexcept{
    if constexpr( Rank != basic_static_strides<OtherE,layout_type>::Rank ){
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

  template<class OtherE>
  constexpr bool operator!=(basic_static_strides<OtherE,layout_type> const& rhs) const noexcept{
    return !(*this == rhs);
  }

private:
  static constexpr base_type const m_data{ detail::strides_helper_v<layout_type,T,Extents...> };
};


} // namespace boost::numeric::ublas

#endif
