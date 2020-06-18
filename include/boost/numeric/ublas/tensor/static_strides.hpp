//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//
/// \file strides.hpp Definition for the basic_strides template class

#ifndef BOOST_UBLAS_TENSOR_STATIC_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_STATIC_STRIDES_HPP

#include <boost/numeric/ublas/tensor/static_extents.hpp>
#include <boost/numeric/ublas/tensor/traits/static_extents_traits.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/numeric/ublas/tensor/detail/utility.hpp>

namespace boost::numeric::ublas{
  
  template <typename E, typename L> struct basic_static_strides;

} // boost::numeric::ublas

namespace boost::numeric::ublas::detail{

  // @returns the static_stride_list containing strides
  // It is a helper function or implementation
  template<typename U, typename... T>
  constexpr auto make_static_strides_helper( boost::mp11::mp_list<T...>, last_order ) noexcept
  {
    constexpr auto const size = sizeof...(T);

    using extents_list = boost::mp11::mp_reverse< boost::mp11::mp_list<T...> >;
    using one_type = boost::mp11::mp_list_c<U,1ul>;
    using final_list = boost::mp11::mp_repeat_c<one_type, size >;


    auto ret = type_static_for<0ul,size>([&](auto iter, auto pres){
      using iter_type = decltype(iter);

      using ext_list = std::decay_t< decltype(std::get<0>(pres)) >;
      using f_list = std::decay_t< decltype(std::get<1>(pres)) >;

      using ext_at = boost::mp11::mp_at_c< ext_list, iter_type::value >;
      using f_at = boost::mp11::mp_at_c< f_list, iter_type::value >;
      using new_list = boost::mp11::mp_replace_at_c< f_list, iter_type::value + 1, std::integral_constant<U, ext_at::value * f_at::value > >;

      if constexpr( iter_type::value + 1 >= size ){
        return new_list{};
      }else{
        return std::make_pair(ext_list{},new_list{});
      }

    }, std::make_pair(extents_list{},final_list{}));
    
    return boost::mp11::mp_reverse< decltype(ret) >{};

  }

  // @returns the static_stride_list containing strides
  // It is a helper function or implementation
  template<typename U, typename... T>
  constexpr auto make_static_strides_helper( boost::mp11::mp_list<T...>, first_order ) noexcept
  {
    constexpr auto const size = sizeof...(T);

    using extents_list = boost::mp11::mp_list<T...>;
    using one_type = boost::mp11::mp_list_c<U,1ul>;
    using final_list = boost::mp11::mp_repeat_c<one_type, size >;


    return type_static_for<1ul,size>([&](auto iter, auto pres){
      using iter_type = decltype(iter);
      
      using ext_list = std::decay_t< decltype(std::get<0>(pres)) >;
      using f_list = std::decay_t< decltype(std::get<1>(pres)) >;

      using ext_at = boost::mp11::mp_at_c< ext_list, iter_type::value - 1 >;
      using f_at = boost::mp11::mp_at_c< f_list, iter_type::value - 1 >;
      using new_list = boost::mp11::mp_replace_at< f_list, iter_type, std::integral_constant<U, ext_at::value * f_at::value > >;

      if constexpr( iter_type::value + 1 >= size ){
        return new_list{};
      }else{
        return std::make_pair(ext_list{},new_list{});
      }

    }, std::make_pair(extents_list{},final_list{}));
    

  }


  // @returns the static_stride_list containing strides for last order
  template<typename L, typename T, T... E>
  constexpr auto make_static_strides( basic_static_extents<T,E...> ) noexcept
  {
    using extents_type = basic_static_extents<T,E...>;
    // checks if extents are vector or scalar
    if constexpr( !( static_traits::is_scalar_v<extents_type> || static_traits::is_vector_v<extents_type> ) ){
      // if extent contains only one element return static_stride_list<T,T(1)>
      using ret_type = std::decay_t< 
        decltype(
            make_static_strides_helper<T>(boost::mp11::mp_list_c<T,E...>{},L{})
          ) 
      >;
      return seq_to_array_v< ret_type >;
    }else{
      // @returns list contining ones if it is vector or scalar
      using ret_type = boost::mp11::mp_repeat_c<boost::mp11::mp_list_c<T,1ul>, sizeof...(E) >;
      return seq_to_array_v< ret_type >;
    }
  }
  

} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas
{
/** @brief Partial Specialization for first_order or column_major
 *
 * @code basic_static_strides<basic_static_extents<4,1,2,3,4>, first_order> s @endcode
 *
 * @tparam R rank of basic_static_extents
 * @tparam Extents paramerter pack of extents
 *
 */
template <typename Layout, typename T, T... Extents>
struct basic_static_strides<basic_static_extents<T,Extents...>, Layout>
{

  static constexpr std::size_t const _size = sizeof...(Extents);

  using layout_type     = Layout;
  using extents_type    = basic_static_extents<T,Extents...>;
  using base_type       = std::array<T, _size>;
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
  constexpr const_reference at(size_type k) const 
  {
    return m_data.at(k);
  }

  [[nodiscard]] inline 
  constexpr const_reference operator[](size_type k) const { return m_data[k]; }

  //@returns the rank of basic_static_extents
  [[nodiscard]] inline 
  constexpr size_type size() const noexcept { return static_cast<size_type>(_size); }

  [[nodiscard]] inline
  constexpr const_reference back () const noexcept{
      return m_data.back();
  }

  // default constructor
  constexpr basic_static_strides() noexcept{
    static_assert( 
      _size == 0 || 
      ( static_traits::is_valid_v<extents_type> &&
        ( static_traits::is_vector_v<extents_type> ||
          static_traits::is_scalar_v<extents_type> ||
          _size >= 2 
        )
      )
      , 
      "Error in boost::numeric::ublas::basic_static_strides() : "
      "Size can be 0 or Shape should be valid and shape can be vector or shape can be scalar or size should be greater than"
      " or equal to 2"
    ); 	
    
    
  }

  template<typename ExtentsType>
  constexpr basic_static_strides(ExtentsType const&) noexcept{
      static_assert( is_extents_v<ExtentsType>, "boost::numeric::ublas::basic_fixed_rank_extents(ExtentsType const&) : " 
          "ExtentsType is not a tensor extents"
      );
  };


  constexpr basic_static_strides(basic_static_strides const &) noexcept = default;
  constexpr basic_static_strides(basic_static_strides &&) noexcept = default;

  constexpr basic_static_strides &
  operator=(basic_static_strides const &) noexcept = default;
  
  constexpr basic_static_strides &
  operator=(basic_static_strides &&) noexcept = default;

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

private:
  static constexpr base_type const m_data{ detail::make_static_strides<layout_type>(extents_type{}) };
};

} // namespace boost::numeric::ublas

#endif
