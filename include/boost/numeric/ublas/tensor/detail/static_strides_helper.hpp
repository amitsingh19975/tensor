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

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_STATIC_STRIDE_HELPER_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_STATIC_STRIDE_HELPER_HPP_

#include <boost/numeric/ublas/functional.hpp>
#include <boost/numeric/ublas/tensor/detail/static_traits.hpp>
#include <algorithm>
#include <type_traits>
#include <utility>
#include <array>


namespace boost::numeric::ublas{

  using first_order = column_major;
  using last_order = row_major;
  
  template <class E, class L> struct basic_static_strides;

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas::detail{

  // list for storing stides as types
  template< typename T, T... P > 
  struct static_stride_list{
      using extents_type = basic_static_extents<T, P...>;
      using type = std::array<T, sizeof...(P)>;
      static constexpr type const value = {P...};
  };

  namespace impl{
 
    // concat two static_stride_list togather
    // @code using type = typename concat< static_stride_list<int, 1,2,3>, static_stride_list<int, 4,5,6> >::type @endcode
    template<typename L1, typename L2>
    struct concat;

    template<typename T, T... N1, T... N2>
    struct concat< static_stride_list<T, N1...>, static_stride_list<T, N2...> > {
      using type = static_stride_list<T, N1..., N2...>;
    };

    template<typename L1, typename L2>
    using concat_t = typename concat<L1,L2>::type;

    // generates static_stride_list containing ones with specific size
    template<typename T, size_t N> 
    struct make_sequence_of_ones;

    template<typename T, size_t N> 
    using make_sequence_of_ones_t = typename make_sequence_of_ones<T, N>::type;

    template<typename T, size_t N>
    struct make_sequence_of_ones {
      using type = concat_t<make_sequence_of_ones_t<T, N/2>, make_sequence_of_ones_t<T, N - N/2>>;
    };

    template<typename T> 
    struct make_sequence_of_ones<T, 0ul> {
      using type = static_stride_list<T>;
    };
    template<typename T> 
    struct make_sequence_of_ones<T, 1ul>{ 
      using type = static_stride_list<T, T(1)>;
    };

  } // impl


  template<typename T, T N> 
  using make_sequence_of_ones_t = impl::make_sequence_of_ones_t<T,N>;

  // @returns the static_stride_list containing strides for first order
  // It is a helper function or implementation
  template<typename T, T E0, T... E, T... R, T... P>
  constexpr auto make_first_order_strides_helper( static_stride_list<T, E0, E...>, 
    static_stride_list<T,R...>, static_stride_list<T,P...>)
  {
    if constexpr( sizeof...(E) == 0 ){
      return static_stride_list<T, T(1), P...>{};
    }else{
      // add extent to the list which will be used for
      // take product for next iteration
      auto n = static_stride_list<T, R..., E0>{};

      // result list containing the strides
      // on each iteration calculate the product
      auto np = static_stride_list< T, P..., static_traits::product_v< basic_static_extents<T, R..., E0> > >{};
      return make_first_order_strides_helper( static_stride_list<T, E...>{}, n, np );
    }
  }


  // @returns the static_stride_list containing strides for first order
  template<typename T, T... E>
  constexpr auto make_first_order_strides( static_stride_list<T, E...> )
  {
    using extents_type = typename static_stride_list<T, E...>::extents_type;
    // checks if extents are vector or scalar
    if constexpr( !( static_traits::is_scalar_v<extents_type> || static_traits::is_vector_v<extents_type> )){
      // if extents are empty return empty list
      if constexpr ( sizeof...(E) == 0 ){
        return static_stride_list<T>{};
      }else{
        return make_first_order_strides_helper(static_stride_list<T, E...>{}, static_stride_list<T>{}, static_stride_list<T>{});
      }
    }else{
      // @returns list contining ones if it is vector or scalar
      return make_sequence_of_ones_t<T, sizeof...(E)>{};
    }
  }

    // @returns the static_stride_list containing strides for last order
  // It is a helper function or implementation
  template<typename T, T E0, T... E, T... R, T... P>
  constexpr auto make_last_order_strides_helper( static_stride_list<T, E0, E...>, 
    static_stride_list<T, R...>, static_stride_list<T, P...>)
  {
    if constexpr( sizeof...(E) == 0 ){
      return static_stride_list<T, P..., E0, T(1)>{};
    }else{
      // add extent to the list which will be used for
      // take product for next iteration
      auto n = static_stride_list<T, R...,E0>{};

      // result list containing the strides
      // on each iteration calculate the product
      auto np = static_stride_list< T, P..., static_traits::product_v< basic_static_extents<T, E..., E0> > >{};
      return make_last_order_strides_helper( static_stride_list<T, E...>{}, n, np );
    }
  }

  // @returns the static_stride_list containing strides for last order
  template<typename T, T E0, T... E>
  constexpr auto make_last_order_strides( static_stride_list<T, E0, E...> )
  {
    using extents_type = typename static_stride_list<T, E0, E...>::extents_type;
    // checks if extents are vector or scalar
    if constexpr( !( static_traits::is_scalar_v<extents_type> || static_traits::is_vector_v<extents_type> ) ){
      // if extent contains only one element return static_stride_list<T,T(1)>
      if constexpr( sizeof...(E) == 0 ){
        return static_stride_list<T,T(1)>{};
      }else{
        return make_last_order_strides_helper(static_stride_list<T, E...>{}, static_stride_list<T>{}, static_stride_list<T>{});
      }
    }else{
      // @returns list contining ones if it is vector or scalar
      return make_sequence_of_ones_t<T, sizeof...(E) + 1>{};
    }
  }
  
  // if extents are empty return empty list
  template<typename T>
  constexpr auto make_last_order_strides( static_stride_list<T> )
  {
    return static_stride_list<T>{};
  }

  template<typename Layout, typename T, T... E>
  struct strides_helper;

  // It is use for first order to
  // get std::array containing strides
  template<typename T, T... E>
  struct strides_helper<first_order,T, E...>{
    using type = decltype( make_first_order_strides(static_stride_list<T, E...>{}) );
    static constexpr auto value = type::value;
  };
  
  // It is use for last order to
  // get std::array containing strides
  template<typename T, T... E>
  struct strides_helper<last_order, T, E...>{
    using type = decltype( make_last_order_strides(static_stride_list<T, E...>{}) );
    static constexpr auto value = type::value;
  };

  template<typename Layout, typename T, T... E>
  inline static constexpr auto strides_helper_v = strides_helper<Layout, T, E...>::value;

} // namespace boost::numeric::ublas::detail
#endif
