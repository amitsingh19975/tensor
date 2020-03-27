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

#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/functional.hpp>

#include <algorithm>
#include <type_traits>
#include <utility>
#include <array>


namespace boost::numeric::ublas{

  using first_order = column_major;
  using last_order = row_major;
  
  template <class E, class L> struct static_strides;

  namespace detail{
    
    template< size_t... > struct static_stride_helper;

  } // namespace detail
  

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas::detail{

  // list for storing stides as types
  template< typename T, size_t... P > 
  struct static_stride_list{
      using type = std::array<T, sizeof...(P)>;
      static constexpr type const value = {static_cast<T>(P)...};
  };

  namespace impl{
    
    // used to get E0 * E1 * ... En
    template<size_t... E>
    struct static_stride_product;

    template<size_t E0, size_t... E>
    struct static_stride_product<E0, E...>{
      static constexpr auto const value = E0 * static_stride_product<E...>::value;
    };

    template<size_t E>
    struct static_stride_product<E>{
      static constexpr auto const value = E;
    };

    template<>
    struct static_stride_product<>{
      static constexpr auto const value = 1;
    };

    // check is all Es are ones
    template<size_t... E>
    struct is_all_ones;

    // concat two static_stride_list togather
    // @code using type = typename concat< static_stride_list<int, 1,2,3>, static_stride_list<int, 4,5,6> >::type @endcode
    template<typename L1, typename L2>
    struct concat;

    template<typename T, size_t... N1, size_t... N2>
    struct concat< static_stride_list<T, N1...>, static_stride_list<T, N2...> > {
      using type = static_stride_list<T, N1..., N2...>;
    };

    template<typename L1, typename L2>
    using concat_t = typename concat<L1,L2>::type;

    // generates static_stride_list containing ones with specific size
    template<typename T, size_t N> 
    struct gen_all_one_list;

    template<typename T, size_t N> 
    using gen_all_one_list_t = typename gen_all_one_list<T, N>::type;

    template<typename T, size_t N>
    struct gen_all_one_list {
      using type = concat_t<gen_all_one_list_t<T, N/2>, gen_all_one_list_t<T, N - N/2>>;
    };

    template<typename T> struct gen_all_one_list<T,0> {
      using type = static_stride_list<T>;
    };
    template<typename T> struct gen_all_one_list<T,1>{ 
      using type = static_stride_list<T,1>;
    };

    // check if Extents list is vector or not
    template<size_t... E>
    struct is_vector_static_stride_list;
    
    template<size_t E0, size_t E1, size_t E2, size_t... E>
    struct is_vector_static_stride_list<E0, E1, E2, E...>{
      static constexpr bool const value = (E0 > 1 || E1 > 1) && (E0 == 1 || E1 == 1) && is_all_ones<E2,E...>::value;
    };
    
    template<size_t E0, size_t E1>
    struct is_vector_static_stride_list<E0, E1>{
      static constexpr bool const value = (E0 > 1 || E1 > 1) && (E0 == 1 || E1 == 1);
    };

    template<size_t E0>
    struct is_vector_static_stride_list<E0>{
      static constexpr bool const value = E0 > 1;
    };

    template<>
    struct is_vector_static_stride_list<> : std::integral_constant<bool,false>{};
    
    
    template<size_t E0, size_t... E>
    struct is_all_ones<E0,E...>{
      static constexpr bool const value = (E0 == 1) && is_all_ones<E...>::value;
    };

    template<size_t E>
    struct is_all_ones<E> {
      static constexpr bool const value = (E == 1);
    };

    template<>
    struct is_all_ones<> : std::integral_constant<bool,false>{};
    
    // check if Extents is vector or scalar
    template<size_t... E>
    struct is_valid_static_stride_list{
      static constexpr bool const value = is_all_ones<E...>::value || is_vector_static_stride_list<E...>::value;
    };

  } // impl


  
  template<typename T, size_t N> 
  using gen_all_one_list_t = impl::gen_all_one_list_t<T,N>;

  template<size_t... E>
  inline static constexpr bool is_all_ones_v = impl::is_all_ones<E...>::value;

  template<size_t... E>
  inline static constexpr size_t static_stride_product_v = impl::static_stride_product<E...>::value;
  
  template<size_t... E>
  inline static constexpr bool is_vector_static_stride_list_v = impl::is_vector_static_stride_list<E...>::value;
  
  template<size_t... E>
  inline static constexpr bool is_valid_static_stride_list_v = impl::is_valid_static_stride_list<E...>::value;


  // @returns the static_stride_list containing strides for first order
  // It is a helper function or implementation
  template<typename T, size_t E0, size_t... E, size_t... R, size_t... P>
  constexpr auto get_strides_first_order_helper( static_stride_list<T, E0, E...>, 
    static_stride_list<T,R...>, static_stride_list<T,P...>)
  {
    if constexpr( sizeof...(E) == 0 ){
      return static_stride_list<T, 1ul, P...>{};
    }else{
      // add extent to the list which will be used for
      // take product for next iteration
      auto n = static_stride_list<T, R..., E0>{};

      // result list containing the strides
      // on each iteration calculate the product
      auto np = static_stride_list< T, P..., static_stride_product_v<R..., E0> >{};
      return get_strides_first_order_helper( static_stride_list<T, E...>{}, n, np );
    }
  }


  // @returns the static_stride_list containing strides for first order
  template<typename T, size_t... E>
  constexpr auto get_strides_first_order( static_stride_list<T, E...> )
  {
    // checks if extents are vector or scalar
    if constexpr( !is_valid_static_stride_list_v<E...> ){
      // if extents are empty return empty list
      if constexpr ( sizeof...(E) == 0 ){
        return static_stride_list<T>{};
      }else{
        return get_strides_first_order_helper(static_stride_list<T, E...>{}, static_stride_list<T>{}, static_stride_list<T>{});
      }
    }else{
      // @returns list contining ones if it is vector or scalar
      return gen_all_one_list_t<T, sizeof...(E)>{};
    }
  }

    // @returns the static_stride_list containing strides for last order
  // It is a helper function or implementation
  template<typename T, size_t E0, size_t... E, size_t... R, size_t... P>
  constexpr auto get_strides_last_order_helper( static_stride_list<T, E0, E...>, 
    static_stride_list<T, R...>, static_stride_list<T, P...>)
  {
    if constexpr( sizeof...(E) == 0 ){
      return static_stride_list<T, P..., E0, 1ul>{};
    }else{
      // add extent to the list which will be used for
      // take product for next iteration
      auto n = static_stride_list<T, R...,E0>{};

      // result list containing the strides
      // on each iteration calculate the product
      auto np = static_stride_list< T, P..., static_stride_product_v<E..., E0> >{};
      return get_strides_last_order_helper( static_stride_list<T, E...>{}, n, np );
    }
  }

  // @returns the static_stride_list containing strides for last order
  template<typename T, size_t E0, size_t... E>
  constexpr auto get_strides_last_order( static_stride_list<T, E0, E...> )
  {
    // checks if extents are vector or scalar
    if constexpr( !is_valid_static_stride_list_v<E0,E...> ){
      // if extent contains only one element return static_stride_list<T,1ul>
      if constexpr( sizeof...(E) == 0 ){
        return static_stride_list<T,1ul>{};
      }else{
        return get_strides_last_order_helper(static_stride_list<T, E...>{}, static_stride_list<T>{}, static_stride_list<T>{});
      }
    }else{
      // @returns list contining ones if it is vector or scalar
      return gen_all_one_list_t<T, sizeof...(E) + 1>{};
    }
  }
  
  // if extents are empty return empty list
  template<typename T>
  constexpr auto get_strides_last_order( static_stride_list<T> )
  {
    return static_stride_list<T>{};
  }

  template<typename Layout, typename T, size_t... E>
  struct strides_helper;

  // It is use for first order to
  // get std::array containing strides
  template<typename T, size_t... E>
  struct strides_helper<first_order,T, E...>{
    using type = decltype( get_strides_first_order(static_stride_list<T, E...>{}) );
    static constexpr auto value = type::value;
  };
  
  // It is use for last order to
  // get std::array containing strides
  template<typename T, size_t... E>
  struct strides_helper<last_order, T, E...>{
    using type = decltype( get_strides_last_order(static_stride_list<T, E...>{}) );
    static constexpr auto value = type::value;
  };

  template<typename Layout, typename T, size_t... E>
  inline static constexpr auto strides_helper_v = strides_helper<Layout, T, E...>::value;

} // namespace boost::numeric::ublas::detail
#endif
