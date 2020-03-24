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

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_FUNCTIONS_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_FUNCTIONS_HPP_

#include <boost/numeric/ublas/detail/config.hpp>

#include <boost/numeric/ublas/tensor/detail/meta_functions.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <type_traits>

namespace boost::numeric::ublas{
  
  template <class ExtentsType, size_t... E> struct basic_static_extents;
  template <class ExtentsType, size_t Rank> struct basic_fixed_rank_extents;
  template<class ExtentsType> class basic_extents;

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas::detail{
  
  template <size_t R, size_t...> struct basic_extents_impl;

  template<size_t... N> 
  struct number_list{
    static constexpr size_t const size = sizeof...(N);
  };

  template<size_t E, size_t...N>
  constexpr auto push_back(number_list<N...>) -> number_list<N...,E>;

  template<size_t E, size_t...N>
  constexpr auto push_front(number_list<N...>) -> number_list<E,N...>;

  template<size_t I, size_t E, typename T>
  struct is_same;
  
  template<size_t I, size_t Element, size_t N0, size_t...N>
  struct is_same<I,Element, number_list<N0,N...> >{
    static constexpr bool const value = is_same<I - 1, Element, number_list<N...> >::value;
  };

  template<size_t Element, size_t N0, size_t...N>
  struct is_same<0,Element, number_list<N0,N...> >{
    static constexpr bool const value = (Element == N0);
  };

  template< class ExtentsType, size_t... E >
  constexpr auto number_list_to_static_extents( number_list<E...> ) -> basic_static_extents<ExtentsType, E...>;

  template<size_t Begin, size_t End, size_t I = 0, size_t R, size_t... N>
  BOOST_UBLAS_INLINE
  constexpr auto get_number_list( basic_extents_impl<R>, number_list<N...> num = number_list<>{} ){
    return num;
  }

  template<size_t Begin, size_t End, size_t I = 0, size_t R, size_t E0, size_t... E, size_t... N>
  BOOST_UBLAS_INLINE
  constexpr auto get_number_list( basic_extents_impl<R, E0, E...>, number_list<N...> num = number_list<>{} ){
    // if range is out bound we reture num
    if constexpr( Begin >= End || I >= End){
      return num;
    // if current position is less than start of range
    // we just skip the push
    }else if constexpr( I < Begin ){
      if constexpr( sizeof...(E) == 0ul ){
        return num;
      }else{
        return get_number_list<Begin,End,I+1>(basic_extents_impl<R, E...>{}, num);
      }

    // if current position is less than end of range
    // we push the extents to the number list
    }else if constexpr( I < End ){
      auto n_num_list = decltype( push_back<E0>(num) ){};
      if constexpr( sizeof...(E) == 0ul ){
        return n_num_list;
      }else{
        return get_number_list<Begin,End,I+1>(basic_extents_impl<R, E...>{}, n_num_list);
      }
    }
  }

  template<typename Fn, size_t E0, size_t... E>
  BOOST_UBLAS_INLINE
  constexpr bool none_of_helper( number_list< E0, E...>, Fn pred){
    if constexpr( sizeof...(E) == 0ul ){
      return !pred(E0);
    }else{
      if( pred(E0) ){
        return false;
      }else{
        return none_of_helper( number_list<E...>{}, std::move(pred) );
      }
    }
  }

  template <class Fn, class ExtentsType, typename std::enable_if<is_static_extents<ExtentsType>::value, int>::type = 0>
  BOOST_UBLAS_INLINE
  constexpr bool none_of( ExtentsType const& e, Fn pred){
    auto num_list = get_number_list<0,ExtentsType::Rank>(typename ExtentsType::parent_type{});
    if constexpr( decltype(num_list)::size == 0 ){
      return true;
    }else{
      return none_of_helper(num_list, std::move(pred));
    }
  }

  template <size_t Begin, size_t End, class Fn, class ExtentsType, typename std::enable_if<is_static_extents<ExtentsType>::value, int>::type = 0>
  BOOST_UBLAS_INLINE
  constexpr bool none_of( ExtentsType const& e, Fn pred){
    auto num_list = get_number_list<Begin,End>(typename ExtentsType::parent_type{});
    if constexpr( decltype(num_list)::size == 0 ){
      return true;
    }else{
      return none_of_helper(num_list, std::move(pred));
    }
  }

  template<typename Fn, size_t E0, size_t... E>
  BOOST_UBLAS_INLINE
  constexpr bool all_of_helper( number_list< E0, E...>, Fn pred){
    if constexpr( sizeof...(E) == 0ul ){
      return pred(E0);
    }else{
      return pred(E0) && all_of_helper(number_list<E...>{}, std::move(pred));
    }
  }

  template <class Fn, class ExtentsType, typename std::enable_if<is_static_extents<ExtentsType>::value, int>::type = 0>
  BOOST_UBLAS_INLINE
  constexpr bool all_of( ExtentsType const& e, Fn pred){
    auto num_list = get_number_list<0,ExtentsType::Rank>(typename ExtentsType::parent_type{});
    if constexpr( decltype(num_list)::size == 0 ){
      return true;
    }else{
      return all_of_helper(num_list, std::move(pred));
    }
  }

  template <size_t Begin, size_t End, class Fn, class ExtentsType, typename std::enable_if<is_static_extents<ExtentsType>::value, int>::type = 0>
  BOOST_UBLAS_INLINE
  constexpr bool all_of( ExtentsType const& e, Fn pred){
    auto num_list = get_number_list<Begin,End>(typename ExtentsType::parent_type{});
    if constexpr( decltype(num_list)::size == 0 ){
      return true;
    }else{
      return all_of_helper(num_list, std::move(pred));
    }
  }

  template<typename Fn, size_t E0, size_t... E>
  BOOST_UBLAS_INLINE
  constexpr bool any_of_helper( number_list< E0, E...>, Fn pred){
    if constexpr( sizeof...(E) == 0ul ){
      return pred(E0);
    }else{
      return pred(E0) || any_of_helper(number_list<E...>{}, std::move(pred));
    }
  }

  template <class Fn, class ExtentsType, typename std::enable_if<is_static_extents<ExtentsType>::value, int>::type = 0>
  BOOST_UBLAS_INLINE
  constexpr bool any_of( ExtentsType const& e, Fn pred){
    auto num_list = get_number_list<0,ExtentsType::Rank>(typename ExtentsType::parent_type{});
    if constexpr( decltype(num_list)::size == 0 ){
      return false;
    }else{
      return any_of_helper(num_list, std::move(pred));
    }
  }

  template <size_t Begin, size_t End, class Fn, class ExtentsType, typename std::enable_if<is_static_extents<ExtentsType>::value, int>::type = 0>
  BOOST_UBLAS_INLINE
  constexpr bool any_of( ExtentsType const& e, Fn pred){
    auto num_list = get_number_list<Begin,End>(typename ExtentsType::parent_type{});
    if constexpr( decltype(num_list)::size == 0 ){
      return false;
    }else{
      return any_of_helper(num_list, std::move(pred));
    }
  }

  template <size_t E0, size_t... E, size_t... N>
  BOOST_UBLAS_INLINE
  constexpr auto squeeze_impl_remove_one( number_list<E0,E...>, number_list<N...> num = number_list<>{} ){
    // executed when number_list is size of 1
    // @code number_list<E0> @code
    if constexpr( sizeof...(E) == 0ul ){
      // if element E0 is 1 we return number list but we do not append
      // it to the list
      if constexpr( E0 == 1ul ){
        return num;
      }else{
        // if element E0 is 1 we return number list but we append
        // it to the list
        return decltype(push_back<E0>(num)){};
      }
    }else{
      if constexpr( E0 == 1ul ){
        // if element E0 is 1 we return number list but we do not append
        // it to the list
        return squeeze_impl_remove_one(number_list<E...>{}, num);
      }else{
        // if element E0 is 1 we return number list but we append
        // it to the list
        auto n_num_list = decltype(push_back<E0>(num)){};
        return squeeze_impl_remove_one(number_list<E...>{}, n_num_list);
      }
    }
  }

  template <class ExtentsType, typename std::enable_if<is_static<ExtentsType>::value, int>::type = 0>
  BOOST_UBLAS_INLINE
  constexpr auto squeeze_impl( ExtentsType const& e ){
    
    if constexpr( ExtentsType::size() <= typename ExtentsType::size_type(2) ){
      return e;
    }

    using value_type = typename ExtentsType::value_type;

    auto num_list = get_number_list<0,ExtentsType::Rank>(typename ExtentsType::parent_type{});
    auto one_free_num_list = squeeze_impl_remove_one(num_list);

    // check after removing 1s from the list are they same
    // if same that means 1s does not exist and no need to
    // squeeze
    if constexpr( decltype(one_free_num_list)::size != decltype(num_list)::size ){
      
      // after squeezing, all the extents are 1s we need to
      // return extents of (1, 1)
      if constexpr( decltype(one_free_num_list)::size == 0 ){

        return decltype( number_list_to_static_extents<value_type>(number_list<value_type(1),value_type(1)>{}) ){};

      }else if constexpr( decltype(one_free_num_list)::size == 1 ){
        // to comply with GNU Octave this check is made
        // if position 2 contains 1 we push at back
        // else we push at front
        if constexpr( is_same<1, 1ul, decltype(num_list) >::value ){
          return decltype( number_list_to_static_extents<value_type>(
                    decltype(push_back<value_type(1)>(one_free_num_list)){}
                  ) ){};
        }else{
          return decltype( number_list_to_static_extents<value_type>(
                    decltype(push_front<value_type(1)>(one_free_num_list)){}
                  ) ){};
        }

      }else{
        return decltype( number_list_to_static_extents<value_type>(one_free_num_list) ){};

      }

    }else{
      return decltype( number_list_to_static_extents<value_type>( num_list ) ) {};
    }
    
  }

  template <class ExtentsType, typename std::enable_if<is_dynamic<ExtentsType>::value && is_dynamic_rank<ExtentsType>::value, int>::type = 0>
  BOOST_UBLAS_INLINE
  constexpr auto squeeze_impl( ExtentsType const& e ){
    
    if( e.size() <= 2 ){
      return e;
    }

    using base_type = typename ExtentsType::base_type;
    using value_type = typename ExtentsType::value_type;
    base_type n_extents;

    // copying non 1s to the new extents
    std::copy_if(e.begin(), e.end(), std::back_inserter(n_extents), [](auto const& el){
      return el != value_type(1);
    });
    
    // checking if extents size goes blow 2
    switch( n_extents.size() ){
      // if size of extents goes to zero
      // return (1,1)
      case 0ul:{
        return ExtentsType{value_type(1),value_type(1)};
      }
      // if size of extents goes to 1
      // complying with GNU Octave
      // if position 2 contains 1 we push at back
      // else we push at front
      case 1ul:{
        if( e[1] != value_type(1) ){
          n_extents.insert(n_extents.begin(), value_type(1));
        }else{
          n_extents.push_back(value_type(1));
        }
        [[fallthrough]];
      }
      default:{
        return ExtentsType(n_extents); 
      }
    }
    
  }

  template <class ExtentsType, typename std::enable_if<is_dynamic<ExtentsType>::value && is_static_rank<ExtentsType>::value, int>::type = 0>
  BOOST_UBLAS_INLINE
  constexpr auto squeeze_impl( ExtentsType const& e ){
    if constexpr( ExtentsType::size() <= 2 ){
      return e;
    }else{
      return squeeze_impl(e.to_dynamic_extents());
    }
  }
    
    

} // namespace boost::numeric::ublas::detail


namespace boost::numeric::ublas {

/** @brief Returns true if size > 1 and all elements > 0 or size == 1 && e[0] == 1 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_INLINE 
constexpr bool valid(ExtentsType const &e) {

  if (e.size() == 1 && e[0] == 1)
      return true;

  if constexpr (detail::is_dynamic<ExtentsType>::value) {
    return e.size() > typename ExtentsType::size_type(1) &&
           std::none_of(e.begin(), e.end(), [](auto const &a) {
             return a == typename ExtentsType::value_type(0);
           });
  } else {
    return e.size() > typename ExtentsType::size_type(1) &&
           none_of(e, [](auto const &a) {
             return a == typename ExtentsType::value_type(0);
           });
  }
}

/**
 * @code static_extents<4,1,2,3,4> s;
 * std::cout<<to_string(extents); // {1,2,3,4}
 * @endcode
 * @returns the string of extents
 */

template <class T
  , typename std::enable_if<
      detail::is_extents<T>::value || 
      detail::is_strides<T>::value
      , int
    >::type = 0
>
BOOST_UBLAS_INLINE
std::string to_string(T const &e) {
  if (e.empty()) {
    return "[]";
  }

  std::string s = "[ ";
  
  for (auto i = typename T::size_type(0); i < e.size() - 1; i++) {
      s += std::to_string(e.at(i)) + ", ";
  }
  
  s += std::to_string(e.back()) + " ]";

  return s;
}

/** @brief Returns true if this has a scalar shape
 *
 * @returns true if (1,1,[1,...,1])
 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_INLINE 
constexpr bool is_scalar(ExtentsType const &e) {
  if (e.size() == typename ExtentsType::size_type(0)) {
    return false;
  }
  if constexpr (detail::is_dynamic<ExtentsType>::value) {
    return std::all_of(e.begin(), e.end(), [](auto const &a) {
      return a == typename ExtentsType::value_type(1);
    });
  } else {
    return detail::all_of(e, [](auto const &a) {
      return a == typename ExtentsType::value_type(1);
    });
  }
}

/**
 * @brief Returns true if this is a pure scalar. i.e rank=1 and product=1
 *
 * @note free scalars are used by expression templates to determine that an
 * operand is not bounded by shapes. In the following expression `5` has an
 * extent of free_scalar in the AST
 *
 * @code auto expr = 5 * tensor<int>{shape{3,3}}; @endcode
 *
 * @returns true if (1)
 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_INLINE 
constexpr bool is_free_scalar(ExtentsType const &e) {
  return e.size() == typename ExtentsType::size_type(1) && e[0] == typename ExtentsType::value_type(1);
}

/** @brief Returns true if this has a vector shape
 *
 * @returns true if (1,n,[1,...,1]) or (n,1,[1,...,1]) with n > 1
 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_INLINE 
constexpr bool is_vector(ExtentsType const &e) {
  
  if (e.size() == typename ExtentsType::size_type(0)) {
    return false;
  } else if (e.size() == typename ExtentsType::size_type(1)) {
    return e.at(0) > typename ExtentsType::value_type(1);
  }

  auto greater_one = [](auto const &a) {
    return a > typename ExtentsType::value_type(1);
  };
  auto equal_one = [](auto const &a) { return a == typename ExtentsType::value_type(1); };

  if constexpr (detail::is_dynamic<ExtentsType>::value) {
    return std::any_of(e.begin(), e.begin() + 2, greater_one) &&
           std::any_of(e.begin(), e.begin() + 2, equal_one) &&
           std::all_of(e.begin() + 2, e.end(), equal_one);
  } else {
    return detail::any_of<0, 2>(e, greater_one) &&
           detail::any_of<0, 2>(e, equal_one) &&
           detail::all_of<2, ExtentsType::rank()>(e, equal_one);
  }
}

/** @brief Returns true if this has a matrix shape
 *
 * @returns true if (m,n,[1,...,1]) with m > 1 and n > 1
 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_INLINE 
constexpr bool is_matrix(ExtentsType const &e) {
  if (e.size() < typename ExtentsType::size_type(2)) {
    return false;
  }

  auto greater_one = [](auto const &a) {
    return a > typename ExtentsType::value_type(1);
  };
  auto equal_one = [](auto const &a) { return a == typename ExtentsType::value_type(1); };

  if constexpr (detail::is_dynamic<ExtentsType>::value) {
    return std::all_of(e.begin(), e.begin() + 2, greater_one) &&
           std::all_of(e.begin() + 2, e.end(), equal_one);
  } else {
    return detail::all_of<0,2>(e, greater_one) &&
           detail::all_of<2,ExtentsType::rank()>(e, equal_one);
  }
}

/** @brief Returns true if this is has a tensor shape
 *
 * @returns true if !empty() && !is_scalar() && !is_vector() && !is_matrix()
 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_INLINE 
constexpr bool is_tensor(ExtentsType const &e) {
  if (e.size() < typename ExtentsType::size_type(3)) {
    return false;
  }

  auto greater_one = [](auto const &a) {
    return a > typename ExtentsType::value_type(1);
  };

  if constexpr (detail::is_dynamic<ExtentsType>::value) {
    return std::any_of(e.begin() + 2, e.end(), greater_one);
  } else {
    return detail::any_of<2,ExtentsType::rank()>(e, greater_one);
  }
}

/** @brief Eliminates singleton dimensions when size > 2
 *
 * squeeze {  1,1} -> {  1,1}
 * squeeze {  2,1} -> {  2,1}
 * squeeze {  1,2} -> {  1,2}
 *
 * squeeze {1,2,3} -> {  2,3}
 * squeeze {2,1,3} -> {  2,3}
 * squeeze {1,3,1} -> {  1,3}
 *
 * @returns basic_extents<int_type> with squeezed extents
 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_INLINE
auto squeeze(ExtentsType const &e) {
  return detail::squeeze_impl(e); 
}

/** @brief Returns the product of extents */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_INLINE
constexpr auto product(ExtentsType const &e) {

  if (e.empty()) {
    return typename ExtentsType::value_type(0);
  }

  if constexpr( detail::is_static<ExtentsType>::value){
      return detail::product_helper<ExtentsType>::value;
  }else {
    return typename ExtentsType::value_type(
        std::accumulate(e.begin(), e.end(), 1ul, std::multiplies<>()));
  }
}

} // namespace boost::numeric::ublas
#endif
