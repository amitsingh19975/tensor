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

#include <boost/numeric/ublas/tensor/detail/config.hpp>
#include <boost/numeric/ublas/tensor/detail/meta_functions.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <type_traits>

namespace boost::numeric::ublas {

/** @brief Returns true if size > 1 and all elements > 0 or size == 1 && e[0] == 1 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_TENSOR_INLINE 
constexpr bool valid(ExtentsType const &e) {

  if (e.size() == 1 && e[0] == 1)
      return true;

  if constexpr (!detail::is_static_extents<ExtentsType>::value) {
    return e.size() > typename ExtentsType::size_type(1) &&
           std::none_of(e.begin(), e.end(), [](auto const &a) {
             return a == typename ExtentsType::value_type(0);
           });
  } else {
    auto arr = e.to_array();
    return arr.size() > typename ExtentsType::size_type(1) &&
           std::none_of(arr.begin(), arr.end(), [](auto const &a) {
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
BOOST_UBLAS_TENSOR_INLINE
std::string to_string(T const &e) {
  if (e.empty()) {
    return "{}";
  };
  std::string s = "{ ";
  for (auto i = 0; i < e.size() - 1; i++) {
      s += std::to_string(e.at(i)) + ", ";
  }
  s += std::to_string(e.at(e.size() - 1)) + " }";
  return s;
}

/** @brief Returns true if this has a scalar shape
 *
 * @returns true if (1,1,[1,...,1])
 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_TENSOR_INLINE 
constexpr bool is_scalar(ExtentsType const &e) {
  if (e.size() == typename ExtentsType::size_type(0)) {
    return false;
  }
  if constexpr (!detail::is_static_extents<ExtentsType>::value) {
    return std::all_of(e.begin(), e.end(), [](auto const &a) {
      return a == typename ExtentsType::value_type(1);
    });
  } else {
    auto arr = e.to_array();
    return std::all_of(arr.begin(), arr.end(), [](auto const &a) {
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
BOOST_UBLAS_TENSOR_INLINE 
constexpr bool is_free_scalar(ExtentsType const &e) {
  return e.size() == 1 && e[0] == 1;
}

/** @brief Returns true if this has a vector shape
 *
 * @returns true if (1,n,[1,...,1]) or (n,1,[1,...,1]) with n > 1
 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_TENSOR_INLINE 
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

  if constexpr (!detail::is_static_extents<ExtentsType>::value) {
    return std::any_of(e.begin(), e.begin() + 2, greater_one) &&
           std::any_of(e.begin(), e.begin() + 2, equal_one) &&
           std::all_of(e.begin() + 2, e.end(), equal_one);
  } else {
    auto arr = e.to_array();
    return std::any_of(arr.begin(), arr.begin() + 2, greater_one) &&
           std::any_of(arr.begin(), arr.begin() + 2, equal_one) &&
           std::all_of(arr.begin() + 2, arr.end(), equal_one);
  }
}

/** @brief Returns true if this has a matrix shape
 *
 * @returns true if (m,n,[1,...,1]) with m > 1 and n > 1
 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_TENSOR_INLINE 
constexpr bool is_matrix(ExtentsType const &e) {
  if (e.size() < typename ExtentsType::size_type(2)) {
    return false;
  }

  auto greater_one = [](auto const &a) {
    return a > typename ExtentsType::value_type(1);
  };
  auto equal_one = [](auto const &a) { return a == typename ExtentsType::value_type(1); };

  if constexpr (!detail::is_static_extents<ExtentsType>::value) {
    return std::all_of(e.begin(), e.begin() + 2, greater_one) &&
           std::all_of(e.begin() + 2, e.end(), equal_one);
  } else {
    auto arr = e.to_array();
    return std::all_of(arr.begin(), arr.begin() + 2, greater_one) &&
           std::all_of(arr.begin() + 2, arr.end(), equal_one);
  }
}

/** @brief Returns true if this is has a tensor shape
 *
 * @returns true if !empty() && !is_scalar() && !is_vector() && !is_matrix()
 */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_TENSOR_INLINE 
constexpr bool is_tensor(ExtentsType const &e) {
  if (e.size() < typename ExtentsType::size_type(3)) {
    return false;
  }

  auto greater_one = [](auto const &a) {
    return a > typename ExtentsType::value_type(1);
  };

  if constexpr (!detail::is_static_extents<ExtentsType>::value) {
    return std::any_of(e.begin() + 2, e.end(), greater_one);
  } else {
    auto arr = e.to_array();
    return std::any_of(arr.begin() + 2, arr.end(), greater_one);
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
auto squeeze(ExtentsType const &e) {
  
  if (e.size() <= 2) {
    if constexpr (detail::is_static_extents<ExtentsType>::value) {
      return e.to_dynamic_extents();
    } else {
      return e;
    }
  }

  auto new_extent = typename ExtentsType::base_type();
  auto old_extent = e.base();

  auto insert_iter =
      std::back_insert_iterator<typename ExtentsType::base_type>(new_extent);

  auto c = std::count_if(old_extent.begin(), old_extent.end(), [](auto &n) {
    return n == typename ExtentsType::value_type(1);
  });

  auto num = old_extent.size() - c;

  if (old_extent[0] == typename ExtentsType::value_type(1) &&
      old_extent[1] != typename ExtentsType::value_type(1) && num == 1u) {
    new_extent.push_back(typename ExtentsType::value_type(1));
    new_extent.push_back(old_extent[1]);
  } else {
    std::remove_copy(old_extent.begin(), old_extent.end(), insert_iter,
                     typename ExtentsType::value_type(1));
  }

  while (new_extent.size() < 2)
    new_extent.push_back(typename ExtentsType::value_type(1));

  return basic_extents<typename ExtentsType::value_type>{new_extent};
}

/** @brief Returns the number of elements a tensor holds with this */
template <class ExtentsType, typename std::enable_if<detail::is_extents<ExtentsType>::value, int>::type = 0>
BOOST_UBLAS_TENSOR_INLINE 
constexpr auto product(ExtentsType const &e) {

  if (e.empty()) {
    return typename ExtentsType::value_type(0);
  }

  if constexpr( detail::is_static<ExtentsType>::value){
    if constexpr (ExtentsType::Rank == 0) {
      return typename ExtentsType::value_type(0);
    } else {
      return detail::product_helper<ExtentsType>::value;
    }
  }else {
    if constexpr (!detail::is_static_extents<ExtentsType>::value) {
      return typename ExtentsType::value_type(
          std::accumulate(e.begin(), e.end(), 1ul, std::multiplies<>()));
    } else {
      auto p = typename ExtentsType::value_type(1);
      for (auto i = 0; i < e.size(); i++) {
        p *= e.at(i);
      }
      return p;
    }
  }
}

} // namespace boost::numeric::ublas
#endif
