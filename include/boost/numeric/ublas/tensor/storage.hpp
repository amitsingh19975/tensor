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

#ifndef BOOST_UBLAS_TENSOR_STORAGE_HPP
#define BOOST_UBLAS_TENSOR_STORAGE_HPP

#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <boost/numeric/ublas/tensor/detail/type_traits.hpp>
#include <boost/numeric/ublas/tensor/detail/extents_functions.hpp>

namespace boost::numeric::ublas::storage {

namespace dense_tensor {
  
  template <typename T, typename E, typename A, typename = void>
  struct default_storage;

  template <typename T, typename E, typename A>
  struct default_storage<T, E, A,
                        typename std::enable_if<boost::numeric::ublas::detail::is_dynamic_v<E>>::type> {
    using type = std::vector<T, A>;
  };

  template <typename T, typename E, typename A>
  struct default_storage<T, E, A, typename std::enable_if<boost::numeric::ublas::detail::is_static_v<E>>::type> {
    using type = std::array<T, static_traits::product_v<E>>;
  };

  template <typename T, typename E, typename A>
  using default_storage_t = typename default_storage<T, E, A>::type;

} // namespace dense_tensor

} // namespace boost::numeric::ublas::storage

#endif
