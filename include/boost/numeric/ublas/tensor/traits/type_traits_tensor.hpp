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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_TENSOR_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_TENSOR_HPP

#include <boost/numeric/ublas/tensor/traits/basic_type_traits.hpp>
#include <boost/numeric/ublas/tensor/traits/type_traits_extents.hpp>
#include <boost/numeric/ublas/tensor/traits/storage_traits.hpp>

namespace boost::numeric::ublas{
    using first_order   = column_major;
    using last_order    = row_major;
    
    template<typename T> class tensor_core;

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas {

    template<typename T>
    struct is_valid_tensor: std::false_type{};
    
    template<typename T>
    struct is_valid_tensor< tensor_core<T> >: std::true_type{};

    template<typename T>
    inline static constexpr bool is_valid_tensor_v = is_valid_tensor<T>::value;

    template<typename E, typename A, typename ValueType>
    struct rebind_storage{
        using type = typename storage_traits<A>::template rebind<ValueType>;
    };

    template<typename ValueType, typename U, std::size_t N, typename T, T... Ns>
    struct rebind_storage< basic_static_extents<T,Ns...>, std::array<U,N>, ValueType >
    {
        using type = std::array< ValueType, ( ... * Ns) >;
    };

    template<typename ValueType, typename U, std::size_t N, typename T>
    struct rebind_storage< basic_static_extents<T>, std::array<U,N>, ValueType >
    {
        using type = std::array< ValueType, N >;
    };

    template<typename E, typename A, typename ValueType>
    using rebind_storage_t = typename rebind_storage<E,A,ValueType>::type;

} // namespace boost::numeric::ublas

#endif
