//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//


#ifndef BOOST_UBLAS_TENSOR_TAGS_HPP
#define BOOST_UBLAS_TENSOR_TAGS_HPP

#include <cstddef>
#include <boost/numeric/ublas/tensor/detail/type_traits_tensor.hpp>

namespace boost::numeric::ublas {

    template <typename T> class basic_extents;
    template <typename T, typename L> class basic_strides;
    template <typename T, T...> struct basic_static_extents;
    template <typename E, typename L> struct basic_static_strides;
    template <typename T, std::size_t N> struct basic_fixed_rank_extents;
    template <typename T, std::size_t N, typename L> class basic_fixed_rank_strides;

    namespace tag{
        struct tensor{};

        struct dynamic_shape{
            using type = basic_extents<std::size_t>;
        };

        struct dynamic_layout{
            using type = basic_strides<std::size_t,first_order>;
            using layout_type = first_order;
        };
        struct dynamic_storage{};

        template<std::size_t... Ns>
        struct static_shape {
            using type = basic_static_extents< std::size_t, Ns...>;
        };
        
        template<std::size_t... Ns>
        struct static_layout{
            using type = basic_static_strides< basic_static_extents< std::size_t, Ns...>, custom_order>;
            using layout_type = custom_order;
        };
        
        template<std::size_t...>
        struct static_storage;
        
        template<std::size_t MaxSize>
        struct static_storage<MaxSize>{
            static constexpr auto const value = MaxSize;
        };
        
        template<>
        struct static_storage<>{
            static constexpr auto const value = 0;
        };

        template<std::size_t N>
        struct fixed_shape{
            using type = basic_fixed_rank_extents<std::size_t,N>;
        };

        template<std::size_t N>
        struct fixed_layout{
            using type = basic_fixed_rank_strides<std::size_t,N,first_order>;
            using layout_type = first_order;
        };

    } // namespace tag
    

} // boost::numeric::ublas

#endif
