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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_TAGS_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_TAGS_HPP

#include <boost/numeric/ublas/tensor/detail/basic_type_traits.hpp>

namespace boost::numeric::ublas::tag{       

    struct dynamic_shape;

    struct dynamic_layout;
    struct dynamic_storage;

    template<std::size_t...>
    struct static_shape;
    
    template<std::size_t...>
    struct static_layout;
    
    template<std::size_t...>
    struct static_storage;

    template<std::size_t>
    struct fixed_shape;

    template<std::size_t>
    struct fixed_layout;
    
} // namespace boost::numeric::ublas::tag


namespace boost::numeric::ublas {

    template<typename T>
    struct is_tag
        : std::integral_constant<bool, 
            std::is_same_v<T,tag::dynamic_shape> ||
            std::is_same_v<T,tag::dynamic_layout> ||
            std::is_same_v<T,tag::dynamic_storage>
        >
    {};

    template<std::size_t... Ns>
    struct is_tag<tag::static_shape<Ns...>>
        : std::integral_constant<bool,true>
    {};

    template<std::size_t... Ns>
    struct is_tag<tag::static_layout<Ns...>>
        : std::integral_constant<bool,true>
    {};

    template<std::size_t N>
    struct is_tag<tag::fixed_shape<N>>
        : std::integral_constant<bool,true>
    {};

    template<std::size_t N>
    struct is_tag<tag::fixed_layout<N>>
        : std::integral_constant<bool,true>
    {};

    template<typename T>
    inline static constexpr auto const is_tag_v = is_tag<T>::value;

    template<typename T>
    struct is_static_layout
        : std::false_type{};
    
    template<std::size_t... Ns>
    struct is_static_layout< tag::static_layout<Ns...> > 
        : std::true_type{};

    template<typename T>
    struct is_static_storage
        : std::false_type{};
    
    template<std::size_t... Ns>
    struct is_static_storage< tag::static_storage<Ns...> > 
        : std::true_type{};

    
    template<typename T>
    inline static constexpr auto const is_static_layout_v = is_static_layout<T>::value;
    
    template<typename T>
    inline static constexpr auto const is_static_storage_v = is_static_storage<T>::value;

} // namespace boost::numeric::ublas


#endif
