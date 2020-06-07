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


#ifndef BOOST_UBLAS_TENSOR_IMPL_HPP
#define BOOST_UBLAS_TENSOR_IMPL_HPP

#include <boost/numeric/ublas/tensor/basic_tensor.hpp>

namespace boost::numeric::ublas {

    namespace detail{
        
        template<typename T>
        struct match_type_from_tag{
            using type = T;
        };
        
        template<>
        struct match_type_from_tag<first_order>{
            using layout_type = first_order;
        };
        
        template<>
        struct match_type_from_tag<last_order>{
            using layout_type = last_order;
        };
        
        template<>
        struct match_type_from_tag<tag::dynamic_shape> {
            using type = typename tag::dynamic_shape::type;
        };

        template<>
        struct match_type_from_tag<tag::dynamic_layout> {
            using type = typename tag::dynamic_layout::type;
            using layout_type = typename tag::dynamic_layout::layout_type;
        };

        template<std::size_t... Ns>
        struct match_type_from_tag<tag::static_layout<Ns...>> {
            using type = typename tag::static_layout<Ns...>::type;
            using layout_type = typename tag::static_layout<Ns...>::layout_type;
        };
  
        template<std::size_t... Ns>
        struct match_type_from_tag<tag::static_shape<Ns...>> {
            using type = typename tag::static_shape<Ns...>::type;
        };
  
        template<std::size_t N>
        struct match_type_from_tag<tag::fixed_shape<N>> {
            using type = typename tag::fixed_shape<N>::type;
        };
  
        template<std::size_t N>
        struct match_type_from_tag<tag::fixed_layout<N>> {
            using type = typename tag::fixed_layout<N>::type;
            using layout_type = typename tag::fixed_layout<N>::layout_type;
        };

        template<typename T>
        using match_type_from_tag_t = typename match_type_from_tag<T>::type;


        template<typename E, typename T>
        struct select_strides;

        template<typename E, typename L>
        struct select_strides
            : std::conditional<
                is_strides_v< detail::match_type_from_tag_t< L > >,
                detail::match_type_from_tag_t< L >,
                strides_t<E,L>
            >
        {};

        template<typename E, typename L>
        using select_strides_t = typename select_strides<E,L>::type;

        template<typename E, typename T>
        struct select_container_size{
            static_assert(always_false_v<T>, "boost::numeric::ublas::detail::select_container_size : " 
                "tensor can not have dynamic extents with static storage with no max storage size provided"
            );
        };
        
        template<std::size_t MaxSize, typename T, T... Es>
        struct select_container_size< basic_static_extents<T,Es...>, tag::static_storage<MaxSize>>
            : std::integral_constant< std::size_t, std::max( static_product_v< basic_static_extents<T,Es...> >, MaxSize ) >
        {
            static_assert( ( static_product_v< basic_static_extents<T,Es...> > ) < MaxSize, "boost::numeric::ublas::detail::select_container_size : " 
                "Max size provided is less than it requires to store the provided extents"
            );
        };

        template<typename T, T... Es>
        struct select_container_size< basic_static_extents<T,Es...>, tag::static_storage<>>
            : std::integral_constant< std::size_t, static_product_v< basic_static_extents<T,Es...> > >
        {};

        template<typename E, std::size_t MaxSize>
        struct select_container_size< E, tag::static_storage<MaxSize>>
            : std::integral_constant< std::size_t, MaxSize >
        {};


        template<typename E, typename T>
        inline static constexpr auto const select_container_size_v = select_container_size<E,T>::value;

    } // namespace detail
    

    template<typename... Args>
    struct tensor_traits;
    
    template<typename ValueType, typename ExtentsType, typename LayoutType>
    struct tensor_traits<ValueType,ExtentsType,LayoutType,tag::dynamic_storage>{
        using container_type    = std::vector< ValueType >;
        using extents_type      = detail::match_type_from_tag_t<ExtentsType>;
        using layout_type       = typename detail::match_type_from_tag<LayoutType>::layout_type;
        using strides_type      = detail::select_strides_t<extents_type,LayoutType>;
        using container_tag     = tag::dynamic_storage;
    };

    template<typename ValueType, typename ExtentsType, typename LayoutType, std::size_t... Sz>
    struct tensor_traits<ValueType, ExtentsType, LayoutType, tag::static_storage<Sz...> >{
        using extents_type      = detail::match_type_from_tag_t<ExtentsType>;
        using container_type    = std::array< ValueType, detail::select_container_size_v<extents_type, tag::static_storage<Sz...> > >;
        using layout_type       = typename detail::match_type_from_tag<LayoutType>::layout_type;
        using strides_type      = detail::select_strides_t<extents_type,LayoutType>;
        using container_tag     = tag::static_storage<>;
    };
    
    template<typename ValueType, typename ExtentsType, typename LayoutType, typename ContainerType, typename ContainerTag>
    struct tensor_traits<ValueType, ExtentsType, LayoutType, ContainerType, ContainerTag >{
        using extents_type      = detail::match_type_from_tag_t<ExtentsType>;
        using container_type    = ContainerType;
        using layout_type       = typename detail::match_type_from_tag<LayoutType>::layout_type;
        using strides_type      = detail::select_strides_t<extents_type,LayoutType>;
        using container_tag     = std::conditional_t< std::is_same_v<tag::dynamic_storage, ContainerTag >, ContainerTag, tag::static_storage<> >;
    };



    template<typename... Traits>
    using tensor = basic_tensor< tensor_traits<Traits...> >;

} // boost::numeric::ublas

#endif
