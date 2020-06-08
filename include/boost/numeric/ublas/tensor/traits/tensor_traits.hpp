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


#ifndef _BOOST_UBLAS_TRAITS_TENSOR_TRAITS_HPP_
#define _BOOST_UBLAS_TRAITS_TENSOR_TRAITS_HPP_

#include <boost/numeric/ublas/tensor/traits/storage_traits.hpp>
#include <boost/numeric/ublas/tensor/tags.hpp>

namespace boost::numeric::ublas{
    
    template <typename T> class basic_extents;
    template <typename T, typename L> class basic_strides;
    template <typename T, T...> struct basic_static_extents;
    template <typename E, typename L> struct basic_static_strides;
    template <typename T, std::size_t N> struct basic_fixed_rank_extents;
    template <typename T, std::size_t N, typename L> class basic_fixed_rank_strides;

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas{



    template<typename...>
    struct extents_traits;

    template<typename T>
    struct extents_traits< basic_extents<T> >{
        using extents_type = basic_extents<T>;
        using extents_tag = extents_resizable_dynamic_tag;
    };
    
    template<typename T, std::size_t N>
    struct extents_traits< basic_fixed_rank_extents<T,N> >{
        using extents_type = basic_fixed_rank_extents<T,N>;
        using extents_tag = extents_fixed_dynamic_tag;
    };
    
    template<typename T, T... Ns>
    struct extents_traits< basic_static_extents<T,Ns...> >{
        using extents_type = basic_static_extents<T,Ns...>;
        using extents_tag = extents_static_tag;
    };
    
    template<>
    struct extents_traits<>
        : extents_traits< basic_extents<std::size_t> >
    {};

    template<typename...>
    struct layout_traits;

    template<typename Layout, typename T>
    struct layout_traits< basic_extents<T>, Layout >{
        using layout_type = Layout;
        using strides_type = basic_strides<T,layout_type>;
        using layout_tag = layout_resizable_dynamic_tag;
    };

    template<typename Layout, typename T, T... Ns>
    struct layout_traits< basic_static_extents<T,Ns...>, Layout >{
        using layout_type = Layout;
        using strides_type = basic_static_strides<basic_static_extents<T,Ns...>,layout_type>;
        using layout_tag = layout_static_tag;
    };

    template<typename Layout, typename T, std::size_t N>
    struct layout_traits< basic_fixed_rank_extents<T,N>, Layout >{
        using layout_type = Layout;
        using strides_type = basic_fixed_rank_strides<T,N,layout_type>;
        using layout_tag = layout_fixed_dynamic_tag;
    };

    template<typename T,typename Layout>
    struct layout_traits< extents_traits<T>, Layout > 
        : layout_traits<T,Layout>
    {};

    template<typename Layout>
    struct layout_traits< Layout > 
        : layout_traits< basic_extents<std::size_t> ,Layout>
    {};

    template< typename ExtentTraits, typename LayoutTraits, typename Storage >
    struct tensor_engine_traits;
    
    template<typename ExtentTraits, typename LayoutTraits, typename Storage>
    struct tensor_engine_traits{
        using extents_type 	        = typename ExtentTraits::extents_type;
        using layout_type 	        = typename LayoutTraits::layout_type;
        using strides_type 	        = typename LayoutTraits::strides_type;
        using container_type        = typename storage_traits<Storage>::array_type;
        using container_tag         = typename storage_traits<Storage>::container_tag;
        using resizable_tag         = typename storage_traits<Storage>::resizable_tag;
    };


} // namespace boost::numeric::ublas

#endif // _BOOST_UBLAS_TRAITS_TENSOR_TRAITS_HPP_

