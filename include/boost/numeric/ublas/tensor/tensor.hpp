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

#ifndef BOOST_UBLAS_TENSOR_ENGINE_IMPL_HPP
#define BOOST_UBLAS_TENSOR_ENGINE_IMPL_HPP

#include <boost/numeric/ublas/tensor/tensor_core.hpp>
#include <boost/numeric/ublas/tensor/traits/storage_traits.hpp>

namespace boost::numeric::ublas{

    namespace layout{

        template<typename...>
        struct first_order;
        
        template<>
        struct first_order<>{
            
            template<typename ExtentsType>
            struct strides{
                using extents_type = ExtentsType;
                using layout_type = ::boost::numeric::ublas::last_order;
                using strides_type = strides_t<ExtentsType,layout_type>;
            };
            
        };
        
        template<typename ExtentsType>
        struct first_order<ExtentsType>{
            
            template<typename>
            struct strides{
                using extents_type = ExtentsType;
                using layout_type = ::boost::numeric::ublas::last_order;
                using strides_type = strides_t<ExtentsType,layout_type>;
            };

        };

        template<typename...>
        struct last_order;

        template<>
        struct last_order<>{
            
            template<typename ExtentsType>
            struct strides{
                using extents_type = ExtentsType;
                using layout_type = ::boost::numeric::ublas::last_order;
                using strides_type = strides_t<ExtentsType,layout_type>;
            };
            
        };
        
        template<typename ExtentsType>
        struct last_order<ExtentsType>{
            
            template<typename>
            struct strides{
                using extents_type = ExtentsType;
                using layout_type = ::boost::numeric::ublas::last_order;
                using strides_type = strides_t<ExtentsType,layout_type>;
            };

        };

        template<typename,typename>
        struct extract_strides;

        template<typename ExtentsType, typename Layout>
        struct extract_strides
        {
            using type = typename Layout::template strides<ExtentsType>;
        };

        template<typename ExtentsType, typename Layout>
        using extract_strides_t = typename extract_strides<ExtentsType,Layout>::type;
        
    } // namespace layout

    
    template<typename...>
    struct tensor_engine;

    template<typename ExtentsType, typename LayoutType, typename StorageType>
    struct tensor_engine<ExtentsType, LayoutType, StorageType>{
        using extents_type 	        = ExtentsType;
        using layout_type 	        = typename layout::extract_strides_t<extents_type,LayoutType>::layout_type;
        using strides_type 	        = typename layout::extract_strides_t<extents_type,LayoutType>::strides_type;
        using container_type        = typename storage_traits<StorageType>::array_type;
        using container_tag         = typename storage_traits<StorageType>::container_tag;
        using resizable_tag         = typename storage_traits<StorageType>::resizable_tag;
    };
    
    template<typename LayoutType, typename StorageType>
    struct tensor_engine<LayoutType, StorageType>{
        using extents_type 	        = typename layout::extract_strides_t<int,LayoutType>::extents_type;
        using layout_type 	        = typename layout::extract_strides_t<int,LayoutType>::layout_type;
        using strides_type 	        = typename layout::extract_strides_t<int,LayoutType>::strides_type;
        using container_type        = typename storage_traits<StorageType>::array_type;
        using container_tag         = typename storage_traits<StorageType>::container_tag;
        using resizable_tag         = typename storage_traits<StorageType>::resizable_tag;
    };


} // namespace boost::numeric::ublas


#endif
