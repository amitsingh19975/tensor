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

#ifndef BOOST_UBLAS_TENSOR_ENGINE_TRAITS_IMPL_HPP
#define BOOST_UBLAS_TENSOR_ENGINE_TRAITS_IMPL_HPP

#include <boost/numeric/ublas/tensor/basic_tensor.hpp>

namespace boost::numeric::ublas{
    
    namespace detail{
        


    } // namespace detail
    

    template<typename... Traits>
    struct tensor_engine_traits;
    
    template<typename ShapeTraits, typename LayoutTraits, typename StorageTraits>
    struct tensor_engine_traits<ShapeTraits, LayoutTraits, StorageTraits >{
        using extents_type 	    = typename ShapeTraits::extents_type;
        using layout_type 	    = typename LayoutTraits::layout_type;
        using strides_type 	    = typename LayoutTraits::strides_type;
        using container_type    = typename StorageTraits::container_type;
        using container_tag	    = typename StorageTraits::container_tag;
    };

    namespace traits{

        struct dynamic_shape{
            using extents_type = dynamic_extents<>;
        };
        
        template<std::size_t N>
        struct fixed_shape{
            using extents_type = dynamic_extents<N>;
        };
        
        template<std::size_t... Ns>
        struct static_shape{
            using extents_type = static_extents<Ns...>;
        };

        template<typename Layout>
        struct dynamic_layout{
            using layout_type = Layout;
            using strides_type = basic_strides<std::size_t,layout_type>;
        };

        template<typename ExtentsTraits, typename Layout>
        struct static_layout{
        private:

            using extents_type = typename ExtentsTraits::extents_type;
            static_assert(is_static_v<extents_type>,"boost::numeric::ublas::traits::static_layout : "
                "the static layout cannot have dynamic extents"
            );
        
        public:
            using layout_type = Layout;
            using strides_type = strides_t<extents_type,layout_type>;
            
        };

        template<std::size_t N, typename Layout>
        struct fixed_layout{
            using layout_type = Layout;
            using strides_type = basic_fixed_rank_strides<std::size_t, N, Layout>;
            
        };

        template< typename ValueType, typename Allocator = std::allocator<ValueType> >
        struct dynamic_storage{
            using container_type    = std::vector<ValueType, Allocator>;
            using container_tag	    = tag::dynamic_storage;
        };
        
        // template<typename ValueType, std::size_t N>
        // struct fixed_storage{
        //     using container_type    = std::vector<ValueType>;
        //     using container_tag	    = tag::fixed_storage;
        //     using buffer_type       = std::array<std::byte,N>;
        // };
        
        template<typename, typename, std::size_t...>
        struct static_storage;

        template<typename ValueType, std::size_t... Es>
        struct static_storage<static_shape<Es...>, ValueType>{
        private:

            using extents_type = typename static_shape<Es...>::extents_type;
        
        public:
            using container_type    = std::array<ValueType,static_product_v<extents_type>>;
            using container_tag	    = tag::static_storage;
        };

        template<typename ValueType, std::size_t N, std::size_t... Es>
        struct static_storage<static_shape<Es...>, ValueType, N>{
        private:

            using extents_type = typename static_shape<Es...>::extents_type;
            static_assert( static_product_v<extents_type> <= N, 
                "boost::numeric::ublas::traits::static_storage<basic_static_extents<T,Es...>, ValueType, N> : "
                "N should always be greater than or equal to product of the extents"
            );

        public:
            using container_type    = std::array<ValueType,N>;
            using container_tag	    = tag::static_storage;
        };

        template<typename ExtentsTraits, typename ValueType, std::size_t N>
        struct static_storage<ExtentsTraits, ValueType, N>{
            using container_type    = std::array<ValueType,N>;
            using container_tag	    = tag::static_storage;
        };

    } // namespace traits

} // namespace boost::numeric::ublas


#endif
