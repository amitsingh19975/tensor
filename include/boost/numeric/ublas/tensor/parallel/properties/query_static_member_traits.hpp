#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_QUERY_STATIC_MEMBER_TRAITS_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_QUERY_STATIC_MEMBER_TRAITS_HPP

#include "../fwd.hpp"
#include <type_traits>

namespace boost::numeric::ublas::parallel{
    
    namespace properties{
        
        template<typename Entity, typename Property, typename = void>
        struct query_static_member_traits{
            static constexpr auto const is_valid = false;
            static constexpr auto const is_noexcept = false;
        };
        
        template<typename Entity, typename Property>
        struct query_static_member_traits<
            Entity, Property,
            std::void_t<
                decltype( std::decay_t<Entity>::query(std::declval<Property>()) )
            >    
        >{
            using return_type = decltype( std::decay_t<Entity>::query(std::declval<Property>()) );
            static constexpr auto const is_valid = true;
            static constexpr auto const is_noexcept = noexcept(std::decay_t<Entity>::query(std::declval<Property>()));
        };

    } // namespace properties
    

} // namespace boost::numeric::ublas::parallel

#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_QUERY_STATIC_MEMBER_TRAITS_HPP