#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_QUERY_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_QUERY_HPP

#include "query_free_traits.hpp"
#include "query_member_traits.hpp"
#include "query_static_traits.hpp"
#include <utility>

namespace boost::numeric::ublas::parallel{
    
    namespace properties::query_impl{
        
        struct query_fn{
            template<typename Entity, typename Property>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( Entity&&, Property&& ) const 
                -> std::enable_if_t<
                    is_applicable_property< std::decay_t<Entity>, std::decay_t<Property> >::value
                    && properties::query_static_traits<Entity,Property>::is_valid
                    ,typename properties::query_static_traits<Entity,Property>::return_type
                >
            {
                return std::decay_t<Property>::template static_query_v<std::decay_t<Entity>>;
            }

            template<typename Entity, typename Property>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( Entity&& e, Property&& p) const
                noexcept(properties::query_member_traits<Entity,Property>::is_noexcept)
                -> std::enable_if_t<
                    is_applicable_property< std::decay_t<Entity>, std::decay_t<Property> >::value
                    && !properties::query_static_traits<Entity,Property>::is_valid
                    && properties::query_member_traits<Entity,Property>::is_valid
                    ,typename properties::query_member_traits<Entity,Property>::return_type
                >
            {
                return std::forward<Entity>(e).query(std::forward<Property>(p));
            }

            template<typename Entity, typename Property>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( Entity&& e, Property&& p) const
                noexcept(properties::query_free_traits<Entity,Property>::is_noexcept)
                -> std::enable_if_t<
                    is_applicable_property< std::decay_t<Entity>, std::decay_t<Property> >::value
                    && !properties::query_static_traits<Entity,Property>::is_valid
                    && !properties::query_member_traits<Entity,Property>::is_valid
                    && properties::query_free_traits<Entity,Property>::is_valid
                    ,typename properties::query_free_traits<Entity,Property>::return_type
                >
            {
                return query(std::forward<Entity>(e), std::forward<Property>(p));
            }

        };
        
        template<typename T = query_fn> inline static constexpr T customization_point{};

    } // namespace properties::query_impl
    

} // namespace boost::numeric::ublas::parallel

#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_QUERY_HPP