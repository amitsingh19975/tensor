#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_PREFER_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_PREFER_HPP

#include "prefer_free_traits.hpp"
#include "require_concept_member_traits.hpp"
#include "require_concept_static_traits.hpp"
#include <utility>

namespace boost::numeric::ublas::parallel{
    
    namespace properties::prefer_impl{
        
        struct prefer_fn{
            template<typename Entity, typename Property>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( Entity&& e, Property&& ) const 
                -> std::enable_if_t<
                    is_applicable_property< std::decay_t<Entity>, std::decay_t<Property> >::value
                    && properties::require_static_traits<Entity,Property>::is_valid
                    && std::decay_t<Property>::is_preferable
                    ,typename properties::require_static_traits<Entity,Property>::return_type
                >
            {
                return std::forward<Entity>(e);
            }

            template<typename Entity, typename Property>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( Entity&& e, Property&& p) const
                noexcept(properties::require_member_traits<Entity,Property>::is_noexcept)
                -> std::enable_if_t<
                    is_applicable_property< std::decay_t<Entity>, std::decay_t<Property> >::value
                    && !properties::require_static_traits<Entity,Property>::is_valid
                    && properties::require_member_traits<Entity,Property>::is_valid
                    && std::decay_t<Property>::is_preferable
                    ,typename properties::require_member_traits<Entity,Property>::return_type
                >
            {
                return std::forward<Entity>(e).require(std::forward<Property>(p));
            }

            template<typename Entity, typename Property>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( Entity&& e, Property&& p) const
                noexcept(properties::prefer_free_traits<Entity,Property>::is_noexcept)
                -> std::enable_if_t<
                    is_applicable_property< std::decay_t<Entity>, std::decay_t<Property> >::value
                    && !properties::require_static_traits<Entity,Property>::is_valid
                    && !properties::require_member_traits<Entity,Property>::is_valid
                    && properties::prefer_free_traits<Entity,Property>::is_valid
                    && std::decay_t<Property>::is_preferable
                    ,typename properties::prefer_free_traits<Entity,Property>::return_type
                >
            {
                return prefer(std::forward<Entity>(e), std::forward<Property>(p));
            }

            template<class Entity, class Property0, class Property1, class... PropertyN>
            constexpr auto operator()(Entity&& ex, Property0&& p0, Property1&& p1, PropertyN&&... pn) const
                noexcept(noexcept(std::declval<prefer_fn>()(std::declval<prefer_fn>()(std::forward<Entity>(ex), std::forward<Property0>(p0)), std::forward<Property1>(p1), std::forward<PropertyN>(pn)...)))
                -> decltype(std::declval<prefer_fn>()(std::declval<prefer_fn>()(std::forward<Entity>(ex), std::forward<Property0>(p0)), std::forward<Property1>(p1), std::forward<PropertyN>(pn)...))
            {
                return (*this)((*this)(std::forward<Entity>(ex), std::forward<Property0>(p0)), std::forward<Property1>(p1), std::forward<PropertyN>(pn)...);
            }
        };

        template<typename T = prefer_fn> inline static constexpr T customization_point{};

    } // namespace properties::require_concept
    

} // namespace boost::numeric::ublas::parallel

#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_PREFER_HPP