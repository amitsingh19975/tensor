#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_HPP

#include "require_free_traits.hpp"
#include "require_member_traits.hpp"
#include "require_static_traits.hpp"
#include <utility>

#include <boost/core/demangle.hpp>
#include <iostream>

namespace boost::numeric::ublas::parallel{
    
    namespace properties::require_impl{
        
        struct require_fn{
            template<class Entity, class Property>
            constexpr auto operator()(Entity&& ex, Property&& p) const
            {
                if constexpr(
                    is_applicable_property_v<std::decay_t<Entity>, std::decay_t<Property>>
                    && std::decay_t<Property>::is_requirable
                    && require_static_traits<Entity, Property>::is_valid
                ){
                    return std::forward<Entity>(ex);
                }else if constexpr(
                    is_applicable_property_v<std::decay_t<Entity>, std::decay_t<Property>>
                    && std::decay_t<Property>::is_requirable
                    && !require_static_traits<Entity, Property>::is_valid
                    && require_member_traits<Entity, Property>::is_valid
                ){
                    return std::forward<Entity>(ex).require(std::forward<Property>(p));
                }else if constexpr(
                    is_applicable_property_v<std::decay_t<Entity>, std::decay_t<Property>>
                    && std::decay_t<Property>::is_requirable
                    && !require_static_traits<Entity, Property>::is_valid
                    && !require_member_traits<Entity, Property>::is_valid
                    && require_free_traits<Entity, Property>::is_valid
                ){
                    return require(std::forward<Entity>(ex), std::forward<Property>(p));
                }
            }

            template<class Entity, class Property0, class Property1, class... PropertyN>
            constexpr auto operator()(Entity&& ex, Property0&& p0, Property1&& p1, PropertyN&&... pn) const
                noexcept(noexcept(std::declval<require_fn>()(std::declval<require_fn>()(std::forward<Entity>(ex), std::forward<Property0>(p0)), std::forward<Property1>(p1), std::forward<PropertyN>(pn)...)))
                -> decltype(std::declval<require_fn>()(std::declval<require_fn>()(std::forward<Entity>(ex), std::forward<Property0>(p0)), std::forward<Property1>(p1), std::forward<PropertyN>(pn)...))
            {
                return (*this)((*this)(std::forward<Entity>(ex), std::forward<Property0>(p0)), std::forward<Property1>(p1), std::forward<PropertyN>(pn)...);
            }

        };
        
        template<typename T = require_fn> inline static constexpr T customization_point{};

    } // namespace properties::require_impl
    

} // namespace boost::numeric::ublas::parallel

#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_HPP