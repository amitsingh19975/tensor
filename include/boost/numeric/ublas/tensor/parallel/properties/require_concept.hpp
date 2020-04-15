#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_CONCEPT_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_CONCEPT_HPP

#include "require_concept_free_traits.hpp"
#include "require_concept_member_traits.hpp"
#include "require_concept_static_traits.hpp"
#include <utility>

namespace boost::numeric::ublas::parallel{
    
    namespace properties::require_concept_impl{
        
        struct require_concept_fn{
            template<class Entity, class Property>
            constexpr auto operator()(Entity&& ex, Property&& p) const
            {
                if constexpr(
                    is_applicable_property_v<std::decay_t<Entity>, std::decay_t<Property>>
                    && std::decay_t<Property>::is_requirable_concept
                    && require_concept_static_traits<Entity, Property>::is_valid 
                ){
                    return std::forward<Entity>(ex);
                }else if constexpr(
                    is_applicable_property_v<std::decay_t<Entity>, std::decay_t<Property>>
                    && std::decay_t<Property>::is_requirable_concept
                    && !require_concept_static_traits<Entity, Property>::is_valid
                    && !require_concept_member_traits<Entity, Property>::is_valid
                    && require_concept_free_traits<Entity, Property>::is_valid
                ){
                    return require_concept(std::forward<Entity>(ex), std::forward<Property>(p));
                }else if constexpr(
                    is_applicable_property_v<std::decay_t<Entity>, std::decay_t<Property>>
                    && std::decay_t<Property>::is_requirable_concept
                    && !require_concept_static_traits<Entity, Property>::is_valid
                    && require_concept_member_traits<Entity, Property>::is_valid
                ){
                    return std::forward<Entity>(ex).require_concept(std::forward<Property>(p));
                }
            }
        };

        template<typename T = require_concept_fn> inline static constexpr T customization_point{};

    } // namespace properties::require_concept_impl
    

} // namespace boost::numeric::ublas::parallel

#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_CONCEPT_HPP