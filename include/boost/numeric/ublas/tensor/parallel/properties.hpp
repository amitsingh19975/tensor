#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_HPP

#include "fwd.hpp"
#include "properties/is_applicable_property.hpp"
#include "properties/require.hpp"
#include "properties/query.hpp"
#include "properties/prefer.hpp"
#include "properties/require_concept.hpp"

namespace boost::numeric::ublas::parallel
{
    inline static constexpr auto const& require_concept    = properties::require_concept_impl::customization_point<>;
    inline static constexpr auto const& require            = properties::require_impl::customization_point<>;
    inline static constexpr auto const& prefer             = properties::prefer_impl::customization_point<>;
    inline static constexpr auto const& query              = properties::query_impl::customization_point<>;

    template<typename Entity, typename Property> 
    inline static constexpr auto const can_require_concept_v  = can_require_concept<Entity,Property>::value;
    
    template<typename Entity, typename Property> 
    inline static constexpr auto const can_query_v            = can_query<Entity,Property>::value;
    
    template<typename Entity, typename... Properties> 
    inline static constexpr auto const can_require_v          = can_require<Entity,Properties...>::value;
    
    template<typename Entity, typename... Properties>
    inline static constexpr auto const can_prefer_v           = can_prefer<Entity,Properties...>::value;

} // namespace boost::numeric::ublas::parallel


#include "properties/can_prefer.hpp"
#include "properties/can_require_concept.hpp"
#include "properties/can_require.hpp"
#include "properties/can_query.hpp"


#endif