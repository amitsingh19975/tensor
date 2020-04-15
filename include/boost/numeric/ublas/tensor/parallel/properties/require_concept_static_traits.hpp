#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_CONCEPT_STATIC_TRAITS_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_CONCEPT_STATIC_TRAITS_HPP

#include "../fwd.hpp"
#include <type_traits>

namespace boost::numeric::ublas::parallel{
    
    namespace properties{
        
        template<typename Entity, typename Property, typename = void>
        struct require_concept_static_traits{
            static constexpr auto const is_valid = false;
            static constexpr auto const is_noexcept = false;
        };
        
        template<typename Entity, typename Property>
        struct require_concept_static_traits<
            Entity, Property,
            std::enable_if_t<
                std::decay_t<Property>::value() == std::decay_t<Property>::template static_query_v<std::decay_t<Entity>>
            >    
        >{
            using return_type = std::decay_t<Entity>;
            static constexpr auto const is_valid = true;
            static constexpr auto const is_noexcept = noexcept(return_type(std::declval<Entity>()));
        };

    } // namespace properties
    

} // namespace boost::numeric::ublas::parallel

#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_CONCEPT_STATIC_TRAITS_HPP