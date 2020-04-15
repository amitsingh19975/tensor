#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_CONCEPT_FREE_TRAITS_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_CONCEPT_FREE_TRAITS_HPP

#include "../fwd.hpp"
#include <type_traits>

namespace boost::numeric::ublas::parallel{
    
    namespace properties{
        
        template<typename Entity, typename Property, typename = void>
        struct require_concept_free_traits{
            static constexpr auto const is_valid = false;
            static constexpr auto const is_noexcept = false;
        };
        
        template<typename Entity, typename Property>
        struct require_concept_free_traits<
            Entity, Property,
            std::void_t<
                decltype(require_concept(std::declval<Entity>(), std::declval<Property>()))
            >    
        >{
            using return_type = decltype(require_concept(std::declval<Entity>(), std::declval<Property>()));
            static constexpr auto const is_valid = true;
            static constexpr auto const is_noexcept = noexcept(require_concept(std::declval<Entity>(), std::declval<Property>()));
        };

    } // namespace properties
    

} // namespace boost::numeric::ublas::parallel

#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_REQUIRE_CONCEPT_FREE_TRAITS_HPP