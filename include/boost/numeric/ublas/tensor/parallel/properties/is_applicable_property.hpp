#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_IS_APPLICABLE_PROPERTY_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_IS_APPLICABLE_PROPERTY_HPP

#include "../fwd.hpp"
#include <type_traits>

namespace boost::numeric::ublas::parallel{
    
    template<typename Entity, typename Property, typename = void>
    struct is_applicable_property: std::false_type{};

    template<typename Entity, typename Property>
    inline static constexpr bool const is_applicable_property_v = is_applicable_property<Entity, Property>::value;

} // namespace boost::numeric::ublas::parallel

#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_IS_APPLICABLE_PROPERTY_HPP