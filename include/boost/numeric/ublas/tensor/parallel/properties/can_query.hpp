#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_CAN_QUERY_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_CAN_QUERY_HPP

#include "../fwd.hpp"
#include <type_traits>
#include <tuple>

namespace boost::numeric::ublas::parallel{
    
    namespace properties::can_query_impl{
        
        template<class Entity, class Properties, class = std::void_t<>>
        struct eval : std::false_type {};

        template<class Entity, class Property>
        struct eval<Entity, std::tuple<Property>,
        std::void_t<decltype(
            query(std::declval<Entity>(), std::declval<Property>())
        )>> : std::true_type {};
    }

    template<typename Entity, typename Property> 
    struct can_query : properties::can_query_impl::eval<Entity,std::tuple<Property>>{};

} // namespace boost::numeric::ublas::parallel

#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_CAN_QUERY_HPP