#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_CAN_PREFER_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_CAN_PREFER_HPP

#include "../fwd.hpp"
#include <type_traits>
#include <tuple>

namespace boost::numeric::ublas::parallel{
    
    namespace properties::can_prefer_impl{
        
        template<class Entity, class Properties, class = std::void_t<>>
        struct eval : std::false_type {};

        template<class Entity, class... Properties>
        struct eval<Entity, std::tuple<Properties...>,
        std::void_t<decltype(
            prefer(std::declval<Entity>(), std::declval<Properties>()...)
        )>> : std::true_type {};
    }

    template<typename Entity, typename... Properties> 
    struct can_prefer : properties::can_prefer_impl::eval<Entity, std::tuple<Properties...>>{};

} // namespace boost::numeric::ublas::parallel

#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_CAN_PREFER_HPP