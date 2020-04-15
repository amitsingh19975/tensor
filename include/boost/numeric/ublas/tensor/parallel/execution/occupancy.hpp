#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_OCCUPANCY_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_OCCUPANCY_HPP

#include <type_traits>
#include "../detail/parallel_macro.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail::occupancy
    {
        
        template<typename Derived>
        struct property_base{
            
            using polymorphic_query_result_type = size_t;

            static constexpr auto const is_requireable  = false;
            static constexpr auto const is_preferable   = false;

            template<class Executor, class Type = decltype(Executor::query(*static_cast<Derived*>(0)))>
            static constexpr Type static_query_v = Executor::query(Derived());
        };

    } // namespace detail::occupancy
    
    struct occupancy_t : detail::occupancy::property_base< occupancy_t > {};

    constexpr auto occupancy = occupancy_t{};

} // namespace boost::numeric::ublas::parallel::execution


#endif