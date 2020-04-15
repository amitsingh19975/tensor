#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_ONEWAY_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_ONEWAY_HPP

#include "is_executor.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    struct oneway_t{
        
        using polymorphic_query_result_type = bool;

        static constexpr auto const is_requirable_concept   = true;
        static constexpr auto const is_requirable           = false;
        static constexpr auto const is_preferable           = false;

        template<typename... SupportableProperties>
        struct polymorphic_executor_type;

        template<typename Executor>
        static constexpr bool const static_query_v = is_oneway_executor<Executor>::value;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr bool value() noexcept { return true; }
    };

    constexpr oneway_t oneway;

} // namespace boost::numeric::ublas::parallel::execution

namespace boost::numeric::ublas::parallel
{
    template<class Entity>
    struct is_applicable_property<Entity, execution::oneway_t,
        std::enable_if_t< execution::is_executor_v<Entity> >
    >
        : std::true_type {};
    
} // namespace boost::numeric::ublas::parallel




#endif