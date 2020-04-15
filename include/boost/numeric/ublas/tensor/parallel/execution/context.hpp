#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_CONTEXT_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_CONTEXT_HPP

#include "is_executor.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail::context
    {
        template<typename D>
        struct property_base{
            static constexpr auto const is_requirable_concept   = false;
            static constexpr auto const is_requirable           = false;
            static constexpr auto const is_preferable           = false;

            template<typename Executor, typename Type = decltype(Executor::query(*static_cast<D*>(0)))>
            static constexpr Type const static_query_v = Executor::query(D{});
        };

    } // namespace detail::context
    
    struct context_t : detail::context::property_base<context_t> {};

    constexpr context_t context;

} // namespace boost::numeric::ublas::parallel::execution

namespace boost::numeric::ublas::parallel
{
    template<typename Entity>
    struct is_applicable_property<Entity, execution::context_t,
        std::enable_if_t< execution::is_executor_v<Entity> >
    >
        : std::true_type {};
    
} // namespace boost::numeric::ublas::parallel




#endif