#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_IS_ONEWAY_EXECUTOR_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_IS_ONEWAY_EXECUTOR_HPP

#include "executor_concept.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail::is_oneway_executor
    {
        
        template<typename T, typename = void>
        struct eval : std::false_type{};
        
        template<typename T>
        struct eval<
            T,
            std::enable_if_t< std::is_same_v< std::decay_t< decltype(executor_concept_t::static_query_v<T>) > , oneway_t> >
        > : std::true_type{};

    } // namespace detail::is_oneway_executor
    
    template<typename Executor>
    struct is_oneway_executor : detail::is_oneway_executor::eval<Executor> {};

    template<typename Executor>
    inline static constexpr auto const is_oneway_executor_v = is_oneway_executor<Executor>::value;

} // namespace boost::numeric::ublas::parallel::execution


#endif