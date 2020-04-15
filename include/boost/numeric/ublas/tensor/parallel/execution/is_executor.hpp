#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_IS_EXECUTOR_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_IS_EXECUTOR_HPP

#include "executor_concept.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail::is_executor
    {
        
        template<typename T, typename = void>
        struct eval : std::false_type{};
        
        template<typename T>
        struct eval<
            T,
            std::void_t< decltype(executor_concept_t::static_query_v<T>) >
        > : std::true_type{};

    } // namespace detail::is_executor
    
    template<typename Executor>
    struct is_executor : detail::is_executor::eval<Executor> {};

    template<typename Executor>
    inline static constexpr auto const is_executor_v = is_executor<Executor>::value;

} // namespace boost::numeric::ublas::parallel::execution


#endif