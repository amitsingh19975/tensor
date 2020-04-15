#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_EXECUTOR_INDEX_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_EXECUTOR_INDEX_HPP

#include "executor_shape.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail::executor_index
    {
        
        template<typename Executor, typename = void>
        struct eval{
            using type = executor_shape_t<Executor>;
        };

        template<typename Executor>
        struct eval<Executor, std::void_t<typename Executor::index_type>>{
            using type = typename decltype(require_concept(std::declval<const Executor&>(), execution::bulk_oneway))::index_type;
        };

    } // namespace detail::executor_index
    
    template<class Executor>
    struct executor_index : detail::executor_index::eval<Executor> {};

    template<typename Executor>
    using executor_index_t = typename executor_index<Executor>::type;

} // namespace boost::numeric::ublas::parallel::execution


#endif