#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_EXECUTOR_SHAPE_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_EXECUTOR_SHAPE_HPP

#include <cstddef>
#include <type_traits>

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail::executor_shape
    {
        
        template<typename Executor, typename = void>
        struct eval{
            using type = size_t;
        };

        template<typename Executor>
        struct eval<Executor, std::void_t<typename Executor::shape_type>>{
            using type = typename decltype(require_concept(std::declval<const Executor&>(), execution::bulk_oneway))::shape_type;
        };

    } // namespace detail::executor_shape
    
    template<class Executor>
    struct executor_shape : detail::executor_shape::eval<Executor> {};

    template<typename Executor>
    using executor_shape_t = typename executor_shape<Executor>::type;

} // namespace boost::numeric::ublas::parallel::execution


#endif