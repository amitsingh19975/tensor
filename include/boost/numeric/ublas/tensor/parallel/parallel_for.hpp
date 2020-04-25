#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PARALLEL_FOR_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PARALLEL_FOR_HPP

#include "fwd.hpp"
#include "execution/range.hpp"
#include "detail/noexcept_function.hpp"
#include "execution.hpp"
#include "detail/query_or.hpp"


namespace boost::numeric::ublas::parallel{
    
    namespace detail
    {
        template<typename ExecutionPolicy, typename Range, typename Size, typename Function>
        void parallel_for(ExecutionPolicy&& policy, Range first, Size n, Function f, Size stride){

            static_assert( (std::is_invocable_v<Function,Size> || std::is_invocable_v<Function>), "boost::numeric::ublas::parallel::detail::parallel_for : Invalid arguments passed" );

            auto noexcept_fn = detail::make_noexcept_fn<Function>(std::move(f));
            try{
                auto ex = require(
                    require_concept(policy.executor(),execution::bulk_oneway),
                    policy.execution_requirement,
                    execution::blocking.always
                );

                Size num_subranges = std::min( n, detail::query_or(ex, execution::occupancy, Size(4)));

                ex.bulk_execute(
                    [=](size_t subrange_idx, const execution::range<size_t>& block)
                    {   
                        for(Size i = block.begin(subrange_idx); i < block.end(subrange_idx); ++i)
                        {
                            if constexpr( std::is_invocable_v<Function,Size> ){
                                noexcept_fn( i );
                            }else if constexpr( std::is_invocable_v<Function> ){
                                noexcept_fn();
                            }
                        }
                    },
                    num_subranges,
                    [=]
                    {
                        return execution::range<size_t>(first, n, stride, num_subranges);
                    }
                );
            }catch(...){
                for(Size i = first; i < n; i += stride)
                {
                    if constexpr( std::is_invocable_v<Function,Size> ){
                        noexcept_fn( i );
                    }else if constexpr( std::is_invocable_v<Function> ){
                        noexcept_fn();
                    }
                }
            }

        }
    } // namespace detail
    


    template<typename ExecutionPolicy, typename Range, typename Size1, typename Size2, typename Function>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE void parallel_for(ExecutionPolicy&& policy, Range first, Size1 n, Function f, Size2 stride)
    {
        using size_type = std::common_type_t<Size1,Size2>;
        detail::parallel_for(std::forward<ExecutionPolicy>(policy), first, static_cast<size_type>(n), std::move(f), static_cast<size_type>(stride));
    }

    template<typename ExecutionPolicy, typename Range, typename Size, typename Function>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE void parallel_for(ExecutionPolicy&& policy, Range first, Size n, Function f)
    {
        parallel_for(std::forward<ExecutionPolicy>(policy), first, n, std::move(f), Size(1) );
    }

} // namespace boost::numeric::ublas::parallel


#endif