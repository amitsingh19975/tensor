#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BULK_ONEWAY_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BULK_ONEWAY_HPP

#include "adapter.hpp"
#include "is_bulk_oneway_executor.hpp"
#include "is_executor.hpp"
#include "is_oneway_executor.hpp"
#include <memory>

namespace boost::numeric::ublas::parallel::execution{
    
    struct bulk_oneway_t{
        using polymorphic_query_result_type = bool;

        static constexpr bool is_requirable_concept = true;
        static constexpr bool is_requirable         = false;
        static constexpr bool is_preferable         = false;

        template<class... SupportableProperties>
        class polymorphic_executor_type;

        template<class Executor>
        static constexpr bool static_query_v = is_bulk_oneway_executor<Executor>::value;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr bool value() noexcept { return true; }

    private:
        template<typename Executor>
        struct adapter: detail::adapter<adapter,Executor>{
            using super_type = detail::adapter<adapter,Executor>;
            using detail::adapter<adapter,Executor>::adapter;
            using detail::adapter<adapter,Executor>::require;
            using detail::adapter<adapter,Executor>::query;

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr bulk_oneway_t query(executor_concept_t) { return {}; }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE Executor require_concept(oneway_t) const noexcept{
                return this->executor_;
            }

            template<typename Function, typename SharedFactory>
            void bulk_execute(Function f, size_t n, SharedFactory sf) const{
                auto shared_index = std::make_shared< std::atomic<size_t> >(0);
                auto shared_state = std::make_shared< decltype(sf) >(sf());

                for( auto i = size_t(0); i < n; ++i ){
                    try{
                        this->m_executor.execute(
                            [fn = std::move(f), shared_index, n, shared_state]() mutable{
                                for(auto i = shared_index->load(); i < n; ){
                                    if( shared_index->compare_exchange_weak(i, i + 1)){
                                        fn(i, *shared_state);
                                    }
                                }
                            }
                        );
                    }catch(...){
                        if( i == 0 ) throw;
                        else break;
                    }
                }
            }
        };

    public:
        template <typename Executor, typename =
        std::enable_if_t<
            is_oneway_executor<Executor>::value && !is_bulk_oneway_executor<Executor>::value,
            adapter<Executor>
        >>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend adapter<Executor> require_concept(Executor ex, bulk_oneway_t){
            return adapter<Executor>(std::move(ex));
        }

    };
    
    constexpr bulk_oneway_t bulk_oneway;

} // namespace boost::numeric::ublas::parallel::execution

namespace boost::numeric::ublas::parallel{
    
    template<class Entity>
    struct is_applicable_property<Entity, execution::bulk_oneway_t,
        std::enable_if_t< execution::is_executor_v<Entity>> > : std::true_type {};

} // namespace boost::numeric::ublas::parallel



#endif