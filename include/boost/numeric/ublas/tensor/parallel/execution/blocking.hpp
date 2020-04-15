#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BLOCKING_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BLOCKING_HPP

#include "blocking_adaptation.hpp"
#include "enumeration.hpp"
#include "enumerator_adapter.hpp"
#include <future>


namespace boost::numeric::ublas::parallel::execution{
    
    struct blocking_t : detail::enumeration<blocking_t,3>{
        using detail::enumeration<blocking_t, 3>::enumeration;

        using possibly_t = enumerator<0>;
        using always_t = enumerator<1>;
        using never_t = enumerator<2>;

        static constexpr possibly_t const possibly    = possibly_t{};
        static constexpr always_t const always        = always_t{};
        static constexpr never_t const never          = never_t{};

    private:
        template<typename Executor>
        struct adapter: detail::enumerator_adapter<adapter,Executor, blocking_t, always_t>{
        
        private:
            template<typename T>
            static auto inner_declval() -> decltype(std::declval<Executor>());
        
        public:
            using detail::enumerator_adapter<adapter,Executor, blocking_t, always_t>::enumerator_adapter;

            template<class Function> 
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto execute(Function f) const
                -> decltype(inner_declval<Function>().execute(std::move(f)))
            {
                std::promise<void> promise;
                std::future<void> future = promise.get_future();
                this->m_executor.execute([fn = std::move(f), p = std::move(promise)]() mutable {
                    fn();
                });
            }

            template<class Function>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto twoway_execute(Function f) const
                -> decltype(inner_declval<Function>().twoway_execute(std::move(f)))
            {
                auto future = this->m_executor.twoway_execute(std::move(f));
                future.wait();
                return future;
            }

            template<class Function, class SharedFactory>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto bulk_execute(Function f, std::size_t n, SharedFactory sf) const
                -> decltype(inner_declval<Function>().bulk_execute(std::move(f), n, std::move(sf)))
            {
                std::promise<void> promise;
                std::future<void> future = promise.get_future();
                this->m_executor.execute([fn = std::move(f), p = std::move(promise)](auto i, auto s) mutable {
                    fn();
                }, n, std::move(sf));
            }

            template<class Function, class ResultFactory, class SharedFactory>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto bulk_twoway_execute(Function f, std::size_t n, ResultFactory rf, SharedFactory sf) const
                -> decltype(inner_declval<Function>().bulk_twoway_execute(std::move(f), n, std::move(rf), std::move(sf)))
            {
                auto future = this->m_executor.bulk_twoway_execute(std::move(f), n, std::move(rf), std::move(sf));
                future.wait();
                return future;
            }

        };
    public:
        template<typename Executor,
            typename = std::enable_if_t<
            blocking_adaptation_t::static_query_v<Executor>
                == blocking_adaptation.allowed
            >>
        friend adapter<Executor> require(Executor ex, always_t)
        {
            return adapter<Executor>(std::move(ex));
        }
    };

    constexpr blocking_t blocking{};
    inline constexpr blocking_t::possibly_t blocking_t::possibly;
    inline constexpr blocking_t::always_t blocking_t::always;
    inline constexpr blocking_t::never_t blocking_t::never;

} // namespace boost::numeric::ublas::parallel::execution



namespace boost::numeric::ublas::parallel{
    
    template<typename Entity>
    struct is_applicable_property<Entity, execution::blocking_t,
        std::enable_if_t< execution::is_executor_v<Entity>> >
            : std::true_type {};
    
} // namespace boost::numeric::ublas::parallel


#endif