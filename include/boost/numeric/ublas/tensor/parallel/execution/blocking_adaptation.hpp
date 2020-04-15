#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BLOCKING_ADAPTATION_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BLOCKING_ADAPTATION_HPP

#include "enumeration.hpp"
#include "enumerator_adapter.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    struct blocking_adaptation_t : detail::enumeration<blocking_adaptation_t, 2>{

        using detail::enumeration<blocking_adaptation_t, 2>::enumeration;

        using disallowed_t  = enumerator<0>;
        using allowed_t     = enumerator<1>;

        static constexpr auto const disallowed = disallowed_t{};
        static constexpr auto const allowed = allowed_t{};

    private:

        template<typename Executor>
        struct adapter: detail::enumerator_adapter<adapter,Executor, blocking_adaptation_t, allowed_t>{
        
        private:
            template<typename T>
            static auto inner_declval() -> decltype(std::declval<Executor>());
        
        public:
            using detail::enumerator_adapter<adapter,Executor, blocking_adaptation_t, allowed_t>::enumerator_adapter;

            template<class Function> 
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto execute(Function f) const
                -> decltype(inner_declval<Function>().execute(std::move(f)))
            {
                return this->m_executor.execute(std::move(f));
            }

            template<class Function>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto twoway_execute(Function f) const
                -> decltype(inner_declval<Function>().twoway_execute(std::move(f)))
            {
                return this->m_executor.twoway_execute(std::move(f));
            }

            template<class Function, class SharedFactory>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto bulk_execute(Function f, std::size_t n, SharedFactory sf) const
                -> decltype(inner_declval<Function>().bulk_execute(std::move(f), n, std::move(sf)))
            {
                return this->m_executor.bulk_execute(std::move(f), n, std::move(sf));
            }

            template<class Function, class ResultFactory, class SharedFactory>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto bulk_twoway_execute(Function f, std::size_t n, ResultFactory rf, SharedFactory sf) const
                -> decltype(inner_declval<Function>().bulk_twoway_execute(std::move(f), n, std::move(rf), std::move(sf)))
            {
                return this->m_executor.bulk_twoway_execute(std::move(f), n, std::move(rf), std::move(sf));
            }
        };
    public:
        template <typename Executor>
        friend adapter<Executor> require(Executor ex, allowed_t)
        {
            return adapter<Executor>(std::move(ex));
        }
    };
    
    constexpr blocking_adaptation_t blocking_adaptation{};
    inline constexpr blocking_adaptation_t::disallowed_t blocking_adaptation_t::disallowed;
    inline constexpr blocking_adaptation_t::allowed_t blocking_adaptation_t::allowed;

} // namespace boost::numeric::ublas::parallel::execution


namespace boost::numeric::ublas::parallel{
    
    template<typename Entity>
    struct is_applicable_property<Entity, execution::blocking_adaptation_t,
        std::enable_if_t< execution::is_executor_v<Entity>> >
            : std::true_type {};
    
} // namespace boost::numeric::ublas::parallel


#endif