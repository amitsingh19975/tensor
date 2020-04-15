#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_ADAPTER_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_ADAPTER_HPP

#include "../properties/query_static_member_traits.hpp"
#include "../properties/query_member_traits.hpp"
#include "../properties/require_member_traits.hpp"
#include "../properties/require_concept_member_traits.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail{
        template< template<typename> typename Derived, typename Executor>
        struct adapter{

            adapter(Executor ex): m_executor(std::move(ex)){}

            template<typename Property>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto require_concept(Property const& p) const
                noexcept( properties::require_concept_member_traits<Executor,Property>::is_noexcept )
                -> Derived< typename properties::require_concept_member_traits<Executor,Property>::result_type >
            {
                auto ret = m_executor.require_concept(p);
                return Derived< decltype(p) >(p);
            }

            template<typename Property>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto require(Property const& p) const
                noexcept( properties::require_member_traits<Executor,Property>::is_noexcept )
                -> Derived< typename properties::require_member_traits<Executor,Property>::result_type >
            {
                auto ret = m_executor.require(p);
                return Derived< decltype(p) >(p);
            }

            template<typename Property>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto query(Property const& p) const
                noexcept( properties::query_static_member_traits<Executor,Property>::is_noexcept )
                -> typename properties::query_static_member_traits<Executor,Property>::result_type
            {
                return Executor::query(p);
            }

            template<typename Property>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto query(Property const& p) const
                noexcept( properties::query_static_member_traits<Executor,Property>::is_noexcept )
                -> std::enable_if_t<
                    !properties::query_static_member_traits<Executor, Property>::is_valid,
                    typename properties::query_member_traits<Executor, Property>::result_type
                    >
            {
                return m_executor.query(p);
            }
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend constexpr bool operator==(const Derived<Executor>& a, const Derived<Executor>& b) noexcept {
                return a.m_executor == b.m_executor;
            }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend constexpr bool operator!=(const Derived<Executor>& a, const Derived<Executor>& b) noexcept{
                return a.m_executor != b.m_executor;
            }
        protected:
            Executor m_executor;
        };

    } // namespace detail
    

} // namespace boost::numeric::ublas::parallel::execution


#endif