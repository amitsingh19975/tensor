#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_OUTSTANDING_WORK_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_OUTSTANDING_WORK_HPP

#include "enumeration.hpp"

namespace boost::numeric::ublas::parallel::execution{

    struct outstanding_work_t : detail::enumeration<outstanding_work_t,2>{
        using detail::enumeration<outstanding_work_t,2>::enumeration;

        using untracked_t   = enumerator<0>;
        using tracked_t     = enumerator<1>;

        static constexpr auto const untracked= untracked_t{};
        static constexpr auto const tracked  = tracked_t{};
    };

    constexpr outstanding_work_t outstanding_work{};
    inline constexpr outstanding_work_t::untracked_t outstanding_work_t::untracked;
    inline constexpr outstanding_work_t::tracked_t outstanding_work_t::tracked;

} // namespace boost::numeric::ublas::parallel::execution

namespace boost::numeric::ublas::parallel{
    
    template<typename Entity>
    struct is_applicable_property<Entity, execution::outstanding_work_t,
        std::enable_if_t< execution::is_executor_v<Entity>> >
            : std::true_type {};
    
} // namespace boost::numeric::ublas::parallel

#endif