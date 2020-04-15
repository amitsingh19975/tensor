#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BULK_GUARANTEE_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BULK_GUARANTEE_HPP

#include "enumeration.hpp"

namespace boost::numeric::ublas::parallel::execution{

    struct bulk_guarantee_t : detail::enumeration<bulk_guarantee_t,4>{
        using detail::enumeration<bulk_guarantee_t,4>::enumeration;

        using unsequenced_t     = enumerator<0>;
        using sequenced_t       = enumerator<1>;
        using parallel_t        = enumerator<2>;
        using distributed_t     = enumerator<3>;

        static constexpr auto const unsequenced     = unsequenced_t{};
        static constexpr auto const sequenced       = sequenced_t{};
        static constexpr auto const parallel        = parallel_t{};
        static constexpr auto const distributed     = distributed_t{};
    };

    constexpr bulk_guarantee_t bulk_guarantee{};
    inline constexpr bulk_guarantee_t::unsequenced_t bulk_guarantee_t::unsequenced;
    inline constexpr bulk_guarantee_t::sequenced_t bulk_guarantee_t::sequenced;
    inline constexpr bulk_guarantee_t::parallel_t bulk_guarantee_t::parallel;
    inline constexpr bulk_guarantee_t::distributed_t bulk_guarantee_t::distributed;

} // namespace boost::numeric::ublas::parallel::execution

namespace boost::numeric::ublas::parallel{
    
    template<typename Entity>
    struct is_applicable_property<Entity, execution::bulk_guarantee_t,
        std::enable_if_t< execution::is_executor_v<Entity>> >
            : std::true_type {};
    
} // namespace boost::numeric::ublas::parallel

#endif