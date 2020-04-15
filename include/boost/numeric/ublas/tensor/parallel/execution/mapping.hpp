#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_MAPPING_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_MAPPING_HPP

#include "enumeration.hpp"

namespace boost::numeric::ublas::parallel::execution{

    struct mapping_t : detail::enumeration<mapping_t,3>{
        using detail::enumeration<mapping_t,3>::enumeration;

        using thread_t      = enumerator<0>;
        using new_thread_t  = enumerator<1>;
        using other_t       = enumerator<2>;

        static constexpr auto const thread      = thread_t{};
        static constexpr auto const new_thread  = new_thread_t{};
        static constexpr auto const other       = other_t{};
    };

    constexpr mapping_t mapping{};
    inline constexpr mapping_t::thread_t mapping_t::thread;
    inline constexpr mapping_t::new_thread_t mapping_t::new_thread;
    inline constexpr mapping_t::other_t mapping_t::other;

} // namespace boost::numeric::ublas::parallel::execution

namespace boost::numeric::ublas::parallel{
    
    template<typename Entity>
    struct is_applicable_property<Entity, execution::mapping_t,
        std::enable_if_t< execution::is_executor_v<Entity>> >
            : std::true_type {};
    
} // namespace boost::numeric::ublas::parallel

#endif