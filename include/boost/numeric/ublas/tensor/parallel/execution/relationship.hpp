#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_RELATIONSHIP_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_RELATIONSHIP_HPP

#include "enumeration.hpp"

namespace boost::numeric::ublas::parallel::execution{

    struct relationship_t : detail::enumeration<relationship_t,3>{
        using detail::enumeration<relationship_t,3>::enumeration;

        using fork_t            = enumerator<0>;
        using chunk_t           = enumerator<1>;
        using continuation_t    = enumerator<2>;

        static constexpr auto const fork            = fork_t{};
        static constexpr auto const chunk           = chunk_t{};
        static constexpr auto const continuation    = continuation_t{};
    };

    constexpr relationship_t relationship{};
    inline constexpr relationship_t::fork_t relationship_t::fork;
    inline constexpr relationship_t::chunk_t relationship_t::chunk;
    inline constexpr relationship_t::continuation_t relationship_t::continuation;

} // namespace boost::numeric::ublas::parallel::execution

namespace boost::numeric::ublas::parallel{
    
    template<typename Entity>
    struct is_applicable_property<Entity, execution::relationship_t,
        std::enable_if_t< execution::is_executor_v<Entity>> >
            : std::true_type {};
    
} // namespace boost::numeric::ublas::parallel

#endif