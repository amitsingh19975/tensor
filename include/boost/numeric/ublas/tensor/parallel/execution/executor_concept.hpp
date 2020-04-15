#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_EXECUTOR_CONCEPT_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_EXECUTOR_CONCEPT_HPP

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail::executor_concept
    {
        template<typename D>
        struct property_base{
            static constexpr auto const is_requirable_concept   = false;
            static constexpr auto const is_requirable           = false;
            static constexpr auto const is_preferable           = false;

            template<typename Executor, typename Type = decltype(Executor::query(*static_cast<D*>(0)))>
            static constexpr Type const static_query_v = Executor::query(D());
        };

    } // namespace detail::executor_concept
    
    struct executor_concept_t : detail::executor_concept::property_base<executor_concept_t> {};
    
    constexpr executor_concept_t executor_concept{};

} // namespace boost::numeric::ublas::parallel::execution


namespace boost::numeric::ublas::parallel{
    template<typename Entity>
    struct is_applicable_property<Entity, execution::executor_concept_t> : std::true_type {};
} // namespace boost::numeric::ublas::parallel


#endif