#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_IS_INTERFACE_PROPERTY_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_IS_INTERFACE_PROPERTY_HPP

#include <type_traits>
#include "../detail/parallel_macro.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail::is_interface_property
    {
        
        template<typename T, typename = void>
        struct eval : std::false_type{};
        
        template<typename T>
        struct eval<
            T,
            std::void_t< typename T::template polymorphic_executor_type<> >
        > : std::integral_constant<bool, T::is_requirable_concept> {};

    } // namespace detail::is_interface_property
    
    template<typename Executor>
    struct is_interface_property : detail::is_interface_property::eval<Executor> {};

    template<typename Executor>
    inline static constexpr auto const is_interface_property_v = is_interface_property<Executor>::value;

} // namespace boost::numeric::ublas::parallel::execution


#endif