#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_ENUMERATOR_ADAPTER_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_ENUMERATOR_ADAPTER_HPP

#include "adapter.hpp"
#include "../properties/require.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail{
        
        template<template<typename> typename Derived, typename Executor, typename Enumeration, typename Enumerator>
        struct enumerator_adapter : adapter<Derived, Executor>{
            
            using adapter<Derived, Executor>::adapter;
            using adapter<Derived, Executor>::require;
            using adapter<Derived, Executor>::query;

            template<int N>
            constexpr auto require(const typename Enumeration::template enumerator<N>& p) const
                -> decltype(require(std::declval<Executor>(), p))
            {
                return require(this->executor_, p);
            }

            static constexpr Enumeration query(const Enumeration&) noexcept
            {
                return Enumerator{};
            }

            template<int N>
            static constexpr Enumeration query(const typename Enumeration::template enumerator<N>&) noexcept
            {
                return Enumerator{};
            }

        };

    } // namespace detail
    

} // namespace boost::numeric::ublas::parallel::execution




#endif