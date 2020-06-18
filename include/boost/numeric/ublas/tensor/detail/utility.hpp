//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//

#ifndef BOOST_NUMERIC_UBLAS_DETAIL_UTILITY_HPP
#define BOOST_NUMERIC_UBLAS_DETAIL_UTILITY_HPP

#include <type_traits>
#include <utility>
#include <array>
#include <boost/mp11/detail/mp_list.hpp>
#include <boost/mp11/integer_sequence.hpp>
#include <boost/numeric/ublas/tensor/traits/type_traits_extents.hpp>

namespace boost::numeric::ublas::detail {
        
    template<std::size_t MaxIter,std::size_t I = 0>
    struct static_for_impl
        : std::integral_constant<std::size_t, MaxIter>
    {
        template<typename BinaryFn, typename T>
        constexpr auto operator()(BinaryFn fn, T ret) const{
            if constexpr( MaxIter <= I ){
                return ret;
            }else{
                std::integral_constant<std::size_t, I> info{};
                auto n_ret = fn(info,ret);
                return static_for_impl<MaxIter,I + 1>{}(std::move(fn),std::move(n_ret));
            }
        }
    };

    template<std::size_t Start, std::size_t MaxIter, typename T, typename BinaryFn>
    constexpr auto type_static_for(BinaryFn fn, T ret){
        return std::decay_t< decltype( static_for_impl<MaxIter,Start>{}(std::move(fn), std::move(ret)) ) > {};
    }

    template<std::size_t Start, std::size_t MaxIter, typename T, typename BinaryFn>
    constexpr auto static_for(BinaryFn fn, T ret){
        return static_for_impl<MaxIter,Start>{}(std::move(fn), std::move(ret));
    }
    
    template<typename T>
    struct seq_to_static_extents;
    
    template<typename T, T... Ns>
    struct seq_to_static_extents<boost::mp11::mp_list< std::integral_constant<T,Ns>... >>
    {
        using type = basic_static_extents<T,Ns...>;
    };
    
    template<typename T, T... Ns>
    struct seq_to_static_extents< std::integer_sequence< T, Ns... > >
    {
        using type = basic_static_extents<T,Ns...>;
    };

    template<typename T>
    using seq_to_static_extents_t = typename seq_to_static_extents<T>::type;
    
    template<typename T>
    struct static_extents_to_seq;
    
    template<typename T, T... Ns>
    struct static_extents_to_seq<basic_static_extents<T,Ns...>>
    {
        using type = boost::mp11::mp_list<std::integral_constant<T,Ns>...>;
    };
    
    template<typename T>
    using static_extents_to_seq_t = typename static_extents_to_seq<T>::type;

    template<typename T>
    struct seq_to_array;

    template<typename T, T... Ns>
    struct seq_to_array<boost::mp11::mp_list< std::integral_constant<T,Ns>... > >
        : seq_to_array< std::integer_sequence<T, Ns...> >
    {};
    
    template<typename T, T... Ns>
    struct seq_to_array< std::integer_sequence< T, Ns... > >
    {
        using type = std::array<T, sizeof...(Ns)>;
        static constexpr type const value{Ns...};
    };
    
    template<typename T>
    inline static constexpr auto const seq_to_array_v = seq_to_array<T>::value;

} // namespace boost::numeric::ublas::detail

#endif
