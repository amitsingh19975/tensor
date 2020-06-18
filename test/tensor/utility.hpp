//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef _BOOST_UBLAS_TEST_TENSOR_UTILITY_
#define _BOOST_UBLAS_TEST_TENSOR_UTILITY_

#include <utility>
#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>

template<class ... types>
struct zip_helper;

template<class type1, class ... types3>
struct zip_helper<std::tuple<types3...>, type1>
{
    template<class ... types2>
    struct with
    {
        using type = std::tuple<types3...,std::pair<type1,types2>...>;
    };
    template<class ... types2>
    using with_t = typename with<types2...>::type;
};


template<class type1, class ... types3, class ... types1>
struct zip_helper<std::tuple<types3...>, type1, types1...>
{
    template<class ... types2>
    struct with
    {
        using next_tuple = std::tuple<types3...,std::pair<type1,types2>...>;
        using type       = typename zip_helper<next_tuple, types1...>::template with<types2...>::type;
    };

    template<class ... types2>
    using with_t = typename with<types2...>::type;
};

template<class ... types>
using zip = zip_helper<std::tuple<>,types...>;

template<class CallBack, class... Ts>
constexpr auto for_each_tuple(std::tuple<Ts...>& t, CallBack call_back){
    boost::mp11::mp_for_each< boost::mp11::mp_iota_c< sizeof...(Ts) > >([&, call_back = std::move(call_back)](auto iter){
        constexpr auto const I = decltype(iter)::value;
        call_back(iter,std::get<I>(t));
    });
}

template<typename T>
struct mp_list_to_int_seq;

template<typename T>
using mp_list_to_int_seq_t = typename mp_list_to_int_seq<T>::type;

template<typename T, T... Ns>
struct mp_list_to_int_seq< boost::mp11::mp_list< std::integral_constant<T,Ns>... > >
{
    using type = std::integer_sequence< T, Ns... >;
};

template<>
struct mp_list_to_int_seq< boost::mp11::mp_list<> >
{
    using type = std::integer_sequence< std::size_t >;
};

#include <complex>

// To counter msvc warninig C4244
template<typename T>
struct inner_type{
    using type = T;
};

template<typename T>
struct inner_type< std::complex<T> >{
    using type = T;
};

template<typename T>
using inner_type_t = typename inner_type<T>::type;

/** @brief Implementing compile-time lexicographic_order permutation
 * @link https://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order
 * Steps:
 * 
 * 1. Find the largest index k such that a[k] < a[k + 1]. 
 * If no such index exists, the permutation is the last permutation.
 * 
 * 2. Find the largest index l greater than k such that a[k] < a[l].
 * 
 * 3. Swap the value of a[k] with that of a[l].
 * 
 * 4. Reverse the sequence from a[k + 1] up to and including the final element a[n].
*/

namespace lex::detail{
    

    // Step 1
    template< std::size_t I, typename List, typename Comp = std::less<>, std::size_t LargetIdx = std::numeric_limits<std::size_t>::max() >
    constexpr auto find_largest_index_k() noexcept{
        constexpr auto const npos = std::numeric_limits<std::size_t>::max();
        constexpr auto const size = boost::mp11::mp_size<List>::value;
        
        static_assert(size > 0ul, "find_largest_index_k: type list cannot be empty");

        if constexpr( size == 1 ){
            return std::integral_constant<std::size_t,0>{};
        }else{
            if constexpr( I >= size - 1 ){
                if constexpr( LargetIdx == npos ){
                    return std::integral_constant<std::size_t,size - 2>{};
                }else{
                    return std::integral_constant<std::size_t,LargetIdx>{};
                }
            }else{
                constexpr auto const prev = boost::mp11::mp_at_c<List,I - 1>::value;
                constexpr auto const curr = boost::mp11::mp_at_c<List,I>::value;
                constexpr auto const next = boost::mp11::mp_at_c<List,I + 1>::value;

                if constexpr( Comp{}(prev,curr) ){
                    if constexpr( Comp{}(curr,next) ){
                        return find_largest_index_k<I + 1, List, Comp, npos>();
                    }else{
                        return find_largest_index_k<I + 1, List, Comp, I - 1>();
                    }
                }else{
                    if constexpr( Comp{}(curr,next) ){
                        return find_largest_index_k<I + 1, List, Comp, npos>();
                    }else{
                        return find_largest_index_k<I + 1, List, Comp, LargetIdx>();
                    }
                }
            }
        }
    }

    // Step 2
    template< std::size_t K, std::size_t I, typename List, typename Comp = std::less<>, std::size_t LargestIdx = K>
    constexpr auto find_largest_index_i() noexcept{
        constexpr auto const size = boost::mp11::mp_size<List>::value;
        
        static_assert(size > 0 );

        if constexpr(size == 1){
            return std::integral_constant<std::size_t,0>{};
        }else{
            if constexpr( I >= size ){
                return std::integral_constant<std::size_t,LargestIdx>{};
            }else{
                constexpr auto const curr = boost::mp11::mp_at_c<List,K>::value;
                constexpr auto const next = boost::mp11::mp_at_c<List,I>::value;
                if constexpr( Comp{}(curr,next) ){
                    return find_largest_index_i<K, I + 1, List, Comp, I>();
                }else{
                    return find_largest_index_i<K, I + 1, List, Comp, LargestIdx>();
                }
                
            }
        }
    }

    // Step 3
    template<std::size_t I, std::size_t J, typename List>
    constexpr auto swap() noexcept{
        if constexpr( I == J ){
            return List{};
        }else{
            using i_temp = boost::mp11::mp_at_c<List,I>;
            using j_temp = boost::mp11::mp_at_c<List,J>;
            using temp = boost::mp11::mp_replace_at_c<List,I,j_temp>;
            return boost::mp11::mp_replace_at_c<temp,J,i_temp>{};
        }
    }

    // Step 4
    template<std::size_t I, typename List>
    constexpr auto reverse() noexcept{
        constexpr auto const size = boost::mp11::mp_size<List>::value;
        if constexpr( I >= size ){
            return List{};
        }else{
            using first_half = boost::mp11::mp_take_c<List,I>; 
            using second_half = boost::mp11::mp_drop_c<List,I>;
            using reverse_second_half = boost::mp11::mp_reverse<second_half>;
            return boost::mp11::mp_append<first_half,reverse_second_half>{};
        }
    }
    
    // Combining steps from 1 to 4
    template<typename List, typename Comp>
    constexpr auto static_permutation() noexcept{
        using step_1 = decltype(find_largest_index_k<1, List, Comp>());
        using step_2 = decltype(find_largest_index_i<step_1::value, step_1::value + 1, List, Comp>());
        using step_3 = decltype(swap<step_1::value,step_2::value,List>());
        return reverse<step_1::value + 1,step_3>();
    }
} // namespace detail

template<typename List>
using static_next_permutation = decltype(lex::detail::static_permutation<List,std::less<>>());

template<typename List>
using static_prev_permutation = decltype(lex::detail::static_permutation<List,std::greater<>>());

namespace detail{
    
    template<typename T>
    struct static_factorial_impl;

    template<typename T, T... Ns>
    struct static_factorial_impl< boost::mp11::mp_list< std::integral_constant<T,Ns>... > >
        : std::integral_constant< T, (... * Ns) >
    {};
    template<>
    struct static_factorial_impl< boost::mp11::mp_list<> >
        : std::integral_constant< std::size_t, 1ul>
    {};

} // namespace detail


template<std::size_t F>
struct static_factorial
    : std::integral_constant< 
        std::size_t, 
        detail::static_factorial_impl< 
            boost::mp11::mp_pop_front< boost::mp11::mp_iota_c<F + 1> >
        >::value 
    >
{};

template<std::size_t F>
inline static constexpr std::size_t const static_factorial_v = static_factorial<F>::value;
// creates e.g.
// using test_types = zip<long,float>::with_t<first_order,last_order>; // equals
// using test_types = std::tuple< std::pair<float, first_order>, std::pair<float, last_order >, std::pair<double,first_order>, std::pair<double,last_order >
//>;
//static_assert(std::is_same< std::tuple_element_t<0,std::tuple_element_t<0,test_types2>>, float>::value,"should be float ");
//static_assert(std::is_same< std::tuple_element_t<1,std::tuple_element_t<0,test_types2>>, boost::numeric::ublas::first_order>::value,"should be boost::numeric::ublas::first_order ");

#endif
