#ifndef _BOOST_UBLAS_TENSOR_DETAIL_SIMD_HELPER_IMPL_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_SIMD_HELPER_IMPL_HPP

#include "simd_macro.hpp"
#include <cstddef>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <functional>

namespace boost::numeric::ublas::simd::detail{
    
    template< size_t N, typename T > struct is_valid_vector_size;
    template< size_t N, typename T > struct block_size;
    template< size_t N, typename T > struct simd_type;
    template< size_t N, typename T > struct assignment;
    template< size_t N, typename T > struct dot_product;
    template< size_t N, typename T > struct load;
    template< size_t N, typename T > struct store;
    template< typename T > struct multiplies;
    template< typename T > struct addition;
    template< typename T > struct subtraction;
    template< typename T > struct divides;
    template< typename T > struct horizontal_addition;

} // namespace boost::numeric::ublas::detail


namespace boost::numeric::ublas::simd::detail{
    
    template<size_t N, typename T> 
    struct is_valid_vector_size{
        constexpr static auto const value = ( N == 64 ) || ( N == 128 ) || ( N == 256 ) || ( N == 512 );
    };
    
    template<size_t N> 
    struct is_valid_vector_size<N,float>{
        constexpr static auto const value = ( N == 128 ) || ( N == 256 ) || ( N == 512 );
    };
    
    template<size_t N> 
    struct is_valid_vector_size<N,double>{
        constexpr static auto const value = ( N == 128 ) || ( N == 256 ) || ( N == 512 );
    };

    template<size_t N, typename T>
    struct block_size{
        constexpr static auto const value = N / ( 8 * sizeof(T) );
    };

    template<typename T>
    struct simd_type<64, T>{
        using type = __m64;
    };

    template<typename T>
    struct simd_type<128, T>{
        using type = __m128i;
    };

    template<typename T>
    struct simd_type<256, T>{
        using type = __m256i;
    };

    template<typename T>
    struct simd_type<512, T>{
        using type = __m512i;
    };

    template<>
    struct simd_type<128, float>{
        using type = __m128;
    };

    template<>
    struct simd_type<256, float>{
        using type = __m256;
    };

    template<>
    struct simd_type<512, float>{
        using type = __m512;
    };

    template<>
    struct simd_type<128, double>{
        using type = __m128d;
    };

    template<>
    struct simd_type<256, double>{
        using type = __m256d;
    };

    template<>
    struct simd_type<512, double>{
        using type = __m512d;
    };
} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas::simd::detail{
    
    template< size_t N, typename T > 
    inline constexpr static auto const is_valid_vector_size_v = is_valid_vector_size<N,T>::value;
    template< size_t N, typename T > 
    inline constexpr static auto const block_size_v = block_size<N,T>::value;
    template< size_t N, typename T > 
    using simd_type_t = typename simd_type<N,T>::type;

} // namespace boost::numeric::ublas::detail

#include <sstream>
#include <iostream>

namespace boost::numeric::ublas::simd::tool{
    
    namespace detail{
        template <typename A, typename... R>
        void crash_impl (std::ostringstream &oss, A const& arg, R... rest){
            oss << arg;
            if constexpr( sizeof...(rest) == 0 ){
                oss << std::endl;
                std::cerr<<oss.str();
                exit(1);
            }else{
                crash_impl(oss, std::forward<R>(rest)...);
            }
        }
    } // namespace detail
    

    template <typename... Args>
    void crash(Args&&... args){
        std::ostringstream oss;
        detail::crash_impl(oss, std::forward<Args>(args)...);
    }

} // namespace boost::numeric::ublas::simd::tool


#endif 
