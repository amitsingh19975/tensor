#ifndef _BOOST_UBLAS_TENSOR_DETAIL_SIMD_LOAD_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_SIMD_LOAD_HPP

#include "simd_assignment_impl.hpp"

namespace boost::numeric::ublas::simd::detail{
    
    template<typename T>
    struct load<128,T>{
        using m128i = simd_type_t<128,int>;
        using m128 = simd_type_t<128,float>;
        using m128d = simd_type_t<128,double>;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( T const* p ) const noexcept{
            if constexpr(std::is_same_v<T, int16_t> || std::is_same_v<T, short>){
                return _mm_loadu_si16(p);
            }else if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int>){
                return _mm_loadu_si32(p);
            }else if constexpr(std::is_same_v<T, int64_t> || std::is_same_v<T, long long>){
                return _mm_loadu_si64(p);
            }else{
                return *reinterpret_cast<m128i const*>(p);
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( float const* p, size_t k, size_t w ) const noexcept{
            if( k == 1 ){
                return assignment<128,T>{}( *(p + w * 0), 0, 0, 0);
            }else if( k == 2 ){
                return assignment<128,T>{}( *(p + w * 0), *(p + w * 1), 0, 0);
            }else if( k == 3 ){
                return assignment<128,T>{}( *(p + w * 0), *(p + w * 1), *(p + w * 2), 0);
            }else if( k == 4 ){
                return assignment<128,T>{}( *(p + w * 0), *(p + w * 1), *(p + w * 2), *(p + w * 3));
            }else{
                return assignment<128,T>{}(0);
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( float const* p, size_t w ) const noexcept{
            return assignment<128,T>{}( *(p + w * 0), *(p + w * 1), *(p + w * 2), *(p + w * 3) );
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( m128i const* p ) const noexcept{
            return _mm_loadu_si128(p);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( float const* p ) const noexcept{
            return _mm_loadu_ps(p);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( double const* p ) const noexcept{
            return _mm_loadu_pd(p);
        }
    };

    template<typename T>
    struct load<256,T>{
        using m256i = simd_type_t<256,int>;
        using m256 = simd_type_t<256,float>;
        using m256d = simd_type_t<256,double>;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( m256i const* p ) const noexcept{
            return _mm256_loadu_si256(p);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( float const* p ) const noexcept{
            return _mm256_loadu_ps(p);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( float const* p, size_t k, size_t w ) const noexcept{
            if( k == 1 ){
                return assignment<256,T>{}( *(p + w * 0), 0, 0, 0, 0, 0, 0, 0);
            }else if( k == 2 ){
                return assignment<256,T>{}( *(p + w * 0), *(p + w * 1), 0, 0, 0, 0, 0, 0);
            }else if( k == 3 ){
                return assignment<256,T>{}( *(p + w * 0), *(p + w * 1), *(p + w * 2), 0, 0, 0, 0, 0);
            }else if( k == 4 ){
                return assignment<256,T>{}( *(p + w * 0), *(p + w * 1), *(p + w * 2), *(p + w * 3), 0, 0, 0, 0);
            }else if( k == 5 ){
                return assignment<256,T>{}( *(p + w * 0), *(p + w * 1), *(p + w * 2), *(p + w * 3), *(p + w * 4), 0, 0, 0);
            }else if( k == 6 ){
                return assignment<256,T>{}( *(p + w * 0), *(p + w * 1), *(p + w * 2), *(p + w * 3), *(p + w * 4), *(p + w * 5), 0, 0);
            }else if( k == 7 ){
                return assignment<256,T>{}( *(p + w * 0), *(p + w * 1), *(p + w * 2), *(p + w * 3), *(p + w * 4), *(p + w * 5), *(p + w * 6), 0);
            }else if( k == 8 ){
                return assignment<256,T>{}( *(p + w * 0), *(p + w * 1), *(p + w * 2), *(p + w * 3), *(p + w * 4), *(p + w * 5), *(p + w * 6), *(p + w * 7));
            }else{
                return assignment<256,T>{}(0);
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( float const* p, size_t w ) const noexcept{
            return assignment<256,T>{}( *(p + w * 0), *(p + w * 1), *(p + w * 2), *(p + w * 3), *(p + w * 4), *(p + w * 5), *(p + w * 6), *(p + w * 7));
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( double const* p ) const noexcept{
            return _mm256_loadu_pd(p);
        }
    };

    template<typename T>
    struct load<512,T>{

        using m512i = simd_type_t<512,int>;
        using m512 = simd_type_t<512,float>;
        using m512d = simd_type_t<512,double>;

        // TODO: Complete the body

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( T const* p ) const noexcept{
            return *reinterpret_cast<m512i const*>(p);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( m512i const* p ) const noexcept{
            return _mm512_loadu_si512(p);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( float const* p ) const noexcept{
            return _mm512_loadu_ps(p);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( double const* p ) const noexcept{
            return _mm512_loadu_pd(p);
        }
    };

} // namespace boost::numeric::ublas::simd::detail{


#endif // _BOOST_UBLAS_TENSOR_DETAIL_SIMD_LOAD_HPP