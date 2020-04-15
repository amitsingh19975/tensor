#ifndef _BOOST_UBLAS_TENSOR_DETAIL_SIMD_STORE_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_SIMD_STORE_HPP

#include "simd_assignment_impl.hpp"

namespace boost::numeric::ublas::simd::detail{
    
    template<typename T>
    struct store<128,T>{
        
        using m128i = simd_type_t<128,int>;
        using m128 = simd_type_t<128,float>;
        using m128d = simd_type_t<128,double>;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( T* p,  m128i const& a) const noexcept{
            if constexpr(std::is_same_v<T, int16_t> || std::is_same_v<T, short>){
                _mm_storeu_si16(p,a);
            }else if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int>){
                _mm_storeu_si32(p,a);
            }else if constexpr(std::is_same_v<T, int64_t> || std::is_same_v<T, long long>){
                _mm_storeu_si64(p,a);
            }else{
                *reinterpret_cast<m128i*>(p) = a;
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( m128i* p, m128i const& a ) const noexcept{
            _mm_storeu_si128(p, a);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( float* p, m128 const& a ) const noexcept{
            _mm_storeu_ps(p,a);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( double* p, m128d const& a ) const noexcept{
            _mm_storeu_pd(p,a);
        }
    };

    template<typename T>
    struct store<256,T>{

        using m256i = simd_type_t<256,int>;
        using m256 = simd_type_t<256,float>;
        using m256d = simd_type_t<256,double>;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( T* p,  m256i const& a) const noexcept{
            *reinterpret_cast<m256i*>(p) = a;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( m256i* p, m256i const& a ) const noexcept{
            _mm256_storeu_si256(p, a);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( float* p, m256 const& a ) const noexcept{
            _mm256_storeu_ps(p, a);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( double* p, m256d const& a ) const noexcept{
            _mm256_storeu_pd(p, a);
        }
    };

    template<typename T>
    struct store<512,T>{

        using m512i = simd_type_t<512,int>;
        using m512 = simd_type_t<512,float>;
        using m512d = simd_type_t<512,double>;

        // TODO: Complete the body


        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( T* p,  m512i const& a) const noexcept{
            *reinterpret_cast<m512i*>(p) = a;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( double* p, m512i const& a ) const noexcept{
            _mm512_storeu_si512(p, a);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( float* p, m512 const& a) const noexcept{
            _mm512_storeu_ps(p,a);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( double* p, m512d const& a) const noexcept{
            _mm512_storeu_pd(p,a);
        }
    };

} // namespace boost::numeric::ublas::simd::detail{


#endif // _BOOST_UBLAS_TENSOR_DETAIL_SIMD_LOAD_HPP