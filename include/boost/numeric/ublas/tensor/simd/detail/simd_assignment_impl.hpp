#ifndef _BOOST_UBLAS_TENSOR_DETAIL_SIMD_ASSIGNMENT_IMPL_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_SIMD_ASSIGNMENT_IMPL_HPP

#include "simd_helper_impl.hpp"

namespace boost::numeric::ublas::simd::detail{

    template< size_t N, typename T > 
    struct assignment{
        static_assert(is_valid_vector_size<N,T>::value, "boost::numeric::ublas::simd::detail::assignment: invalid vector size");
    };

    template< typename T > 
    struct assignment< 64, T >{
        static_assert(std::numeric_limits<T>::is_integer && std::numeric_limits<T>::is_signed, "boost::numeric::ublas::simd::detail::assignment: tensor supports only signed value currently");
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()() const noexcept{
            return _mm_setzero_si64();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int8_t a0, int8_t a1, int8_t a2, int8_t a3, int8_t a4, int8_t a5, int8_t a6, int8_t a7
        ) const noexcept{
            return _mm_setr_pi8(a0,a1,a2,a3,a4,a5,a6,a7);
        }
        

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int16_t a0, int16_t a1, int16_t a2, int16_t a3) const noexcept{
            return _mm_setr_pi16(a0,a1,a2,a3);
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int32_t a0, int32_t a1) const noexcept{
            return _mm_setr_pi32(a0,a1);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int16_t a0) const noexcept{
            return _mm_set1_pi16(a0);
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int8_t a0
        ) const noexcept{
            return _mm_set1_pi8(a0);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int32_t a0) const noexcept{
            return _mm_set1_pi32(a0);
        }

    };


    template< typename T > 
    struct assignment< 128, T >{
        static_assert(std::numeric_limits<T>::is_integer && std::numeric_limits<T>::is_signed, "boost::numeric::ublas::simd::detail::assignment: tensor supports only signed value currently");
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()() const noexcept{
            return _mm_setzero_si128();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int8_t a0, int8_t a1, int8_t a2, int8_t a3, int8_t a4, int8_t a5, int8_t a6, int8_t a7,
            int8_t a8, int8_t a9, int8_t a10, int8_t a11, int8_t a12, int8_t a13, int8_t a14, int8_t a15
        ) const noexcept{
            return _mm_setr_epi8(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15);
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int16_t a0, int16_t a1, int16_t a2, int16_t a3, int16_t a4, int16_t a5, int16_t a6, int16_t a7
        ) const noexcept{
            return _mm_setr_epi16(a0,a1,a2,a3,a4,a5,a6,a7);
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int32_t a0, int32_t a1, int32_t a2, int32_t a3) const noexcept{
            return _mm_setr_epi32(a0,a1,a2,a3);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int64_t a0, int64_t a1) const noexcept{
            return _mm_set_epi64x(a1,a0);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int8_t a0
        ) const noexcept{
            return _mm_set1_epi8(a0);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int16_t a0) const noexcept{
            return _mm_set1_epi16(a0);
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int32_t a0) const noexcept{
            return _mm_set1_epi32(a0);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int64_t a0) const noexcept{
            return _mm_set1_epi64x(a0);
        }

    };

    template< typename T > 
    struct assignment< 256, T >{
        static_assert(std::numeric_limits<T>::is_integer && std::numeric_limits<T>::is_signed, "boost::numeric::ublas::simd::detail::assignment: tensor supports only signed value currently");
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()() const noexcept{
            return _mm256_setzero_si256();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int8_t a0, int8_t a1, int8_t a2, int8_t a3, int8_t a4, int8_t a5, int8_t a6, int8_t a7,
            int8_t a8, int8_t a9, int8_t a10, int8_t a11, int8_t a12, int8_t a13, int8_t a14, int8_t a15,
            int8_t a16, int8_t a17, int8_t a18, int8_t a19, int8_t a20, int8_t a21, int8_t a22, int8_t a23,
            int8_t a24, int8_t a25, int8_t a26, int8_t a27, int8_t a28, int8_t a29, int8_t a30, int8_t a31
        ) const noexcept{
            return _mm256_setr_epi8(
                a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31
            );
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int16_t a0, int16_t a1, int16_t a2, int16_t a3, int16_t a4, int16_t a5, int16_t a6, int16_t a7,
            int16_t a8, int16_t a9, int16_t a10, int16_t a11, int16_t a12, int16_t a13, int16_t a14, int16_t a15
        ) const noexcept{
            return _mm256_setr_epi16(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15);
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int32_t a0, int32_t a1, int32_t a2, int32_t a3, int32_t a4, int32_t a5, int32_t a6, int32_t a7
        ) const noexcept{
            return _mm256_setr_epi32(a0,a1,a2,a3,a4,a5,a6,a7);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int64_t a0, int64_t a1, int64_t a2, int64_t a3) const noexcept{
            return _mm256_setr_epi64x(a0,a1,a2,a3);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int8_t a0
        ) const noexcept{
            return _mm256_set1_epi8(a0);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int16_t a0) const noexcept{
            return _mm256_set1_epi16(a0);
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int32_t a0) const noexcept{
            return _mm256_set1_epi32(a0);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int64_t a0) const noexcept{
            return _mm256_set1_epi64x(a0);
        }
    };
    
    template<typename T>
    struct assignment<512,T>{
        
        static_assert(std::numeric_limits<T>::is_integer && std::numeric_limits<T>::is_signed, "boost::numeric::ublas::simd::detail::assignment: tensor supports only signed value currently");
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()() const noexcept{
            return _mm512_setzero_si512();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int8_t a0, int8_t a1, int8_t a2, int8_t a3, int8_t a4, int8_t a5, int8_t a6, int8_t a7,
            int8_t a8, int8_t a9, int8_t a10, int8_t a11, int8_t a12, int8_t a13, int8_t a14, int8_t a15,
            int8_t a16, int8_t a17, int8_t a18, int8_t a19, int8_t a20, int8_t a21, int8_t a22, int8_t a23,
            int8_t a24, int8_t a25, int8_t a26, int8_t a27, int8_t a28, int8_t a29, int8_t a30, int8_t a31,
            int8_t a32, int8_t a33, int8_t a34, int8_t a35, int8_t a36, int8_t a37, int8_t a38, int8_t a39,
            int8_t a40, int8_t a41, int8_t a42, int8_t a43, int8_t a44, int8_t a45, int8_t a46, int8_t a47,
            int8_t a48, int8_t a49, int8_t a50, int8_t a51, int8_t a52, int8_t a53, int8_t a54, int8_t a55,
            int8_t a56, int8_t a57, int8_t a58, int8_t a59, int8_t a60, int8_t a61, int8_t a62, int8_t a63
        ) const noexcept{
            return _mm512_set_epi8(
                a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,
                a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63
            );
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int16_t a0, int16_t a1, int16_t a2, int16_t a3, int16_t a4, int16_t a5, int16_t a6, int16_t a7,
            int16_t a8, int16_t a9, int16_t a10, int16_t a11, int16_t a12, int16_t a13, int16_t a14, int16_t a15,
            int16_t a16, int16_t a17, int16_t a18, int16_t a19, int16_t a20, int16_t a21, int16_t a22, int16_t a23,
            int16_t a24, int16_t a25, int16_t a26, int16_t a27, int16_t a28, int16_t a29, int16_t a30, int16_t a31
        ) const noexcept{
            return _mm512_set_epi16(
                a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31
            );
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int32_t a0, int32_t a1, int32_t a2, int32_t a3, int32_t a4, int32_t a5, int32_t a6, int32_t a7,
            int32_t a8, int32_t a9, int32_t a10, int32_t a11, int32_t a12, int32_t a13, int32_t a14, int32_t a15
        ) const noexcept{
            return _mm512_set_epi32(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7
        ) const noexcept{
            return _mm512_set_epi64(a0,a1,a2,a3,a4,a5,a6,a7);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            int8_t a0
        ) const noexcept{
            return _mm512_set1_epi8(a0);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int16_t a0) const noexcept{
            return _mm512_set1_epi16(a0);
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int32_t a0) const noexcept{
            return _mm512_set1_epi32(a0);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(int64_t a0) const noexcept{
            return _mm512_set1_epi64(a0);
        }
    };

    template<>
    struct assignment<128, float>{
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()() const noexcept{
            return _mm_setzero_ps();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            float a0, float a1, float a2, float a3
        ) const noexcept{
            return _mm_setr_ps(a0,a1,a2,a3);
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            float a0
        ) const noexcept{
            return _mm_set1_ps(a0);
        }
    };

    template<>
    struct assignment<256, float>{
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()() const noexcept{
            return _mm256_setzero_ps();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            float a0, float a1, float a2, float a3,
            float a4, float a5, float a6, float a7
        ) const noexcept{
            return _mm256_setr_ps(
                a0,a1,a2,a3,a4,a5,a6,a7
            );
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            float a0
        ) const noexcept{
            return _mm256_set1_ps(a0);
        }
    };

    template<>
    struct assignment<512, float>{
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()() const noexcept{
            return _mm512_setzero_ps();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            float a0, float a1, float a2, float a3,
            float a4, float a5, float a6, float a7,
            float a8, float a9, float a10, float a11,
            float a12, float a13, float a14, float a15
        ) const noexcept{
            return _mm512_setr_ps(
                a0,a1,a2,a3,a4,a5,a6,a7,
                a8,a9,a10,a11,a12,a13,a14,a15
            );
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            float a0
        ) const noexcept{
            return _mm512_set1_ps(a0);
        }
    };

    template<>
    struct assignment<128, double>{
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()() const noexcept{
            return _mm_setzero_pd();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            double a0, double a1
        ) const noexcept{
            return _mm_setr_pd(a0,a1);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            double a0
        ) const noexcept{
            return _mm_set1_pd(a0);
        }
    };

    template<>
    struct assignment<256, double>{
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()() const noexcept{
            return _mm256_setzero_pd();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            double a0, double a1, double a2, double a3
        ) const noexcept{
            return _mm256_setr_pd(
                a0,a1,a2,a3
            );
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            double a0
        ) const noexcept{
            return _mm256_set1_pd(
                a0
            );
        }
        
    };

    template<>
    struct assignment<512, double>{

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()() const noexcept{
            return _mm512_setzero_pd();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            double a0, double a1, double a2, double a3,
            double a4, double a5, double a6, double a7
        ) const noexcept{
            return _mm512_setr_pd(
                a0,a1,a2,a3,a4,a5,a6,a7
            );
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(
            double a0
        ) const noexcept{
            return _mm512_set1_pd(
                a0
            );
        }
    };
    

} // namespace boost::numeric::ublas::detail

#endif 
