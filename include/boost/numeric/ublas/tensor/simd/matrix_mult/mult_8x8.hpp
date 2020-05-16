#if !defined(BOOST_NUMERIC_UBLAS_SIMD_MULT_8X8_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_MULT_8X8_HPP

#include <boost/numeric/ublas/tensor/simd/detail/simd_helper.hpp>

namespace boost::numeric::ublas::simd{


    template< typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void mult_8x8(float* c, SizeType const* nc, SizeType const* wc, 
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb) noexcept
    {
        __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12,
            ymm13, ymm14, ymm15;

        size_t wc_0 = wc[0];
        size_t wc_1 = wc[1];
        
        size_t wa_0 = wa[0];
        size_t wa_1 = wa[1];
        
        size_t wb_0 = wb[0];
        size_t wb_1 = wb[1];

        size_t wb00 = 0;
        size_t wb01 = 1;
        size_t wb02 = 1 + wb01;
        size_t wb03 = 1 + wb02;
        size_t wb04 = 1 + wb03;
        size_t wb05 = 1 + wb04;
        size_t wb06 = 1 + wb05;
        size_t wb07 = 1 + wb06;
        auto bi = b;
        auto ci = c;

        assert(wc_0 == 1 && wa_0 == 1 && wb_0 == 1 && c != nullptr);

        ymm0 = _mm256_loadu_ps(a);
        ymm1 = _mm256_loadu_ps(a + 8);
        ymm2 = _mm256_loadu_ps(a + 8 * 2);
        ymm3 = _mm256_loadu_ps(a + 8 * 3);
        ymm4 = _mm256_loadu_ps(a + 8 * 4);
        ymm5 = _mm256_loadu_ps(a + 8 * 5);
        ymm6 = _mm256_loadu_ps(a + 8 * 6);
        ymm7 = _mm256_loadu_ps(a + 8 * 7);

        auto zero = _mm256_setzero_ps();
        for( int i = 0; i < 8; ++i ){
            ymm9 = _mm256_loadu_ps(ci);
            
            ymm8 = _mm256_broadcast_ss(bi + wb00);
            ymm8 = _mm256_fmadd_ps(ymm8,ymm0,zero);

            ymm9 = _mm256_broadcast_ss(bi + wb01);
            ymm8 = _mm256_fmadd_ps(ymm9,ymm1,ymm8);
            
            ymm10 = _mm256_broadcast_ss(bi + wb02);
            ymm8 = _mm256_fmadd_ps(ymm10,ymm2,ymm8);
            
            ymm11 = _mm256_broadcast_ss(bi + wb03);
            ymm8 = _mm256_fmadd_ps(ymm11,ymm3,ymm8);
            
            ymm12 = _mm256_broadcast_ss(bi + wb04);
            ymm8 = _mm256_fmadd_ps(ymm12,ymm4,ymm8);

            ymm13 = _mm256_broadcast_ss(bi + wb05);
            ymm8 = _mm256_fmadd_ps(ymm13,ymm5,ymm8);
            
            ymm14 = _mm256_broadcast_ss(bi + wb06);
            ymm8 = _mm256_fmadd_ps(ymm14,ymm6,ymm8);

            ymm15 = _mm256_broadcast_ss(bi + wb07);
            ymm8 = _mm256_fmadd_ps(ymm15,ymm7,ymm8);
            
            _mm256_storeu_ps(ci,ymm8);
            bi += 8, ci += 8;
        }
    }   
} // namespace boost::numeric::ublas::simd


#endif // BOOST_NUMERIC_UBLAS_SIMD_MULT_8X8_HPP
