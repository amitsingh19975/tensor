#if !defined(BOOST_NUMERIC_UBLAS_SIMD_ADD_8X8_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_ADD_8X8_HPP

#include <boost/numeric/ublas/tensor/simd/detail/simd_helper.hpp>

namespace boost::numeric::ublas::simd{
    
    template<typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    void add_8x8( float* c, SizeType const* nc, SizeType const* wc, 
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb) noexcept
    {
        assert( c != nullptr && a != nullptr && b!= nullptr );
        __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12,
            ymm13, ymm14, ymm15;

        constexpr size_t k = 0;

        static float res[8];
        
        ymm0 = _mm256_load_ps(a);
        ymm1 = _mm256_load_ps(a + wa[1]);
        ymm2 = _mm256_load_ps(a + wa[1] * 2);
        ymm3 = _mm256_load_ps(a + wa[1] * 3);
        ymm4 = _mm256_load_ps(a + wa[1] * 4);
        ymm5 = _mm256_load_ps(a + wa[1] * 5);
        ymm6 = _mm256_load_ps(a + wa[1] * 6);
        ymm7 = _mm256_load_ps(a + wa[1] * 7);
        
        ymm8 = _mm256_load_ps(b);
        ymm9 = _mm256_load_ps(b + wb[1]);
        ymm10 = _mm256_load_ps(b + wb[1] * 2);
        ymm11 = _mm256_load_ps(b + wb[1] * 3);
        ymm12 = _mm256_load_ps(b + wb[1] * 4);
        ymm13 = _mm256_load_ps(b + wb[1] * 5);
        ymm14 = _mm256_load_ps(b + wb[1] * 6);
        ymm15 = _mm256_load_ps(b + wb[1] * 7);

        _mm256_store_ps(res,_mm256_add_ps(ymm0,ymm8));
        
        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k) + wc[0] * i ) = res[i];
        }

        ymm0 = _mm256_add_ps(ymm1,ymm9);
        _mm256_store_ps(res,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 1) + wc[0] * i ) = res[i];
        }

        ymm0 = _mm256_add_ps(ymm2,ymm10);
        _mm256_store_ps(res,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 2) + wc[0] * i ) = res[i];
        }

        ymm0 = _mm256_add_ps(ymm3,ymm11);
        _mm256_store_ps(res,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 3) + wc[0] * i ) = res[i];
        }

        ymm0 = _mm256_add_ps(ymm4,ymm12);
        _mm256_store_ps(res,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 4) + wc[0] * i ) = res[i];
        }

        ymm0 = _mm256_add_ps(ymm5,ymm13);
        _mm256_store_ps(res,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 5) + wc[0] * i ) = res[i];
        }

        ymm0 = _mm256_add_ps(ymm6,ymm14);
        _mm256_store_ps(res,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 6) + wc[0] * i ) = res[i];
        }

        ymm0 = _mm256_add_ps(ymm7,ymm15);
        _mm256_store_ps(res,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 7) + wc[0] * i ) = res[i];
        }

    }

} // namespace boost::numeric::ublas::simd


#endif // BOOST_NUMERIC_UBLAS_SIMD_ADD_8X8_HPP
