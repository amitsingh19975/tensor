#if !defined(BOOST_NUMERIC_UBLAS_SIMD_SUB_8X8_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_SUB_8X8_HPP

#include <boost/numeric/ublas/tensor/simd/detail/simd_helper.hpp>

namespace boost::numeric::ublas::simd{
    void print(__m256 d ){
        std::cout<<d[0]<<' '<<d[1]<<' '<<d[2]<<' '<<d[3]<<' ';
        std::cout<<d[4]<<' '<<d[5]<<' '<<d[6]<<' '<<d[7]<<'\n';
    }

    #define PRINT(N) print(ymm##N);

    template<typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    void sub_8x8( float* c, SizeType const* nc, SizeType const* wc, 
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb) noexcept
    {
        assert( c != nullptr && a != nullptr && b!= nullptr );
        __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12,
            ymm13, ymm14, ymm15;

        constexpr size_t k = 0;

        float res1[8];
        
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

        _mm256_store_ps(res1,_mm256_sub_ps(ymm0,ymm8));
        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k) + wc[0] * i ) = res1[i];
        }

        ymm0 = _mm256_sub_ps(ymm1,ymm9);
        _mm256_store_ps(res1,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 1) + wc[0] * i ) = res1[i];
        }

        ymm0 = _mm256_sub_ps(ymm2,ymm10);
        _mm256_store_ps(res1,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 2) + wc[0] * i ) = res1[i];
        }

        ymm0 = _mm256_sub_ps(ymm3,ymm11);
        _mm256_store_ps(res1,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 3) + wc[0] * i ) = res1[i];
        }

        ymm0 = _mm256_sub_ps(ymm4,ymm12);
        _mm256_store_ps(res1,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 4) + wc[0] * i ) = res1[i];
        }

        ymm0 = _mm256_sub_ps(ymm5,ymm13);
        _mm256_store_ps(res1,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 5) + wc[0] * i ) = res1[i];
        }

        ymm0 = _mm256_sub_ps(ymm6,ymm14);
        _mm256_store_ps(res1,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 6) + wc[0] * i ) = res1[i];
        }

        ymm0 = _mm256_sub_ps(ymm7,ymm15);
        _mm256_store_ps(res1,ymm0);

        for( size_t i = 0; i < 8; ++i ){
            *( c + wc[1] * (k + 7) + wc[0] * i ) = res1[i];
        }

    }

} // namespace boost::numeric::ublas::simd


#endif // BOOST_NUMERIC_UBLAS_SIMD_SUB_8X8_HPP
