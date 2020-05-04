#if !defined(BOOST_NUMERIC_UBLAS_SIMD_MULT_ADD_4X4_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_MULT_ADD_4X4_HPP

#include <boost/numeric/ublas/tensor/simd/detail/simd_helper.hpp>

namespace boost::numeric::ublas::simd{

    template< typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void mult_add_4x4(float* c, SizeType const* nc, SizeType const* wc, 
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb) noexcept
    {
        assert( c != nullptr && a != nullptr && b!= nullptr );
        __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12,
            ymm13, ymm14, ymm15;

        size_t wc_0 = wc[0];
        size_t wc_1 = wc[1];
        
        size_t wa_0 = wa[0];
        size_t wa_1 = wa[1];
        
        size_t wb_0 = wb[0];
        size_t wb_1 = wb[1];

        assert(wc_0 == 1 && wa_0 == 1 && wb_0 == 1);
        int i = 0;
        
        float res1[8] = {0};
        float res2[8] = {0};
        float res3[8] = {0};
        float res4[8] = {0};

        
        prefetch_4( a, wa_1 );
        prefetch_4( b, wb_1 );
        prefetch_4( c, wc_1 );

        ymm0 = _mm256_load_ps(a);
        ymm1 = _mm256_load_ps(a + wa_1);
        ymm2 = _mm256_load_ps(a + wa_1 * 2);
        ymm3 = _mm256_load_ps(a + wa_1 * 3);
        
        ymm4 = _mm256_load_ps(b);
        ymm5 = _mm256_load_ps(b + wb_1);
        ymm6 = _mm256_load_ps(b + wb_1 * 2);
        ymm7 = _mm256_load_ps(b + wb_1 * 3);

        float& c00 = *( c + wc_0 * 0 + wc_1 * 0 );
        float& c01 = *( c + wc_0 * 0 + wc_1 * 1 );
        float& c02 = *( c + wc_0 * 0 + wc_1 * 2 );
        float& c03 = *( c + wc_0 * 0 + wc_1 * 3 );

        float& c10 = *( c + wc_0 * 1 + wc_1 * 0 );
        float& c11 = *( c + wc_0 * 1 + wc_1 * 1 );
        float& c12 = *( c + wc_0 * 1 + wc_1 * 2 );
        float& c13 = *( c + wc_0 * 1 + wc_1 * 3 );

        float& c20 = *( c + wc_0 * 2 + wc_1 * 0 );
        float& c21 = *( c + wc_0 * 2 + wc_1 * 1 );
        float& c22 = *( c + wc_0 * 2 + wc_1 * 2 );
        float& c23 = *( c + wc_0 * 2 + wc_1 * 3 );

        float& c30 = *( c + wc_0 * 3 + wc_1 * 0 );
        float& c31 = *( c + wc_0 * 3 + wc_1 * 1 );
        float& c32 = *( c + wc_0 * 3 + wc_1 * 2 );
        float& c33 = *( c + wc_0 * 3 + wc_1 * 3 );


        ymm12 = _mm256_dp_ps(ymm0,ymm4,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm4,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm4,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm4,0xff);
        
        _mm256_stream_ps(res1,ymm12);
        c00 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c01 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c02 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c03 += res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm0,ymm5,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm5,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm5,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm5,0xff);


        _mm256_stream_ps(res1,ymm12);
        c10 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c11 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c12 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c13 += res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm0,ymm6,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm6,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm6,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm6,0xff);


        _mm256_stream_ps(res1,ymm12);
        c20 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c21 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c22 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c23 += res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm0,ymm7,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm7,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm7,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm7,0xff);


        _mm256_stream_ps(res1,ymm12);
        c30 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c31 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c32 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c33 += res4[0] + res4[4];

    }
   
} // namespace boost::numeric::ublas::simd


#endif // BOOST_NUMERIC_UBLAS_SIMD_MULT_ADD_4X4_HPP
