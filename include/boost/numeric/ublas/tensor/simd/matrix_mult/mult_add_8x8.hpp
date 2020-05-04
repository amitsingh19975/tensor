#if !defined(BOOST_NUMERIC_UBLAS_SIMD_MULT_ADD_8X8_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_MULT_ADD_8X8_HPP

#include <boost/numeric/ublas/tensor/simd/detail/simd_helper.hpp>

namespace boost::numeric::ublas::simd{

    template< typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void mult_add_8x8(float* c, SizeType const* nc, SizeType const* wc, 
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

        
        prefetch( a, wa_1 );
        prefetch( b, wb_1 );
        prefetch( c, wc_1 );

        ymm0 = _mm256_load_ps(a);
        ymm1 = _mm256_load_ps(a + wa_1);
        ymm2 = _mm256_load_ps(a + wa_1 * 2);
        ymm3 = _mm256_load_ps(a + wa_1 * 3);
        ymm4 = _mm256_load_ps(a + wa_1 * 4);
        ymm5 = _mm256_load_ps(a + wa_1 * 5);
        ymm6 = _mm256_load_ps(a + wa_1 * 6);
        ymm7 = _mm256_load_ps(a + wa_1 * 7);
        
        ymm8 = _mm256_load_ps(b);
        ymm9 = _mm256_load_ps(b + wb_1);
        ymm10 = _mm256_load_ps(b + wb_1 * 2);
        ymm11 = _mm256_load_ps(b + wb_1 * 3);

        float& c00 = *( c + wc_0 * 0 + wc_1 * 0 );
        float& c01 = *( c + wc_0 * 0 + wc_1 * 1 );
        float& c02 = *( c + wc_0 * 0 + wc_1 * 2 );
        float& c03 = *( c + wc_0 * 0 + wc_1 * 3 );
        float& c04 = *( c + wc_0 * 0 + wc_1 * 4 );
        float& c05 = *( c + wc_0 * 0 + wc_1 * 5 );
        float& c06 = *( c + wc_0 * 0 + wc_1 * 6 );
        float& c07 = *( c + wc_0 * 0 + wc_1 * 7 );

        float& c10 = *( c + wc_0 * 1 + wc_1 * 0 );
        float& c11 = *( c + wc_0 * 1 + wc_1 * 1 );
        float& c12 = *( c + wc_0 * 1 + wc_1 * 2 );
        float& c13 = *( c + wc_0 * 1 + wc_1 * 3 );
        float& c14 = *( c + wc_0 * 1 + wc_1 * 4 );
        float& c15 = *( c + wc_0 * 1 + wc_1 * 5 );
        float& c16 = *( c + wc_0 * 1 + wc_1 * 6 );
        float& c17 = *( c + wc_0 * 1 + wc_1 * 7 );

        float& c20 = *( c + wc_0 * 2 + wc_1 * 0 );
        float& c21 = *( c + wc_0 * 2 + wc_1 * 1 );
        float& c22 = *( c + wc_0 * 2 + wc_1 * 2 );
        float& c23 = *( c + wc_0 * 2 + wc_1 * 3 );
        float& c24 = *( c + wc_0 * 2 + wc_1 * 4 );
        float& c25 = *( c + wc_0 * 2 + wc_1 * 5 );
        float& c26 = *( c + wc_0 * 2 + wc_1 * 6 );
        float& c27 = *( c + wc_0 * 2 + wc_1 * 7 );

        float& c30 = *( c + wc_0 * 3 + wc_1 * 0 );
        float& c31 = *( c + wc_0 * 3 + wc_1 * 1 );
        float& c32 = *( c + wc_0 * 3 + wc_1 * 2 );
        float& c33 = *( c + wc_0 * 3 + wc_1 * 3 );
        float& c34 = *( c + wc_0 * 3 + wc_1 * 4 );
        float& c35 = *( c + wc_0 * 3 + wc_1 * 5 );
        float& c36 = *( c + wc_0 * 3 + wc_1 * 6 );
        float& c37 = *( c + wc_0 * 3 + wc_1 * 7 );

        float& c40 = *( c + wc_0 * 4 + wc_1 * 0 );
        float& c41 = *( c + wc_0 * 4 + wc_1 * 1 );
        float& c42 = *( c + wc_0 * 4 + wc_1 * 2 );
        float& c43 = *( c + wc_0 * 4 + wc_1 * 3 );
        float& c44 = *( c + wc_0 * 4 + wc_1 * 4 );
        float& c45 = *( c + wc_0 * 4 + wc_1 * 5 );
        float& c46 = *( c + wc_0 * 4 + wc_1 * 6 );
        float& c47 = *( c + wc_0 * 4 + wc_1 * 7 );

        float& c50 = *( c + wc_0 * 5 + wc_1 * 0 );
        float& c51 = *( c + wc_0 * 5 + wc_1 * 1 );
        float& c52 = *( c + wc_0 * 5 + wc_1 * 2 );
        float& c53 = *( c + wc_0 * 5 + wc_1 * 3 );
        float& c54 = *( c + wc_0 * 5 + wc_1 * 4 );
        float& c55 = *( c + wc_0 * 5 + wc_1 * 5 );
        float& c56 = *( c + wc_0 * 5 + wc_1 * 6 );
        float& c57 = *( c + wc_0 * 5 + wc_1 * 7 );

        float& c60 = *( c + wc_0 * 6 + wc_1 * 0 );
        float& c61 = *( c + wc_0 * 6 + wc_1 * 1 );
        float& c62 = *( c + wc_0 * 6 + wc_1 * 2 );
        float& c63 = *( c + wc_0 * 6 + wc_1 * 3 );
        float& c64 = *( c + wc_0 * 6 + wc_1 * 4 );
        float& c65 = *( c + wc_0 * 6 + wc_1 * 5 );
        float& c66 = *( c + wc_0 * 6 + wc_1 * 6 );
        float& c67 = *( c + wc_0 * 6 + wc_1 * 7 );

        float& c70 = *( c + wc_0 * 7 + wc_1 * 0 );
        float& c71 = *( c + wc_0 * 7 + wc_1 * 1 );
        float& c72 = *( c + wc_0 * 7 + wc_1 * 2 );
        float& c73 = *( c + wc_0 * 7 + wc_1 * 3 );
        float& c74 = *( c + wc_0 * 7 + wc_1 * 4 );
        float& c75 = *( c + wc_0 * 7 + wc_1 * 5 );
        float& c76 = *( c + wc_0 * 7 + wc_1 * 6 );
        float& c77 = *( c + wc_0 * 7 + wc_1 * 7 );

        // ----------------------- row 0 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm8,0xff);
        
        _mm256_stream_ps(res1,ymm12);
        c00 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c01 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c02 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c03 += res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm8,0xff);


        _mm256_stream_ps(res1,ymm12);
        c04 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c05 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c06 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c07 += res4[0] + res4[4];

        
        // ----------------------- row 1 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm9,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm9,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm9,0xff);

        _mm256_stream_ps(res1,ymm12);
        c10 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c11 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c12 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c13 += res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm9,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm9,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);

        _mm256_stream_ps(res1,ymm12);
        c14 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c15 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c16 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c17 += res4[0] + res4[4];

        
        // ----------------------- row 2 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm10,0xff);

        _mm256_stream_ps(res1,ymm12);
        c20 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c21 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c22 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c23 += res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm10,0xff);

        _mm256_stream_ps(res1,ymm12);
        c24 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c25 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c26 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c27 += res4[0] + res4[4];

        
        // ----------------------- row 3 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm11,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm11,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm11,0xff);

        _mm256_stream_ps(res1,ymm12);
        c30 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c31 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c32 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c33 += res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm11,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm11,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);

        _mm256_stream_ps(res1,ymm12);
        c34 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c35 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c36 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c37 += res4[0] + res4[4];

        // -------------------------------------------------------- //

        ymm8 = _mm256_load_ps(b + wb_1 * 4);
        ymm9 = _mm256_load_ps(b + wb_1 * 5);
        ymm10 = _mm256_load_ps(b + wb_1 * 6);
        ymm11 = _mm256_load_ps(b + wb_1 * 7);

        // ----------------------- row 4 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm8,0xff);

        _mm256_stream_ps(res1,ymm12);
        c40 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c41 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c42 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c43 += res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm8,0xff);

        _mm256_stream_ps(res1,ymm12);
        c44 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c45 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c46 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c47 += res4[0] + res4[4];

        
        // ----------------------- row 5 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm9,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm9,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm9,0xff);

        _mm256_stream_ps(res1,ymm12);
        c50 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c51 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c52 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c53 += res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm9,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm9,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);

        _mm256_stream_ps(res1,ymm12);
        c54 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c55 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c56 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c57 += res4[0] + res4[4];

        
        // ----------------------- row 6 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm10,0xff);

        _mm256_stream_ps(res1,ymm12);
        c60 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c61 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c62 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c63 += res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm10,0xff);

        _mm256_stream_ps(res1,ymm12);
        c64 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c65 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c66 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c67 += res4[0] + res4[4];

        
        // ----------------------- row 7 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm11,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm11,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm11,0xff);

        _mm256_stream_ps(res1,ymm12);
        c70 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c71 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c72 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c73 += res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm11,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm11,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);

        _mm256_stream_ps(res1,ymm12);
        c74 += res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        c75 += res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        c76 += res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        c77 += res4[0] + res4[4];

    }
   
} // namespace boost::numeric::ublas::simd


#endif // BOOST_NUMERIC_UBLAS_SIMD_MULT_ADD_8X8_HPP
