#if !defined(BOOST_NUMERIC_UBLAS_SIMD_MULT_8X8_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_MULT_8X8_HPP

#include <boost/numeric/ublas/tensor/simd/detail/simd_helper.hpp>

void print(__m256 d ){
    std::cout<<d[0]<<' '<<d[1]<<' '<<d[2]<<' '<<d[3]<<' ';
    std::cout<<d[4]<<' '<<d[5]<<' '<<d[6]<<' '<<d[7]<<'\n';
}

#define PRINT(N) print(ymm##N);

namespace boost::numeric::ublas::simd{
   
    template< typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void mult_8x8(float* c, SizeType const* nc, SizeType const* wc, 
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb) noexcept
    {
        assert( c != nullptr && a != nullptr && b!= nullptr );
        __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12,
            ymm13, ymm14, ymm15;

        assert(wc[0] == 1 && wa[0] == 1 && wb[0] == 1);
        int i = 0;
        
        prefetch( a, wa[1] );
        prefetch( b, wb[1] );
        prefetch( c, wc[1] );

        float res1[8] = {0};
        float res2[8] = {0};
        float res3[8] = {0};
        float res4[8] = {0};
        
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
        // ----------------------- row 0 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm8,0xff);
        
        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 0 + wc[1] * 0 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 0 + wc[1] * 1 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 0 + wc[1] * 2 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 0 + wc[1] * 3 ) = res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm8,0xff);


        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 0 + wc[1] * 4 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 0 + wc[1] * 5 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 0 + wc[1] * 6 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 0 + wc[1] * 7 ) = res4[0] + res4[4];

        
        // ----------------------- row 1 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm9,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm9,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm9,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 1 + wc[1] * 0 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 1 + wc[1] * 1 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 1 + wc[1] * 2 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 1 + wc[1] * 3 ) = res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm9,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm9,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 1 + wc[1] * 4 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 1 + wc[1] * 5 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 1 + wc[1] * 6 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 1 + wc[1] * 7 ) = res4[0] + res4[4];

        
        // ----------------------- row 2 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm10,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 2 + wc[1] * 0 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 2 + wc[1] * 1 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 2 + wc[1] * 2 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 2 + wc[1] * 3 ) = res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm10,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 2 + wc[1] * 4 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 2 + wc[1] * 5 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 2 + wc[1] * 6 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 2 + wc[1] * 7 ) = res4[0] + res4[4];

        
        // ----------------------- row 3 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm11,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm11,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm11,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 3 + wc[1] * 0 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 3 + wc[1] * 1 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 3 + wc[1] * 2 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 3 + wc[1] * 3 ) = res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm11,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm11,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 3 + wc[1] * 4 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 3 + wc[1] * 5 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 3 + wc[1] * 6 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 3 + wc[1] * 7 ) = res4[0] + res4[4];

        
        // -------------------------------------------------------- //

        ymm8 = _mm256_load_ps(b + wb[1] * 4);
        ymm9 = _mm256_load_ps(b + wb[1] * 5);
        ymm10 = _mm256_load_ps(b + wb[1] * 6);
        ymm11 = _mm256_load_ps(b + wb[1] * 7);

        // ----------------------- row 4 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm8,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 4 + wc[1] * 0 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 4 + wc[1] * 1 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 4 + wc[1] * 2 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 4 + wc[1] * 3 ) = res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm8,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 4 + wc[1] * 4 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 4 + wc[1] * 5 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 4 + wc[1] * 6 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 4 + wc[1] * 7 ) = res4[0] + res4[4];

        
        // ----------------------- row 5 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm9,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm9,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm9,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 5 + wc[1] * 0 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 5 + wc[1] * 1 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 5 + wc[1] * 2 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 5 + wc[1] * 3 ) = res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm9,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm9,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 5 + wc[1] * 4 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 5 + wc[1] * 5 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 5 + wc[1] * 6 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 5 + wc[1] * 7 ) = res4[0] + res4[4];

        
        // ----------------------- row 6 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm10,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 6 + wc[1] * 0 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 6 + wc[1] * 1 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 6 + wc[1] * 2 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 6 + wc[1] * 3 ) = res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm10,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 6 + wc[1] * 4 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 6 + wc[1] * 5 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 6 + wc[1] * 6 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 6 + wc[1] * 7 ) = res4[0] + res4[4];

        
        // ----------------------- row 7 ---------------------------- //
        ymm12 = _mm256_dp_ps(ymm0,ymm11,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm14 = _mm256_dp_ps(ymm2,ymm11,0xff);
        ymm15 = _mm256_dp_ps(ymm3,ymm11,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 7 + wc[1] * 0 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 7 + wc[1] * 1 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 7 + wc[1] * 2 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 7 + wc[1] * 3 ) = res4[0] + res4[4];


        ymm12 = _mm256_dp_ps(ymm4,ymm11,0xff);
        ymm13 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm14 = _mm256_dp_ps(ymm6,ymm11,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);

        _mm256_stream_ps(res1,ymm12);
        *( c + wc[0] * 7 + wc[1] * 4 ) = res1[0] + res1[4];

        _mm256_stream_ps(res2,ymm13);
        *( c + wc[0] * 7 + wc[1] * 5 ) = res2[0] + res2[4];

        _mm256_stream_ps(res3,ymm14);
        *( c + wc[0] * 7 + wc[1] * 6 ) = res3[0] + res3[4];

        _mm256_stream_ps(res4,ymm15);
        *( c + wc[0] * 7 + wc[1] * 7 ) = res4[0] + res4[4];

    }

    template< typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void mult_8x8_asm(float* c, SizeType const* nc, SizeType const* wc, 
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb) noexcept
    {

        __asm__ volatile(R"(

            movq %[ptr_a], %%r10
            movq %[ptr_b], %%r11
            movq %[ptr_c], %%r12

            # loading a
            vmovups (%%r10), %%ymm0
            vmovups (%%r10, %[wa], 4), %%ymm1
            vmovups (%%r10, %[wa], 8), %%ymm2
            leaq (%%r10, %[wa], 8), %%r10
            leaq (%%r10, %[wa], 4), %%r10

            vmovups (%%r10), %%ymm3
            vmovups (%%r10, %[wa], 4), %%ymm4
            vmovups (%%r10, %[wa], 8), %%ymm5
            leaq (%%r10, %[wa], 8), %%r10

            vmovups (%%r10, %[wa], 4), %%ymm6
            vmovups (%%r10, %[wa], 8), %%ymm7

            vmovups (%%r11), %%ymm8
            vmovups (%%r11, %[wb], 4), %%ymm9
            vmovups (%%r11, %[wb], 8), %%ymm10
            leaq (%%r11, %[wb], 8), %%r11
            leaq (%%r11, %[wb], 4), %%r11

            vmovups (%%r11), %%ymm11

            # Iteration 1
            movq %%r12, %%r13

            vdpps $255, %%ymm0, %%ymm8, %%ymm12
            vdpps $255, %%ymm1, %%ymm8, %%ymm13
            vdpps $255, %%ymm2, %%ymm8, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm3, %%ymm8, %%ymm12
            vdpps $255, %%ymm4, %%ymm8, %%ymm13
            vdpps $255, %%ymm5, %%ymm8, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm6, %%ymm8, %%ymm12
            vdpps $255, %%ymm7, %%ymm8, %%ymm13

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)
            leaq (%%r13, %[wc_1], 8), %%r13
            
            leaq (%%r12, %[wc_0], 4), %%r12

            # Iteration 2
            movq %%r12, %%r13

            vdpps $255, %%ymm0, %%ymm9, %%ymm12
            vdpps $255, %%ymm1, %%ymm9, %%ymm13
            vdpps $255, %%ymm2, %%ymm9, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm3, %%ymm9, %%ymm12
            vdpps $255, %%ymm4, %%ymm9, %%ymm13
            vdpps $255, %%ymm5, %%ymm9, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm6, %%ymm9, %%ymm12
            vdpps $255, %%ymm7, %%ymm9, %%ymm13

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)
            leaq (%%r13, %[wc_1], 8), %%r13

            leaq (%%r12, %[wc_0], 4), %%r12

            # Iteration 3
            movq %%r12, %%r13

            vdpps $255, %%ymm0, %%ymm10, %%ymm12
            vdpps $255, %%ymm1, %%ymm10, %%ymm13
            vdpps $255, %%ymm2, %%ymm10, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm3, %%ymm10, %%ymm12
            vdpps $255, %%ymm4, %%ymm10, %%ymm13
            vdpps $255, %%ymm5, %%ymm10, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm6, %%ymm10, %%ymm12
            vdpps $255, %%ymm7, %%ymm10, %%ymm13

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)
            leaq (%%r13, %[wc_1], 8), %%r13

            leaq (%%r12, %[wc_0], 4), %%r12

            # load next 3 row b

            vmovups (%%r11), %%ymm8
            vmovups (%%r11, %[wb], 4), %%ymm9
            vmovups (%%r11, %[wb], 8), %%ymm10
            leaq (%%r11, %[wb], 8), %%r11


            # Iteration 4
            movq %%r12, %%r13

            vdpps $255, %%ymm0, %%ymm8, %%ymm12
            vdpps $255, %%ymm1, %%ymm8, %%ymm13
            vdpps $255, %%ymm2, %%ymm8, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm3, %%ymm8, %%ymm12
            vdpps $255, %%ymm4, %%ymm8, %%ymm13
            vdpps $255, %%ymm5, %%ymm8, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm6, %%ymm8, %%ymm12
            vdpps $255, %%ymm7, %%ymm8, %%ymm13

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)
            leaq (%%r13, %[wc_1], 8), %%r13
            
            leaq (%%r12, %[wc_0], 4), %%r12

            # Iteration 5
            movq %%r12, %%r13

            vdpps $255, %%ymm0, %%ymm9, %%ymm12
            vdpps $255, %%ymm1, %%ymm9, %%ymm13
            vdpps $255, %%ymm2, %%ymm9, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm3, %%ymm9, %%ymm12
            vdpps $255, %%ymm4, %%ymm9, %%ymm13
            vdpps $255, %%ymm5, %%ymm9, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm6, %%ymm9, %%ymm12
            vdpps $255, %%ymm7, %%ymm9, %%ymm13

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)
            leaq (%%r13, %[wc_1], 8), %%r13

            leaq (%%r12, %[wc_0], 4), %%r12

            # Iteration 6
            movq %%r12, %%r13

            vdpps $255, %%ymm0, %%ymm10, %%ymm12
            vdpps $255, %%ymm1, %%ymm10, %%ymm13
            vdpps $255, %%ymm2, %%ymm10, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm3, %%ymm10, %%ymm12
            vdpps $255, %%ymm4, %%ymm10, %%ymm13
            vdpps $255, %%ymm5, %%ymm10, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm6, %%ymm10, %%ymm12
            vdpps $255, %%ymm7, %%ymm10, %%ymm13

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)
            leaq (%%r13, %[wc_1], 8), %%r13

            leaq (%%r12, %[wc_0], 4), %%r12

            # load next 2 row of b

            vmovups (%%r11, %[wb], 4), %%ymm8
            vmovups (%%r11, %[wb], 8), %%ymm9

            # Iteration 6
            movq %%r12, %%r13

            vdpps $255, %%ymm0, %%ymm8, %%ymm12
            vdpps $255, %%ymm1, %%ymm8, %%ymm13
            vdpps $255, %%ymm2, %%ymm8, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm3, %%ymm8, %%ymm12
            vdpps $255, %%ymm4, %%ymm8, %%ymm13
            vdpps $255, %%ymm5, %%ymm8, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm6, %%ymm8, %%ymm12
            vdpps $255, %%ymm7, %%ymm8, %%ymm13

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)
            leaq (%%r13, %[wc_1], 8), %%r13
            
            leaq (%%r12, %[wc_0], 4), %%r12

            # Iteration 7
            movq %%r12, %%r13

            vdpps $255, %%ymm0, %%ymm9, %%ymm12
            vdpps $255, %%ymm1, %%ymm9, %%ymm13
            vdpps $255, %%ymm2, %%ymm9, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm3, %%ymm9, %%ymm12
            vdpps $255, %%ymm4, %%ymm9, %%ymm13
            vdpps $255, %%ymm5, %%ymm9, %%ymm14

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)

            vextractf128 $1, %%ymm14, %%xmm15
            addps %%xmm14, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],8)
            leaq (%%r13, %[wc_1], 8), %%r13
            leaq (%%r13, %[wc_1], 4), %%r13

            vdpps $255, %%ymm6, %%ymm9, %%ymm12
            vdpps $255, %%ymm7, %%ymm9, %%ymm13

            vextractf128 $1, %%ymm12, %%xmm15
            addps %%xmm12, %%xmm15
            extractps $1, %%xmm15, (%%r13)

            vextractf128 $1, %%ymm13, %%xmm15
            addps %%xmm13, %%xmm15
            extractps $1, %%xmm15, (%%r13,%[wc_1],4)
            leaq (%%r13, %[wc_1], 8), %%r13
        )"
            : [ptr_c] "=rm" (c)
            : [ptr_a] "m" (a)
            , [ptr_b] "m" (b)
            , [wa] "r" (wa[1])
            , [wb] "r" (wb[1])
            , [wc_1] "r" (wc[1])
            , [wc_0] "r" (wc[0])
            : "memory"
            , "%ymm0", "%ymm1", "%ymm2", "%ymm3"
            , "%ymm4", "%ymm5", "%ymm6", "%ymm7"
            , "%ymm8", "%ymm9", "%ymm10", "%ymm11"
            , "%ymm12", "%ymm13", "%ymm14", "%ymm15"
            , "%xmm12", "%xmm13", "%xmm14", "%xmm15"
            , "%r10", "%r11", "%r12", "%r13"
        );

    }
   
} // namespace boost::numeric::ublas::simd


#endif // BOOST_NUMERIC_UBLAS_SIMD_MULT_8X8_HPP
