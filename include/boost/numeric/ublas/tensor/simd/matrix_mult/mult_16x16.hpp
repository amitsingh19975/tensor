#if !defined(BOOST_NUMERIC_UBLAS_SIMD_MULT_16X16_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_MULT_16X16_HPP

#include <boost/numeric/ublas/tensor/simd/detail/simd_helper.hpp>
#include <boost/numeric/ublas/tensor/simd/matrix_mult/mult_8x8.hpp>
#include <boost/numeric/ublas/tensor/simd/matrix_add/add.hpp>
#include <boost/numeric/ublas/tensor/simd/matrix_sub/sub.hpp>

namespace boost::numeric::ublas::simd{

    // BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    // void mult_16x16(float* res, float const * a, float const * b, size_t row, size_t col, size_t wc, size_t wr, size_t wra, size_t wca, size_t wrb, size_t wcb) noexcept{
    //     assert( res != nullptr && a != nullptr && b!= nullptr );

    //     constexpr size_t block_size = 8;

    //     if( row == col && row == 8 ){
    //         mult_8x8(res,a,b,1,wr,1,wra);
    //         return;
    //     }

    //     static std::array<float, block_size * block_size> temp1;
    //     static std::array<float, block_size * block_size> temp2;
    //     static std::array<float, block_size * block_size> p1;
    //     static std::array<float, block_size * block_size> p2;
    //     static std::array<float, block_size * block_size> p3;
    //     static std::array<float, block_size * block_size> p4;
    //     static std::array<float, block_size * block_size> p5;
    //     static std::array<float, block_size * block_size> p6;
    //     static std::array<float, block_size * block_size> p7;
        
    //     auto ptemp1 = temp1.data();
    //     auto ptemp2 = temp2.data();
    //     auto pp1 = p1.data();
    //     auto pp2 = p2.data();
    //     auto pp3 = p3.data();
    //     auto pp4 = p4.data();
    //     auto pp5 = p5.data();
    //     auto pp6 = p6.data();
    //     auto pp7 = p7.data();

    //     // First Matrix
    //     float const* A = a;
    //     float const* B = a + wa[1] * 8;
    //     float const* C = a + wc * 8;
    //     float const* D = a + wc * 8 + wr * 8;

    //     // Second Matrix
    //     float const* E = b;
    //     float const* F = b + wb[1] * 8;
    //     float const* G = b + wc * 8;
    //     float const* H = b + wc * 8 + wr * 8;

    //     // Res Matrix
    //     float* C11 = res;
    //     float* C12 = c + wc[0] * 8;
    //     float* C21 = res + wr * 8;
    //     float* C22 = c + wc[0] * 8 + wc[1] * 8;

    //     sub_8x8(ptemp1,F,H, wc, wr, 1, block_size, wc, wr);
    //     mult_16x16(pp1, A, ptemp1, row / 2, col / 2, wc, wr, wc, wr, 1, block_size);
    //     // print(pp1,1,wr);
    //     // exit(0);
    //     add_8x8(ptemp1,A,B,wc,wr, 1, block_size, wc, wr);
    //     mult_16x16(pp2, ptemp1, H, row / 2, col / 2, wc, wr,  1, block_size, wc, wr);

    //     add_8x8(ptemp1,C,D,1,wr, 1, block_size, wc, wr);
    //     mult_16x16(pp3, ptemp1, E, row / 2, col / 2, wc, wr, 1, block_size, wc, wr);

    //     sub_8x8(ptemp1,G,E,1,wr, 1, block_size, wc, wr);
    //     mult_16x16(pp4, D, ptemp1, row / 2, col / 2, wc, wr, wc, wr, 1, block_size);

    //     add_8x8(ptemp1,A,D,1,wr, 1, block_size, wc, wr);
    //     add_8x8(ptemp2,E,H,1,wr, 1, block_size, wc, wr);
    //     mult_16x16(pp5, ptemp1, ptemp2, row / 2, col / 2, wc, wr, 1, block_size, 1, block_size);

    //     sub_8x8(ptemp1,B,D,1,wr, 1, block_size, wc, wr);
    //     add_8x8(ptemp2,G,H,1,wr, 1, block_size, wc, wr);
    //     mult_16x16(pp6, ptemp1, ptemp2, row / 2, col / 2, wc, wr, 1, block_size, 1, block_size);

    //     sub_8x8(ptemp1,A,C,1,wr, 1, block_size, wc, wr);
    //     add_8x8(ptemp2,E,F,1,wr, 1, block_size, wc, wr);
    //     mult_16x16(pp6, ptemp1, ptemp2, row / 2, col / 2, wc, wr, 1, block_size, 1, block_size);


    //     add_8x8(ptemp1,pp5,pp4,1,block_size, 1, block_size, 1, block_size);
    //     sub_8x8(ptemp2,pp6,pp2,1,block_size, 1, block_size, 1, block_size);
    //     add_8x8(pp6,ptemp1,ptemp2,1,block_size, 1, block_size, 1, block_size);
        
    //     auto ci = C11;
    //     auto pi = pp6;
    //     for( size_t i = 0; i < block_size; ++i, ci += wr, pi += 1 ){
    //         auto cj = ci;
    //         auto pj = pi;
    //         for(size_t j = 0; j < block_size; ++j, cj += wc, pj += wr){
    //             *cj = * pj;
    //         }
    //     }
        
    //     add_8x8(ptemp1,pp1,pp2,1,block_size, 1, block_size, 1, block_size);
    //     ci = C12;
    //     pi = ptemp1;
    //     for( size_t i = 0; i < block_size; ++i, ci += wr, pi += 1 ){
    //         auto cj = ci;
    //         auto pj = pi;
    //         for(size_t j = 0; j < block_size; ++j, cj += wc, pj += wr){
    //             *cj = * pj;
    //         }
    //     }

    //     add_8x8(ptemp1,pp3,pp4,1,block_size, 1, block_size, 1, block_size);
    //     ci = C21;
    //     pi = ptemp1;
    //     for( size_t i = 0; i < block_size; ++i, ci += wr, pi += 1 ){
    //         auto cj = ci;
    //         auto pj = pi;
    //         for(size_t j = 0; j < block_size; ++j, cj += wc, pj += wr){
    //             *cj = * pj;
    //         }
    //     }

    //     add_8x8(ptemp1,pp5,pp1,1,block_size, 1, block_size, 1, block_size);
    //     add_8x8(pp6,pp3,pp7,1,block_size, 1, block_size, 1, block_size);
    //     sub_8x8(ptemp2,ptemp1,pp6,1,block_size, 1, block_size, 1, block_size);
    //     ci = C22;
    //     pi = ptemp2;
    //     for( size_t i = 0; i < block_size; ++i, ci += wr, pi += 1 ){
    //         auto cj = ci;
    //         auto pj = pi;
    //         for(size_t j = 0; j < block_size; ++j, cj += wc, pj += wr){
    //             *cj = * pj;
    //         }
    //     }

    // }

    template< typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    void mult_16x16_2(float* c, SizeType const* nc, SizeType const* wc, 
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb) noexcept
    {
        assert( c != nullptr && a != nullptr && b!= nullptr );
        __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12,
            ymm13, ymm14, ymm15;

        float temp_res1[8] = {0};
        float temp_res2[8] = {0};
        float temp_res3[8] = {0};
        float temp_res4[8] = {0};
        
        ymm0 = _mm256_load_ps(a + wa[1] * 0);
        ymm1 = _mm256_load_ps(a + wa[1] * 0 + wa[0] * 8);

        ymm2 = _mm256_load_ps(a + wa[1] * 1);
        ymm3 = _mm256_load_ps(a + wa[1] * 1 + wa[0] * 8);
        
        ymm4 = _mm256_load_ps(a + wa[1] * 2);
        ymm5 = _mm256_load_ps(a + wa[1] * 2 + wa[0] * 8);
        
        ymm6 = _mm256_load_ps(a + wa[1] * 3);
        ymm7 = _mm256_load_ps(a + wa[1] * 3 + wa[0] * 8);
        

        ymm8 = _mm256_load_ps(b);
        ymm9 = _mm256_load_ps(b + wb[1] * 0 + wb[0] * 8);
        
        ymm10 = _mm256_load_ps(b + wb[1]);
        ymm11 = _mm256_load_ps(b + wb[1] * 1 + wb[0] * 8);

        // ----------------------- row 0 col [0,4) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 0 + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 0 + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 0 + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 0 + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row 1 col [0,4) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 1 + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 1 + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 1 + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 1 + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];
        
        //----------------- Store col [0,3] ----------------------- //
        
        float col0_f[8];
        float col0_s[8];
        
        _mm256_store_ps(col0_f,ymm0);
        _mm256_store_ps(col0_s,ymm1);

        float col1_f[8];
        float col1_s[8];
        
        _mm256_store_ps(col1_f,ymm2);
        _mm256_store_ps(col1_s,ymm3);
        
        float col2_f[8];
        float col2_s[8];
        
        _mm256_store_ps(col2_f,ymm4);
        _mm256_store_ps(col2_s,ymm5);
        
        float col3_f[8];
        float col3_s[8];
        
        _mm256_store_ps(col3_f,ymm6);
        _mm256_store_ps(col3_s,ymm7);

        //----------------- Load col [4,7] ----------------------- //
        ymm0 = _mm256_load_ps(a + wa[1] * 4);
        ymm1 = _mm256_load_ps(a + wa[1] * 4 + wa[0] * 8);

        ymm2 = _mm256_load_ps(a + wa[1] * 5);
        ymm3 = _mm256_load_ps(a + wa[1] * 5 + wa[0] * 8);
        
        ymm4 = _mm256_load_ps(a + wa[1] * 6);
        ymm5 = _mm256_load_ps(a + wa[1] * 6 + wa[0] * 8);
        
        ymm6 = _mm256_load_ps(a + wa[1] * 7);
        ymm7 = _mm256_load_ps(a + wa[1] * 7 + wa[0] * 8);

        // ----------------------- row 0 col [5,8) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 0 + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 0 + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 0 + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 0 + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row 1 col [5,8) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 1 + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 1 + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 1 + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 1 + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];
   
        //----------------- Store col [4,7] ----------------------- //
        
        float col4_f[8];
        float col4_s[8];
        
        _mm256_store_ps(col4_f,ymm0);
        _mm256_store_ps(col4_s,ymm1);

        float col5_f[8];
        float col5_s[8];
        
        _mm256_store_ps(col5_f,ymm2);
        _mm256_store_ps(col5_s,ymm3);
        
        float col6_f[8];
        float col6_s[8];
        
        _mm256_store_ps(col6_f,ymm4);
        _mm256_store_ps(col6_s,ymm5);
        
        float col7_f[8];
        float col7_s[8];
        
        _mm256_store_ps(col7_f,ymm6);
        _mm256_store_ps(col7_s,ymm7);


        //----------------- Load col [4,7] ----------------------- //
        ymm0 = _mm256_load_ps(a + wa[1] * 8);
        ymm1 = _mm256_load_ps(a + wa[1] * 8 + wa[0] * 8);

        ymm2 = _mm256_load_ps(a + wa[1] * 9);
        ymm3 = _mm256_load_ps(a + wa[1] * 9 + wa[0] * 8);
        
        ymm4 = _mm256_load_ps(a + wa[1] * 10);
        ymm5 = _mm256_load_ps(a + wa[1] * 10 + wa[0] * 8);
        
        ymm6 = _mm256_load_ps(a + wa[1] * 11);
        ymm7 = _mm256_load_ps(a + wa[1] * 11 + wa[0] * 8);

        // ----------------------- row 0 col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 0 + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 0 + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 0 + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 0 + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 1 + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 1 + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 1 + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 1 + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];
   
        //----------------- Store col [8,11] ----------------------- //
        
        float col8_f[8];
        float col8_s[8];
        
        _mm256_store_ps(col8_f,ymm0);
        _mm256_store_ps(col8_s,ymm1);

        float col9_f[8];
        float col9_s[8];
        
        _mm256_store_ps(col9_f,ymm2);
        _mm256_store_ps(col9_s,ymm3);
        
        float col10_f[8];
        float col10_s[8];
        
        _mm256_store_ps(col10_f,ymm4);
        _mm256_store_ps(col10_s,ymm5);
        
        float col11_f[8];
        float col11_s[8];
        
        _mm256_store_ps(col11_f,ymm6);
        _mm256_store_ps(col11_s,ymm7);


        //----------------- Load col [4,7] ----------------------- //
        ymm0 = _mm256_load_ps(a + wa[1] * 12);
        ymm1 = _mm256_load_ps(a + wa[1] * 12 + wa[0] * 8);

        ymm2 = _mm256_load_ps(a + wa[1] * 13);
        ymm3 = _mm256_load_ps(a + wa[1] * 13 + wa[0] * 8);
        
        ymm4 = _mm256_load_ps(a + wa[1] * 14);
        ymm5 = _mm256_load_ps(a + wa[1] * 14 + wa[0] * 8);
        
        ymm6 = _mm256_load_ps(a + wa[1] * 15);
        ymm7 = _mm256_load_ps(a + wa[1] * 15 + wa[0] * 8);

        // ----------------------- row 0 col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 0 + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 0 + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 0 + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 0 + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 1 + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 1 + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 1 + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 1 + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];
   
        //----------------- Load row [2,3] ----------------------- //

        ymm8 = _mm256_load_ps(b + wb[1] * 2);
        ymm9 = _mm256_load_ps(b + wb[1] * 2 + wb[0] * 8);
        
        ymm10 = _mm256_load_ps(b + wb[1] * 3);
        ymm11 = _mm256_load_ps(b + wb[1] * 3 + wb[0] * 8);

        // ----------------------- row 0 col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 2 + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 2 + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 2 + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 2 + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 3 + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 3 + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 3 + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 3 + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        //----------------- Store col [12,15] ----------------------- //
        
        float col12_f[8];
        float col12_s[8];
        
        _mm256_store_ps(col12_f,ymm0);
        _mm256_store_ps(col12_s,ymm1);

        float col13_f[8];
        float col13_s[8];
        
        _mm256_store_ps(col13_f,ymm2);
        _mm256_store_ps(col13_s,ymm3);
        
        float col14_f[8];
        float col14_s[8];
        
        _mm256_store_ps(col14_f,ymm4);
        _mm256_store_ps(col14_s,ymm5);
        
        float col15_f[8];
        float col15_s[8];
        
        _mm256_store_ps(col15_f,ymm6);
        _mm256_store_ps(col15_s,ymm7);

        //----------------- Load col [8,11] ----------------------- //

        ymm0 = _mm256_load_ps(col8_f); 
        ymm1 = _mm256_load_ps(col8_s);

        ymm2 = _mm256_load_ps(col9_f); 
        ymm3 = _mm256_load_ps(col9_s);
        
        ymm4 = _mm256_load_ps(col10_f); 
        ymm5 = _mm256_load_ps(col10_s);
        
        ymm6 = _mm256_load_ps(col11_f); 
        ymm7 = _mm256_load_ps(col11_s);

        // ----------------------- row 2 col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 2 + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 2 + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 2 + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 2 + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row 3 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 3 + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 3 + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 3 + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 3 + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [4,7] ----------------------- //

        ymm0 = _mm256_load_ps(col4_f); 
        ymm1 = _mm256_load_ps(col4_s);

        ymm2 = _mm256_load_ps(col5_f); 
        ymm3 = _mm256_load_ps(col5_s);
        
        ymm4 = _mm256_load_ps(col6_f); 
        ymm5 = _mm256_load_ps(col6_s);
        
        ymm6 = _mm256_load_ps(col7_f); 
        ymm7 = _mm256_load_ps(col7_s);

        // ----------------------- row 2 col [4,8) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 2 + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 2 + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 2 + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 2 + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row 3 col [4,8) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 3 + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 3 + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 3 + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 3 + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [0,3] ----------------------- //

        ymm0 = _mm256_load_ps(col0_f); 
        ymm1 = _mm256_load_ps(col0_s);

        ymm2 = _mm256_load_ps(col1_f); 
        ymm3 = _mm256_load_ps(col1_s);
        
        ymm4 = _mm256_load_ps(col2_f); 
        ymm5 = _mm256_load_ps(col2_s);
        
        ymm6 = _mm256_load_ps(col3_f); 
        ymm7 = _mm256_load_ps(col3_s);

        // ----------------------- row 2 col [0,4) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 2 + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 2 + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 2 + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 2 + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row 3 col [0,4) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * 3 + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * 3 + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * 3 + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * 3 + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];
        
        size_t j = 4;
        size_t i = j;

        //----------------- Load row [j,1 + j] ----------------------- //

        ymm8 = _mm256_load_ps(b + wb[1] * j);
        ymm9 = _mm256_load_ps(b + wb[1] * j + wb[0] * 8);
        
        ymm10 = _mm256_load_ps(b + wb[1] * (j + 1));
        ymm11 = _mm256_load_ps(b + wb[1] * (j + 1) + wb[0] * 8);

        //----------------- Load row [12,15] ----------------------- //

        ymm0 = _mm256_load_ps(col12_f); 
        ymm1 = _mm256_load_ps(col12_s);

        ymm2 = _mm256_load_ps(col13_f); 
        ymm3 = _mm256_load_ps(col13_s);
        
        ymm4 = _mm256_load_ps(col14_f); 
        ymm5 = _mm256_load_ps(col14_s);
        
        ymm6 = _mm256_load_ps(col15_f); 
        ymm7 = _mm256_load_ps(col15_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [8,11] ----------------------- //

        ymm0 = _mm256_load_ps(col8_f); 
        ymm1 = _mm256_load_ps(col8_s);

        ymm2 = _mm256_load_ps(col9_f); 
        ymm3 = _mm256_load_ps(col9_s);
        
        ymm4 = _mm256_load_ps(col10_f); 
        ymm5 = _mm256_load_ps(col10_s);
        
        ymm6 = _mm256_load_ps(col11_f); 
        ymm7 = _mm256_load_ps(col11_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [4,7] ----------------------- //

        ymm0 = _mm256_load_ps(col4_f); 
        ymm1 = _mm256_load_ps(col4_s);

        ymm2 = _mm256_load_ps(col5_f); 
        ymm3 = _mm256_load_ps(col5_s);
        
        ymm4 = _mm256_load_ps(col6_f); 
        ymm5 = _mm256_load_ps(col6_s);
        
        ymm6 = _mm256_load_ps(col7_f); 
        ymm7 = _mm256_load_ps(col7_s);

        // ----------------------- row i col [4,8) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [4,8) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [0,3] ----------------------- //

        ymm0 = _mm256_load_ps(col0_f); 
        ymm1 = _mm256_load_ps(col0_s);

        ymm2 = _mm256_load_ps(col1_f); 
        ymm3 = _mm256_load_ps(col1_s);
        
        ymm4 = _mm256_load_ps(col2_f); 
        ymm5 = _mm256_load_ps(col2_s);
        
        ymm6 = _mm256_load_ps(col3_f); 
        ymm7 = _mm256_load_ps(col3_s);

        // ----------------------- row i col [0,4) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [0,4) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];

        // ------------------------------------------------------------//
        // XXX: i = 6
        j = 6;
        i = j;

        //----------------- Load row [j,1 + j] ----------------------- //

        ymm8 = _mm256_load_ps(b + wb[1] * j);
        ymm9 = _mm256_load_ps(b + wb[1] * j + wb[0] * 8);
        
        ymm10 = _mm256_load_ps(b + wb[1] * (j + 1));
        ymm11 = _mm256_load_ps(b + wb[1] * (j + 1) + wb[0] * 8);

        //----------------- Load row [12,15] ----------------------- //

        ymm0 = _mm256_load_ps(col12_f); 
        ymm1 = _mm256_load_ps(col12_s);

        ymm2 = _mm256_load_ps(col13_f); 
        ymm3 = _mm256_load_ps(col13_s);
        
        ymm4 = _mm256_load_ps(col14_f); 
        ymm5 = _mm256_load_ps(col14_s);
        
        ymm6 = _mm256_load_ps(col15_f); 
        ymm7 = _mm256_load_ps(col15_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [8,11] ----------------------- //

        ymm0 = _mm256_load_ps(col8_f); 
        ymm1 = _mm256_load_ps(col8_s);

        ymm2 = _mm256_load_ps(col9_f); 
        ymm3 = _mm256_load_ps(col9_s);
        
        ymm4 = _mm256_load_ps(col10_f); 
        ymm5 = _mm256_load_ps(col10_s);
        
        ymm6 = _mm256_load_ps(col11_f); 
        ymm7 = _mm256_load_ps(col11_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [4,7] ----------------------- //

        ymm0 = _mm256_load_ps(col4_f); 
        ymm1 = _mm256_load_ps(col4_s);

        ymm2 = _mm256_load_ps(col5_f); 
        ymm3 = _mm256_load_ps(col5_s);
        
        ymm4 = _mm256_load_ps(col6_f); 
        ymm5 = _mm256_load_ps(col6_s);
        
        ymm6 = _mm256_load_ps(col7_f); 
        ymm7 = _mm256_load_ps(col7_s);

        // ----------------------- row i col [4,8) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [4,8) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [0,3] ----------------------- //

        ymm0 = _mm256_load_ps(col0_f); 
        ymm1 = _mm256_load_ps(col0_s);

        ymm2 = _mm256_load_ps(col1_f); 
        ymm3 = _mm256_load_ps(col1_s);
        
        ymm4 = _mm256_load_ps(col2_f); 
        ymm5 = _mm256_load_ps(col2_s);
        
        ymm6 = _mm256_load_ps(col3_f); 
        ymm7 = _mm256_load_ps(col3_s);

        // ----------------------- row i col [0,4) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [0,4) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];



        // ------------------------------------------------------------//
        // XXX: i = 8
        j = 8;
        i = j;

        //----------------- Load row [j,1 + j] ----------------------- //

        ymm8 = _mm256_load_ps(b + wb[1] * j);
        ymm9 = _mm256_load_ps(b + wb[1] * j + wb[0] * 8);
        
        ymm10 = _mm256_load_ps(b + wb[1] * (j + 1));
        ymm11 = _mm256_load_ps(b + wb[1] * (j + 1) + wb[0] * 8);

        //----------------- Load row [12,15] ----------------------- //

        ymm0 = _mm256_load_ps(col12_f); 
        ymm1 = _mm256_load_ps(col12_s);

        ymm2 = _mm256_load_ps(col13_f); 
        ymm3 = _mm256_load_ps(col13_s);
        
        ymm4 = _mm256_load_ps(col14_f); 
        ymm5 = _mm256_load_ps(col14_s);
        
        ymm6 = _mm256_load_ps(col15_f); 
        ymm7 = _mm256_load_ps(col15_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [8,11] ----------------------- //

        ymm0 = _mm256_load_ps(col8_f); 
        ymm1 = _mm256_load_ps(col8_s);

        ymm2 = _mm256_load_ps(col9_f); 
        ymm3 = _mm256_load_ps(col9_s);
        
        ymm4 = _mm256_load_ps(col10_f); 
        ymm5 = _mm256_load_ps(col10_s);
        
        ymm6 = _mm256_load_ps(col11_f); 
        ymm7 = _mm256_load_ps(col11_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [4,7] ----------------------- //

        ymm0 = _mm256_load_ps(col4_f); 
        ymm1 = _mm256_load_ps(col4_s);

        ymm2 = _mm256_load_ps(col5_f); 
        ymm3 = _mm256_load_ps(col5_s);
        
        ymm4 = _mm256_load_ps(col6_f); 
        ymm5 = _mm256_load_ps(col6_s);
        
        ymm6 = _mm256_load_ps(col7_f); 
        ymm7 = _mm256_load_ps(col7_s);

        // ----------------------- row i col [4,8) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [4,8) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [0,3] ----------------------- //

        ymm0 = _mm256_load_ps(col0_f); 
        ymm1 = _mm256_load_ps(col0_s);

        ymm2 = _mm256_load_ps(col1_f); 
        ymm3 = _mm256_load_ps(col1_s);
        
        ymm4 = _mm256_load_ps(col2_f); 
        ymm5 = _mm256_load_ps(col2_s);
        
        ymm6 = _mm256_load_ps(col3_f); 
        ymm7 = _mm256_load_ps(col3_s);

        // ----------------------- row i col [0,4) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [0,4) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];



        // ------------------------------------------------------------//
        // XXX: i = 10
        j = 10;
        i = j;

        //----------------- Load row [j,1 + j] ----------------------- //

        ymm8 = _mm256_load_ps(b + wb[1] * j);
        ymm9 = _mm256_load_ps(b + wb[1] * j + wb[0] * 8);
        
        ymm10 = _mm256_load_ps(b + wb[1] * (j + 1));
        ymm11 = _mm256_load_ps(b + wb[1] * (j + 1) + wb[0] * 8);

        //----------------- Load row [12,15] ----------------------- //

        ymm0 = _mm256_load_ps(col12_f); 
        ymm1 = _mm256_load_ps(col12_s);

        ymm2 = _mm256_load_ps(col13_f); 
        ymm3 = _mm256_load_ps(col13_s);
        
        ymm4 = _mm256_load_ps(col14_f); 
        ymm5 = _mm256_load_ps(col14_s);
        
        ymm6 = _mm256_load_ps(col15_f); 
        ymm7 = _mm256_load_ps(col15_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [8,11] ----------------------- //

        ymm0 = _mm256_load_ps(col8_f); 
        ymm1 = _mm256_load_ps(col8_s);

        ymm2 = _mm256_load_ps(col9_f); 
        ymm3 = _mm256_load_ps(col9_s);
        
        ymm4 = _mm256_load_ps(col10_f); 
        ymm5 = _mm256_load_ps(col10_s);
        
        ymm6 = _mm256_load_ps(col11_f); 
        ymm7 = _mm256_load_ps(col11_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [4,7] ----------------------- //

        ymm0 = _mm256_load_ps(col4_f); 
        ymm1 = _mm256_load_ps(col4_s);

        ymm2 = _mm256_load_ps(col5_f); 
        ymm3 = _mm256_load_ps(col5_s);
        
        ymm4 = _mm256_load_ps(col6_f); 
        ymm5 = _mm256_load_ps(col6_s);
        
        ymm6 = _mm256_load_ps(col7_f); 
        ymm7 = _mm256_load_ps(col7_s);

        // ----------------------- row i col [4,8) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [4,8) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [0,3] ----------------------- //

        ymm0 = _mm256_load_ps(col0_f); 
        ymm1 = _mm256_load_ps(col0_s);

        ymm2 = _mm256_load_ps(col1_f); 
        ymm3 = _mm256_load_ps(col1_s);
        
        ymm4 = _mm256_load_ps(col2_f); 
        ymm5 = _mm256_load_ps(col2_s);
        
        ymm6 = _mm256_load_ps(col3_f); 
        ymm7 = _mm256_load_ps(col3_s);

        // ----------------------- row i col [0,4) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [0,4) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];



        // ------------------------------------------------------------//
        // XXX: i = 12
        j = 12;
        i = j;

        //----------------- Load row [j,1 + j] ----------------------- //

        ymm8 = _mm256_load_ps(b + wb[1] * j);
        ymm9 = _mm256_load_ps(b + wb[1] * j + wb[0] * 8);
        
        ymm10 = _mm256_load_ps(b + wb[1] * (j + 1));
        ymm11 = _mm256_load_ps(b + wb[1] * (j + 1) + wb[0] * 8);

        //----------------- Load row [12,15] ----------------------- //

        ymm0 = _mm256_load_ps(col12_f); 
        ymm1 = _mm256_load_ps(col12_s);

        ymm2 = _mm256_load_ps(col13_f); 
        ymm3 = _mm256_load_ps(col13_s);
        
        ymm4 = _mm256_load_ps(col14_f); 
        ymm5 = _mm256_load_ps(col14_s);
        
        ymm6 = _mm256_load_ps(col15_f); 
        ymm7 = _mm256_load_ps(col15_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [8,11] ----------------------- //

        ymm0 = _mm256_load_ps(col8_f); 
        ymm1 = _mm256_load_ps(col8_s);

        ymm2 = _mm256_load_ps(col9_f); 
        ymm3 = _mm256_load_ps(col9_s);
        
        ymm4 = _mm256_load_ps(col10_f); 
        ymm5 = _mm256_load_ps(col10_s);
        
        ymm6 = _mm256_load_ps(col11_f); 
        ymm7 = _mm256_load_ps(col11_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [4,7] ----------------------- //

        ymm0 = _mm256_load_ps(col4_f); 
        ymm1 = _mm256_load_ps(col4_s);

        ymm2 = _mm256_load_ps(col5_f); 
        ymm3 = _mm256_load_ps(col5_s);
        
        ymm4 = _mm256_load_ps(col6_f); 
        ymm5 = _mm256_load_ps(col6_s);
        
        ymm6 = _mm256_load_ps(col7_f); 
        ymm7 = _mm256_load_ps(col7_s);

        // ----------------------- row i col [4,8) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [4,8) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [0,3] ----------------------- //

        ymm0 = _mm256_load_ps(col0_f); 
        ymm1 = _mm256_load_ps(col0_s);

        ymm2 = _mm256_load_ps(col1_f); 
        ymm3 = _mm256_load_ps(col1_s);
        
        ymm4 = _mm256_load_ps(col2_f); 
        ymm5 = _mm256_load_ps(col2_s);
        
        ymm6 = _mm256_load_ps(col3_f); 
        ymm7 = _mm256_load_ps(col3_s);

        // ----------------------- row i col [0,4) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [0,4) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];


        // ------------------------------------------------------------//
        // XXX: i = 14
        j = 14;
        i = j;

        //----------------- Load row [j,1 + j] ----------------------- //

        ymm8 = _mm256_load_ps(b + wb[1] * j);
        ymm9 = _mm256_load_ps(b + wb[1] * j + wb[0] * 8);
        
        ymm10 = _mm256_load_ps(b + wb[1] * (j + 1));
        ymm11 = _mm256_load_ps(b + wb[1] * (j + 1) + wb[0] * 8);

        //----------------- Load row [12,15] ----------------------- //

        ymm0 = _mm256_load_ps(col12_f); 
        ymm1 = _mm256_load_ps(col12_s);

        ymm2 = _mm256_load_ps(col13_f); 
        ymm3 = _mm256_load_ps(col13_s);
        
        ymm4 = _mm256_load_ps(col14_f); 
        ymm5 = _mm256_load_ps(col14_s);
        
        ymm6 = _mm256_load_ps(col15_f); 
        ymm7 = _mm256_load_ps(col15_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 12 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 13 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 14 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 15 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [8,11] ----------------------- //

        ymm0 = _mm256_load_ps(col8_f); 
        ymm1 = _mm256_load_ps(col8_s);

        ymm2 = _mm256_load_ps(col9_f); 
        ymm3 = _mm256_load_ps(col9_s);
        
        ymm4 = _mm256_load_ps(col10_f); 
        ymm5 = _mm256_load_ps(col10_s);
        
        ymm6 = _mm256_load_ps(col11_f); 
        ymm7 = _mm256_load_ps(col11_s);

        // ----------------------- row i col [9,13) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [9,13) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 8 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 9 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 10 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 11 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [4,7] ----------------------- //

        ymm0 = _mm256_load_ps(col4_f); 
        ymm1 = _mm256_load_ps(col4_s);

        ymm2 = _mm256_load_ps(col5_f); 
        ymm3 = _mm256_load_ps(col5_s);
        
        ymm4 = _mm256_load_ps(col6_f); 
        ymm5 = _mm256_load_ps(col6_s);
        
        ymm6 = _mm256_load_ps(col7_f); 
        ymm7 = _mm256_load_ps(col7_s);

        // ----------------------- row i col [4,8) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [4,8) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 4 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 5 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 6 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 7 ) = temp_res4[0] + temp_res4[4];

        //----------------- Load col [0,3] ----------------------- //

        ymm0 = _mm256_load_ps(col0_f); 
        ymm1 = _mm256_load_ps(col0_s);

        ymm2 = _mm256_load_ps(col1_f); 
        ymm3 = _mm256_load_ps(col1_s);
        
        ymm4 = _mm256_load_ps(col2_f); 
        ymm5 = _mm256_load_ps(col2_s);
        
        ymm6 = _mm256_load_ps(col3_f); 
        ymm7 = _mm256_load_ps(col3_s);

        // ----------------------- row i col [0,4) ---------------------------- //
        
        ymm12 = _mm256_dp_ps(ymm0,ymm8,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm8,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm9,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm8,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm9,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * i + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * i + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * i + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * i + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];

        // ----------------------- row i + 1 col [0,4) ---------------------------- //

        ymm12 = _mm256_dp_ps(ymm0,ymm10,0xff);
        ymm13 = _mm256_dp_ps(ymm1,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm12,ymm13);
        _mm256_store_ps(temp_res1,ymm13);
        
        ymm13 = _mm256_dp_ps(ymm2,ymm10,0xff);
        ymm14 = _mm256_dp_ps(ymm3,ymm11,0xff);
        ymm15 = _mm256_add_ps(ymm13,ymm14);
        _mm256_store_ps(temp_res2,ymm15);
        
        
        ymm14 = _mm256_dp_ps(ymm4,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm5,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res3,ymm13);

        
        ymm14 = _mm256_dp_ps(ymm6,ymm10,0xff);
        ymm15 = _mm256_dp_ps(ymm7,ymm11,0xff);
        ymm13 = _mm256_add_ps(ymm14,ymm15);
        _mm256_store_ps(temp_res4,ymm13);

        *( c + wc[0] * (i + 1) + wc[1] * 0 ) = temp_res1[0] + temp_res1[4];
        *( c + wc[0] * (i + 1) + wc[1] * 1 ) = temp_res2[0] + temp_res2[4];
        *( c + wc[0] * (i + 1) + wc[1] * 2 ) = temp_res3[0] + temp_res3[4];
        *( c + wc[0] * (i + 1) + wc[1] * 3 ) = temp_res4[0] + temp_res4[4];
    }

} // namespace boost::numeric::ublas::simd


#endif // BOOST_NUMERIC_UBLAS_SIMD_MULT_16X16_HPP
