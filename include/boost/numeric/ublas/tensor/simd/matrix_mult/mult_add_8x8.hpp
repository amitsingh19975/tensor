#if !defined(BOOST_NUMERIC_UBLAS_SIMD_MULT_ADD_8X8_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_MULT_ADD_8X8_HPP

#include <boost/numeric/ublas/tensor/simd/detail/simd_helper.hpp>


void print(__m256 d ){
    std::cout<<d[0]<<' '<<d[1]<<' '<<d[2]<<' '<<d[3]<<' ';
    std::cout<<d[4]<<' '<<d[5]<<' '<<d[6]<<' '<<d[7]<<'\n';
}

#define PRINT(N) print(ymm##N);


namespace boost::numeric::ublas::simd{

    template<int N, typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    auto copy_mat( float const*  a,  SizeType const*  na, SizeType const*  wa) 
         noexcept
    {
        alignas(32) std::array<float, N * N> res;
        auto r = res.data();
        auto ri = res.data();
        auto ai = a;
        #pragma omp simd
        for(auto i = 0; i < N; ++i){
            ai = a + ( i ) * wa[1];
            ri = r + ( i ) * N;
            std::copy(ai, ai + N, ri);
        }
        return res;
    }

    template<typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void mult_add_8x8(float * c, SizeType const * nc, SizeType const * wc, 
        float const * a, SizeType const * na, SizeType const * wa,
        float const * b, SizeType const * nb, SizeType const * wb) noexcept
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
        size_t wb01 = wb_0;
        size_t wb02 = wb_0 + wb01;
        size_t wb03 = wb_0 + wb02;
        size_t wb04 = wb_0 + wb03;
        size_t wb05 = wb_0 + wb04;
        size_t wb06 = wb_0 + wb05;
        size_t wb07 = wb_0 + wb06;
        auto bi = b;
        auto ci = c;

        for( int i = 7; i >= 0; --i ){

            ymm0 = _mm256_load_ps(a);
            ymm10 = _mm256_broadcast_ss(bi + wb00);
            ymm8 = _mm256_fmadd_ps(ymm10,ymm0,_mm256_load_ps(ci));

            ymm0 = _mm256_load_ps(a + wa_1);
            ymm9 = _mm256_broadcast_ss(bi + wb01);
            ymm8 = _mm256_fmadd_ps(ymm9,ymm0,ymm8);
            
            ymm0 = _mm256_load_ps(a + wa_1 * 2);
            ymm10 = _mm256_broadcast_ss(bi + wb02);
            ymm8 = _mm256_fmadd_ps(ymm10,ymm0,ymm8);
            
            ymm0 = _mm256_load_ps(a + wa_1 * 3);
            ymm11 = _mm256_broadcast_ss(bi + wb03);
            ymm8 = _mm256_fmadd_ps(ymm11,ymm0,ymm8);
            
            ymm0 = _mm256_load_ps(a + wa_1 * 4);
            ymm12 = _mm256_broadcast_ss(bi + wb04);
            ymm8 = _mm256_fmadd_ps(ymm12,ymm0,ymm8);

            ymm0 = _mm256_load_ps(a + wa_1 * 5);
            ymm13 = _mm256_broadcast_ss(bi + wb05);
            ymm8 = _mm256_fmadd_ps(ymm13,ymm0,ymm8);
            
            ymm0 = _mm256_load_ps(a + wa_1 * 6);
            ymm14 = _mm256_broadcast_ss(bi + wb06);
            ymm8 = _mm256_fmadd_ps(ymm14,ymm0,ymm8);

            ymm0 = _mm256_load_ps(a + wa_1 * 7);
            ymm15 = _mm256_broadcast_ss(bi + wb07);
            ymm8 = _mm256_fmadd_ps(ymm15,ymm0,ymm8);
            
            _mm256_store_ps(ci,ymm8);
            // _mm_prefetch(ci,_MM_HINT_T0);
            bi += wb_1, ci += wc_1;
        }
    }


    template<typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void kernel_8x8(float* c, SizeType const* nc, SizeType const* wc, 
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb) noexcept
    {
        constexpr SizeType const block_size = 8;
        constexpr SizeType const kernel_size = 8;
        constexpr SizeType const nt[2] = {kernel_size, kernel_size};
        constexpr SizeType const wt[2] = {1,kernel_size};
        auto ppa = a;
        auto ppb = b;
        mult_add_8x8(c, nt, wc, ppa, nt, wa, ppb, nt, wb);

    }

    
} // namespace boost::numeric::ublas::simd


#endif // BOOST_NUMERIC_UBLAS_SIMD_MULT_ADD_8X8_HPP
