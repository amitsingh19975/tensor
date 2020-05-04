#if !defined(BOOST_NUMERIC_UBLAS_SIMD_MULT_8nX8n_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_MULT_8nX8n_HPP

#include <boost/numeric/ublas/tensor/simd/matrix_mult/mult_8x8.hpp>
#include <boost/numeric/ublas/tensor/simd/matrix_mult/mult_16x16.hpp>
#include <boost/numeric/ublas/tensor/simd/matrix_add/add.hpp>
#include <boost/numeric/ublas/tensor/simd/matrix_sub/sub.hpp>
#include <array>
#include <vector>

namespace boost::numeric::ublas::simd{

       
//    BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
//    void prefetch(float const* p, size_t offset) noexcept{
//        _mm_prefetch(p, _MM_HINT_T0);
//        p += offset;
//        _mm_prefetch(p, _MM_HINT_T0);
//        p += offset;
//        _mm_prefetch(p, _MM_HINT_T0);
//        p += offset;
//        _mm_prefetch(p, _MM_HINT_T0);
//        p += offset;
//        _mm_prefetch(p, _MM_HINT_T0);
//        p += offset;
//        _mm_prefetch(p, _MM_HINT_T0);
//        p += offset;
//        _mm_prefetch(p, _MM_HINT_T0);
//        p += offset;
//        _mm_prefetch(p, _MM_HINT_T0);
//    }

    template <class SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    void mult_8nx8n(float* c, SizeType const* nc, SizeType const* wc,
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb
        ) noexcept
    {

        constexpr auto block_size = 8;

        if( na[1] == block_size && nb[0] == block_size ){
            mult_8x8(c, nc, wc, a, na, wa, b, nb, wb);
            return;
        }

        constexpr std::array<SizeType, 2> const nt = {8, 8};
        constexpr std::array<SizeType, 2> const wt = {1,8};

        auto pnt = nt.data();
        auto pwt = wt.data();
        
        auto ai = a;
        auto bi = b;
        auto ci = c;
        
        #pragma omp parallel for nowait
        for( auto i = 0ul; i < na[1]; i += block_size, ai += wa[1] * block_size, ci += wc[1] * block_size ){
            auto bj = bi;
            auto cj = ci;
            for( auto j = 0ul; j < nb[0]; j += block_size, bj += wb[0] * block_size, cj += wc[0] * block_size ){
                auto ak = ai;
                auto bk = bj;
                for( auto k = 0; k < nb[0]; k += block_size, ak += wa[0] * block_size, bk += wb[1] * block_size ){
                    auto pa = ak;
                    auto pb = bk;
                    mult_add_8x8(cj, pnt, wc, ak, pnt, wa, bk, pnt, wb);
                }
            }
        }

    }

} // namespace boost::numeric::ublas::simd


namespace boost::numeric::ublas::simd{

    template<typename SizeType>
    void print(float const* p, SizeType const* np, SizeType const* wp){
        auto pi = p;
        std::cout<<std::endl;
        for(auto i = 0; i < np[0]; ++i, pi += wp[0]){
            auto pj = pi;
            for(auto j = 0; j < np[1]; ++j, pj += wp[1]){
                std::cout<<*pj<<' ';
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }

    template <class SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    void copy_mat(float* c, SizeType const* nc, SizeType const* wc,
        float const* a, SizeType const* na, SizeType const* wa
        ) noexcept
    {
        auto ci = c;
        auto ai = a;
        for( SizeType i = 0; i < nc[0]; ++i, ci += wc[1], ai += wc[1] ){
            auto cj = ci;
            auto aj = ai;
            for( SizeType j = 0; j < nc[1]; ++j, cj += wc[0], aj += wc[0] ){
                *cj = *aj;
            }
        }
    }

    // template <class SizeType>
    // BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    // void mult_8nx8n_2(float* c, SizeType const* nc, SizeType const* wc,
    //     float const* a, SizeType const* na, SizeType const* wa,
    //     float const* b, SizeType const* nb, SizeType const* wb
    //     ) noexcept
    // {

    //     if( na[1] == 8 && nb[0] == 8 ){
    //         mult_8x8(c, nc, wc, a, na, wa, b, nb, wb);
    //         return;
    //     }

    //     //------------- Mat A ------------------ //
    //     // +----------------+
    //     // |        |       |
    //     // |   A    |   B   |
    //     // |--------|-------|
    //     // |   C    |   D   |
    //     // |        |       |
    //     // +----------------+

    //     float const* A = a;
    //     float const* B = a + wa[1] * nt[0];
    //     float const* C = a + wa[0] * nt[0];
    //     float const* D = a + wa[0] * nt[1] + wa[1] * nt[0];
        
    //     //------------- Mat B ------------------ //
    //     // +----------------+
    //     // |        |       |
    //     // |   E    |   F   |
    //     // |--------|-------|
    //     // |   G    |   H   |
    //     // |        |       |
    //     // +----------------+


    //     float const* E = b;
    //     float const* F = b + wb[1] * nt[0];
    //     float const* G = b + wb[0] * nt[0];
    //     float const* H = b + wb[0] * nt[1] + wb[1] * nt[0];
        
    //     //------------- Mat C ------------------ //
    //     // +----------------+
    //     // |        |       |
    //     // |   C11  |   C12 |
    //     // |--------|-------|
    //     // |   C21  |   C22 |
    //     // |        |       |
    //     // +----------------+


    //     float* C11 = c;
    //     float* C12 = c + wc[1] * nt[0];
    //     float* C21 = c + wc[0] * nt[0];
    //     float* C22 = c + wc[0] * nt[1] + wc[1] * nt[0];



    // }

} // namespace boost::numeric::ublas::simd


#endif // BOOST_NUMERIC_UBLAS_SIMD_MULT_8nX8n_HPP
