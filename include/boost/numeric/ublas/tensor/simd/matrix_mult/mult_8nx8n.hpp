#if !defined(BOOST_NUMERIC_UBLAS_SIMD_MULT_8nX8n_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_MULT_8nX8n_HPP

#include <boost/numeric/ublas/tensor/simd/matrix_mult/mult_8x8.hpp>
#include <boost/numeric/ublas/tensor/simd/matrix_mult/mult_add_8x8.hpp>
#include <boost/numeric/ublas/tensor/simd/matrix_mult/mult_16x16.hpp>
#include <boost/numeric/ublas/tensor/simd/matrix_add/add.hpp>
#include <boost/numeric/ublas/tensor/simd/matrix_sub/sub.hpp>
#include <array>
#include <vector>

namespace boost::numeric::ublas::simd{

       
    template <class SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    void mult_8nx8n_helper_2(float* c, SizeType const* nc, SizeType const* wc,
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb
        ) noexcept
    {

        constexpr SizeType const kernel_size = 8;

        auto aii = a;
        auto bii = b;
        auto cii = c;

        SizeType const wa_0 = wa[0] * kernel_size;
        SizeType const wa_1 = wa[1] * kernel_size;
        SizeType const wb_0 = wb[0] * kernel_size;
        SizeType const wb_1 = wb[1] * kernel_size;
        SizeType const wc_0 = wc[0] * kernel_size;
        SizeType const wc_1 = wc[1] * kernel_size;

        for( auto ii = 0ul; ii < na[1]; ii += kernel_size ){
            auto ak = aii;
            auto bk = bii;
            auto ck = cii;
            for(auto k = 0ul; k < nc[0]; k += kernel_size){
                auto ajj = ak;
                auto bjj = bk;
                auto cjj = ck;
                for(auto jj = 0ul; jj < nb[0]; jj += kernel_size ){
                    auto pa = ajj;
                    auto pb = bjj;
                    auto pc = cjj;
                    kernel_8x8(pc,nc,wc,pa,na,wa,pb,nb,wb);
                    bjj += wb_1;
                    cjj += wc_1;
                }
                ak += wa_1;
                bk += wb_0;
            }
            aii += wa_0;
            cii += wc_0;
        }
    }



    template <class SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    void mult_8nx8n(float*  c, SizeType const*  nc, SizeType const*  wc,
        float const*  a, SizeType const*  na, SizeType const*  wa,
        float const*  b, SizeType const*  nb, SizeType const*  wb
        ) noexcept 
    {
        // asm(".align 5");
        assert(na[1] == nb[0]);
        assert(na[0] == nc[0]);
        assert(nb[1] == nc[1]);

        if( na[0] == nb[1] && na[0] == 8 ){
            mult_8nx8n_helper_2(c,nc,wc,a,na,wa,b,nb,wb);
            return;
        }

        constexpr SizeType const block_size = 16;
        constexpr SizeType const nt[2] = {block_size, block_size};
        constexpr SizeType const wt[2] = {1,block_size};

        SizeType wa_0 = wa[0] * block_size;
        SizeType wa_1 = wa[1] * block_size;
        SizeType wb_0 = wb[0] * block_size;
        SizeType wb_1 = wb[1] * block_size;
        SizeType wc_0 = wc[0] * block_size;
        SizeType wc_1 = wc[1] * block_size;

        auto const irem = na[1] / block_size;
        auto const jrem = nb[0] / block_size;
        auto const krem = nb[0] / block_size;
        auto ai = a;
        auto bi = b;
        auto ci = c;

        for(auto i = 0ul; i < na[1] ; i += block_size){
            auto ak = ai;
            auto bk = bi;
            auto ck = ci;
            for(auto k = 0ul; k < nc[0] ; k += block_size){
                auto aj = ak;
                auto bj = bk;
                auto cj = ck;
                for(auto j = 0ul; j < nb[0] ; j += block_size){
                    auto ppa = aj;
                    auto ppb = bj;
                    mult_8nx8n_helper_2(cj, nt, wc, ppa, nt, wa, ppb, nt, wb);
                    bj += wb_1;
                    cj += wc_1;
                }
                ak += wa_1;
                bk += wb_0;
            }
            ci += wc_0;
            ai += wc_0;
        }
    }

    template <class SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    void mult_8nx8n_helper(float* c, SizeType const* nc, SizeType const* wc,
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb
        ) noexcept
    {

        constexpr SizeType const kernel_size = 8;

        for( auto ii = 0; (ii < na[1]); ii += kernel_size ){
            auto ajj = a + ii * wa[0];
            auto bjj = b;
            auto cjj = c + ii * wc[0];
            for(auto jj = 0; (jj < nb[0]); jj += kernel_size ){
                auto ak = ajj;
                auto bk = bjj + jj * wb[1];
                auto ck = cjj + jj * wc[1];
                for(auto k = 0; k < nb[0]; k += kernel_size){
                    auto pa = ak + k * wa[1];
                    auto pb = bk + k * wb[0];
                    auto pc = ck;
                    kernel_8x8(pc,nc,wc,pa,na,wa,pb,nb,wb);
                }
            }
        }
    }

    

    // template <class SizeType>
    // BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    // void mult_8nx8n(float* c, SizeType const* nc, SizeType const* wc,
    //     float const* a, SizeType const* na, SizeType const* wa,
    //     float const* b, SizeType const* nb, SizeType const* wb
    //     ) noexcept
    // {

    //     asm volatile(R"(.align 5)");
    //     constexpr SizeType const block_size = 16;

    //     // if( na[1] == 8 && nb[0] == 8 ){
    //     //     mult_8x8(c, nc, wc, a, na, wa, b, nb, wb);
    //     //     return;
    //     // }

    //     constexpr SizeType const nt[2] = {block_size, block_size};
    //     constexpr SizeType const nw[2] = {1, block_size};

    //     SizeType const wa_0 = wa[0] * block_size;
    //     SizeType const wa_1 = wa[1] * block_size;
    //     SizeType const wb_0 = wb[0] * block_size;
    //     SizeType const wb_1 = wb[1] * block_size;
    //     SizeType const wc_0 = wc[0] * block_size;
    //     SizeType const wc_1 = wc[1] * block_size;

    //     auto ai = a;
    //     auto bi = b;
    //     auto ci = c;

    //     auto const irem = na[1] / block_size;
    //     auto const jrem = nb[0] / block_size;
    //     auto const krem = nb[0] / block_size;
        
    //    {
    //         for( auto i = 0ul; i < irem; ++i){
    //             auto aj = ai;
    //             auto bj = bi;
    //             auto cj = ci;
    //             for( auto j = 0ul; j < jrem; ++j ){
    //                 auto ak = aj;
    //                 auto bk = bj;
    //                 auto ck = cj;
    //                 prefetchN<block_size,_MM_HINT_T1>(ck,wc[1]);
    //                 for( auto k = 0; k < krem; ++k ){
    //                     auto pa = ak;
    //                     auto pb = bk;
    //                     auto pc = ck;
    //                     // prefetchN<block_size,_MM_HINT_T0>(ak,wa[1]);
    //                     // prefetchN<block_size,_MM_HINT_T0>(bk,wb[1]);
    //                     mult_8nx8n_helper_2(pc,nt,wc,pa,nt,wa,pb,nt,wb);
    //                     ak += wa_1;
    //                     bk += wb_0;
    //                 }

    //                 bj += wb_1;
    //                 cj += wc_0;

    //             }
    //             ai += wa_0;
    //             ci += wc_0;
    //         }
    //     }

    // }

} // namespace boost::numeric::ublas::simd

#endif // BOOST_NUMERIC_UBLAS_SIMD_MULT_8nX8n_HPP
