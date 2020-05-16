#if !defined(BOOST_NUMERIC_UBLAS_SIMD_MULT_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_MULT_HPP

#include <boost/numeric/ublas/tensor/simd/detail/simd_helper.hpp>
// #include <unistd.h>
// #include <sys/sysctl.h>

// #include <boost/numeric/ublas/tensor/simd/matrix_mult/mult_8x8.hpp>
// #include <boost/numeric/ublas/tensor/simd/matrix_mult/mult_add_8x8.hpp>
#include <boost/numeric/ublas/tensor/simd/matrix_mult/mult_8nx8n.hpp>
// #include <boost/numeric/ublas/tensor/simd/matrix_mult/mult_16x16.hpp>


namespace boost::numeric::ublas::simd{
    
    template<typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    constexpr void pack_alignA(float* c, SizeType const* nc, SizeType const* wc,
        float const* a, SizeType const* na, SizeType const* wa, SizeType const offset) noexcept
    {
        auto ai = a + offset;
        auto ci = c;
        for( auto i = 0ul; i < na[1];  ++i ){
            auto aj = ai;
            auto cj = ci;
            for( auto j = 0ul; j < na[0]; ++j ){
                *cj = *aj;
                aj += wa[1];
                cj += wc[0];
            }
            ai += wa[0];
            ci += wc[1];
        }

    }
    template<typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    constexpr void pack_align(float* c, SizeType const* nc, SizeType const* wc,
        float const* a, SizeType const* na, SizeType const* wa) noexcept
    {
        auto ai = a;
        auto ci = c;
        for( auto i = 0ul; i < na[0];  ++i ){
            auto aj = ai;
            auto cj = ci;
            for( auto j = 0ul; j < na[1]; ++j ){
                *cj = *aj;
                aj += wa[0];
                cj += wc[0];
            }
            ai += wa[1];
            ci += wc[1];
        }

    }
    
    template<typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    constexpr void pack_alignB(float* c, SizeType const* nc, SizeType const* wc,
        float const* a, SizeType const* na, SizeType const* wa, SizeType const offset) noexcept
    {
        auto ai = a + offset * wa[1];
        auto ci = c;
        for( auto i = 0ul; i < na[1];  ++i ){
            auto aj = ai;
            auto cj = ci;
            for( auto j = 0ul; j < na[0]; ++j ){
                *cj = *aj;
                aj += wa[0];
                cj += wc[0];
            }
            ai += wa[1];
            ci += wc[1];
        }

    }

    struct kernel{
        static constexpr size_t const N = 8;
        static constexpr size_t const M = 8;

        inline constexpr size_t n() const noexcept{
            return 8;
        }

        inline constexpr size_t m() const noexcept{
            return 8;
        }

        template<typename SizeType>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void operator()(
            float* c, SizeType const* nc, SizeType const* wc, 
            float const* a, SizeType const* na, SizeType const* wa,
            float const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{
            
            // for(auto i = 0; i < nb[0]; ++i){
            //     for(auto j = 0; j < nb[1]; ++j){
            //         std::cout<<b[i * wb[0] + j * wb[1]]<<' ';
            //     }
            //     std::cout<<std::endl;
            // }
            // exit(0);

            auto aj = a;
            auto bj = b;
            auto cj = c;
            for( auto j = 0ul; j < na[1]; j += N ){
                auto nj = std::min( na[1] - j, N );
                auto ai = aj;
                auto bi = bj;
                auto ci = cj;
                for( auto i = 0ul; i < nb[0]; i += M){
                    auto ni = std::min( nb[0] - i, M );

                    SizeType const nta[] = { ni, nb[0] };
                    SizeType const ntb[] = { nb[0], nj };
                    SizeType const ntc[] = { ni, nj };

                    SizeType const wta[] = { 1, nb[0] };
                    SizeType const wtb[] = { 1, nj };

                    if( i + M >= wa[0] ) bi += N * nb[0];

                    my_kernel(
                        ci, ntc, wc,
                        ai, nta, wta,
                        bi, ntb, wtb
                    );

                    ai += wa[1];
                    ci += wc[1];
                    bi += wb[1];

                }

                cj += wc[0];
            }
        }

        template<typename SizeType>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void my_kernel(
            float* c, SizeType const* nc, SizeType const* wc, 
            float const* a, SizeType const* na, SizeType const* wa,
            float const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept
        {

            FReg r[16] = {_mm256_setzero_ps()};
            size_t k_iter = na[1] / 4;
            size_t k_rem = na[1] % 4;

            auto ai = a;
            auto bi = b;
            prefetch(c,wc[1]);

            // Iteration 0
            r[0].y = _mm256_load_ps(ai);
            r[1].y = _mm256_load_ps(ai + wa[1]);

            _mm_prefetch(ai + wa[1], _MM_HINT_T0);
                        
            // print(r[4].y);
            // print(r[5].y);
            // print(r[6].y);
            // print(r[7].y);
            // print(r[8].y);
            // print(r[9].y);
            // print(r[10].y);
            // print(r[11].y);
            // print(r[12].y);
            // print(r[13].y);
            // print(r[14].y);
            // print(r[15].y);
            // exit(0);

            std::cout<<*bi<<' '<<wb[0]<<'\n';

            r[2].y = _mm256_broadcast_ss(bi);
            r[3].y = _mm256_broadcast_ss(bi + wb[0]);

            r[4].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[4].y);
            r[5].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[5].y);
            r[6].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[6].y);
            r[7].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[7].y);

            r[2].y = _mm256_broadcast_ss(bi + wb[0] * 2);
            r[3].y = _mm256_broadcast_ss(bi + wb[0] * 3);

            r[8].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[8].y);
            r[9].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[9].y);
            r[10].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[10].y);
            r[11].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[11].y);

            r[2].y = _mm256_broadcast_ss(bi + wb[0] * 4);
            r[3].y = _mm256_broadcast_ss(bi + wb[0] * 5);

            r[12].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[12].y);
            r[13].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[13].y);
            r[14].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[14].y);
            r[15].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[15].y);


            bi += wb[1];
            
            r[0].y = _mm256_load_ps(ai + wa[1] * 2);
            r[1].y = _mm256_load_ps(ai + wa[1] * 3);
            // Iteration 1
            
            _mm_prefetch(ai + wa[1], _MM_HINT_T0);
            
            r[2].y = _mm256_broadcast_ss(bi);
            r[3].y = _mm256_broadcast_ss(bi + wb[0]);

            r[4].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[4].y);
            r[5].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[5].y);
            r[6].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[6].y);
            r[7].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[7].y);

            r[2].y = _mm256_broadcast_ss(bi + wb[0] * 2);
            r[3].y = _mm256_broadcast_ss(bi + wb[0] * 3);

            r[8].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[8].y);
            r[9].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[9].y);
            r[10].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[10].y);
            r[11].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[11].y);

            r[2].y = _mm256_broadcast_ss(bi + wb[0] * 4);
            r[3].y = _mm256_broadcast_ss(bi + wb[0] * 5);

            r[12].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[12].y);
            r[13].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[13].y);
            r[14].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[14].y);
            r[15].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[15].y);
            
            
            bi += wb[1];
  
            r[0].y = _mm256_load_ps(ai + wa[1] * 4);
            r[1].y = _mm256_load_ps(ai + wa[1] * 5);
            // Iteration 2
            
            _mm_prefetch(ai + wa[1], _MM_HINT_T0);
            
            r[2].y = _mm256_broadcast_ss(bi);
            r[3].y = _mm256_broadcast_ss(bi + wb[0]);

            r[4].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[4].y);
            r[5].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[5].y);
            r[6].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[6].y);
            r[7].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[7].y);

            r[2].y = _mm256_broadcast_ss(bi + wb[0] * 2);
            r[3].y = _mm256_broadcast_ss(bi + wb[0] * 3);

            r[8].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[8].y);
            r[9].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[9].y);
            r[10].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[10].y);
            r[11].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[11].y);

            r[2].y = _mm256_broadcast_ss(bi + wb[0] * 4);
            r[3].y = _mm256_broadcast_ss(bi + wb[0] * 5);

            r[12].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[12].y);
            r[13].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[13].y);
            r[14].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[14].y);
            r[15].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[15].y);
            
            
            bi += wb[1];
  
            r[0].y = _mm256_load_ps(ai + wa[1] * 6);
            r[1].y = _mm256_load_ps(ai + wa[1] * 7);
            // Iteration 3
            
            _mm_prefetch(ai + wa[1], _MM_HINT_T0);
            
            r[2].y = _mm256_broadcast_ss(bi);
            r[3].y = _mm256_broadcast_ss(bi + wb[0]);

            r[4].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[4].y);
            r[5].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[5].y);
            r[6].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[6].y);
            r[7].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[7].y);

            r[2].y = _mm256_broadcast_ss(bi + wb[0] * 2);
            r[3].y = _mm256_broadcast_ss(bi + wb[0] * 3);

            r[8].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[8].y);
            r[9].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[9].y);
            r[10].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[10].y);
            r[11].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[11].y);

            r[2].y = _mm256_broadcast_ss(bi + wb[0] * 4);
            r[3].y = _mm256_broadcast_ss(bi + wb[0] * 5);

            r[12].y = _mm256_fmadd_ps(r[0].y,r[2].y,r[12].y);
            r[13].y = _mm256_fmadd_ps(r[1].y,r[2].y,r[13].y);
            r[14].y = _mm256_fmadd_ps(r[0].y,r[3].y,r[14].y);
            r[15].y = _mm256_fmadd_ps(r[1].y,r[3].y,r[15].y);

            // print(r[4].y);
            // print(r[5].y);
            // print(r[6].y);
            // print(r[7].y);
            // print(r[8].y);
            // print(r[9].y);
            // print(r[10].y);
            // print(r[11].y);
            // print(r[12].y);
            // print(r[13].y);
            // print(r[14].y);
            // print(r[15].y);
            exit(0);

        }

    };

    struct partition{
        inline constexpr size_t k() const noexcept{
            return 384;
        }
        inline constexpr size_t m() const noexcept{
            return 128;
        }
        inline constexpr size_t n() const noexcept{
            return 4096;
        }
    };

    template<typename KernelType>
    struct auto_partition{

        auto_partition(){
            auto Nl1 = 64;
            auto Nl2 = 256;
            auto Wl2 = 4;
            auto Cl1 = 32768;
            auto Cl2 = 262144;
            auto Cl3 = 16777216;

        }

        inline constexpr size_t k() const noexcept{
            return m_k;
        }
        inline constexpr size_t m() const noexcept{
            return m_m;
        }
        inline constexpr size_t n() const noexcept{
            return m_n;
        }
    private:
        KernelType m_ker;
        size_t m_k;
        size_t m_m;
        size_t m_n;
    };

    template<typename KernelType = kernel, typename PartitionType = partition, typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void mtm(float* c, SizeType const* nc, SizeType const* wc, 
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb,
        KernelType const& ker = kernel{}, PartitionType const& par = partition{}
        ) noexcept
    {
        
        float* packA = align_allocator<float>{}.allocate( par.k() * par.m() );
        float* packB = align_allocator<float>{}.allocate( par.k() * par.n() );
        auto ppackA = packA;
        auto ppackB = packB;

        // A(m,k) x B(k,n) = C(m,n)
        assert(nc[0] == na[0]);
        assert(nc[1] == nb[1]);
        assert(na[1] == nb[0]);

        auto j_total = nb[1] / par.n();
        auto j_rem = nb[1] % par.n();
        auto i_total = na[0] / par.m();
        auto i_rem = na[0] % par.m();
        auto k_total = nb[0] / par.k();
        auto k_rem = nb[0] % par.k();

        auto aj = a;
        auto bj = b;
        auto cj = c;

        //     |par.n()|    N - par.n() |      |par.n()|    N - par.n() |
        // C = |-------|----------------|, B = |-------|----------------|
        for( auto jc = 0ul; jc < nb[1]; jc += par.n() ){
            auto jb = std::min(nb[1] - jc, par.n());

            auto ak = aj;
            auto bk = bj;
            auto ck = cj;
            //     |par.k()|    K - par.k() |, B = |------------------------| 
            // A = |-------|----------------|,     |                        |
            //                                     |                        |
            //                                     --------------------------
            //                                     |  K - par.k()           |
            //                                     --------------------------
            //                                     |                        |
            //                                     |------------------------|
            for( auto k = 0ul; k < nb[0]; k += par.k() ){
                auto kb = std::min( nb[0] - k, par.k() );
                auto aii = ak;
                auto bii = bk;
                auto temp_pb = ppackB;
                // Pack B
                for( auto ii = 0ul; ii < jb; ii += ker.n() ){
                    auto pb = bii;
                    auto iiw = std::min( jb - ii, ker.n() );
                    SizeType const nt[] = { kb, iiw };
                    SizeType const wt[] = { 1, iiw };
                    pack_align(
                        temp_pb ,nt,wt,
                        pb, nt, wb
                    );
                    bii += wa[0] * iiw ;
                    temp_pb += kb * iiw;
                }

                auto ai = ak;
                auto bi = bk;
                auto ci = ck;

                for( auto ic = 0ul; ic < na[0]; ic += par.m() ){
                    auto ib = std::min( na[0] - ic, par.m() );

                    auto aiii = ai;
                    auto biii = bi;
                    auto ciii = ci;
                    auto temp_pa = ppackA;

                    // Pack A
                    for( auto ii = 0ul; ii < ib; ii += ker.m() ){
                        auto iiw = std::min( ib - ii, ker.m() );
                        SizeType const nt[] = { kb, iiw};
                        SizeType const wt[] = { 1, iiw };

                        auto pa = aiii;

                        pack_align(
                            temp_pa, nt, wt,
                            pa, nt, wa
                        );
                        aiii += wa[0] * iiw;
                        temp_pa += kb * iiw;
                    }

                    SizeType const nta[] = { jb, kb};
                    SizeType const wta[] = { 1, kb };

                    SizeType const ntb[] = { kb, jb};
                    SizeType const wtb[] = { 1, jb };

                    SizeType const ntc[] = { jb, jb};

                    ker(
                        c, ntc, wc,
                        ai, nta, wta,
                        bi, ntb, wtb
                    );
                }

                ak += wa[1] * par.k();
                bk += wa[0] * par.k();

            }

            bj += wb[1] * par.n();
            cj += wc[1] * par.n();

        }
        
        // align_allocator<float>{}.deallocate(packA,0);
        // align_allocator<float>{}.deallocate(packB,0);
    }


} // namespace namespace boost::numeric::ublas::simd



#endif // BOOST_NUMERIC_UBLAS_SIMD_MULT_HPP
