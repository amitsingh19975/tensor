#if !defined(VECTOR_MULT_HPP)
#define VECTOR_MULT_HPP

#include <boost/numeric/ublas/tensor/simd/detail/simd_helper.hpp>


void print(__m256 d, int N ){
    std::cout<<"ymm"<<N<<"[ ";
    std::cout<<d[0]<<' '<<d[1]<<' '<<d[2]<<' '<<d[3]<<' ';
    std::cout<<d[4]<<' '<<d[5]<<' '<<d[6]<<' '<<d[7]<<" ]\n";
}

#define PRINT(N) print(ymm##N,N);

namespace boost::numeric::ublas::simd{
    
    struct partition{
        inline constexpr size_t k() const noexcept{
            return 48;
        }
        inline constexpr size_t m() const noexcept{
            return 88;
        }
        inline constexpr size_t n() const noexcept{
            return 1;
        }

        // template<typename SizeType>
        // BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        // float const* pack(float const* a, SizeType const* na, SizeType const* wa, SizeType const offset = 0) noexcept{
        //     auto ai = a;
        //     auto ci = buffA + off;

        //     for( auto i = na[0]; i > 0 ;  --i ){
        //         _mm_prefetch(ai, _MM_HINT_T0);
        //         _mm_prefetch(ci, _MM_HINT_T0);
        //         auto aj = ai;
        //         auto cj = ci;
        //         _mm256_store_ps(cj,_mm256_load_ps(aj));
        //         ai += wa[1];
        //         ci += 8;
        //     }
        //     return buffA;
        // }

        // ~partition(){
        //     align_allocator<float>{}.deallocate(buffA,0);
        // }
        
        // float* buffA{ align_allocator<float>{}.allocate( k() * m() ) };
    };

    struct Kernel{
        static constexpr size_t M = 8;
        static constexpr size_t N = 8;

        template<typename SizeType>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void operator()(
            float* c, SizeType const* nc, SizeType const* wc,
            float const* a, SizeType const* na, SizeType const* wa,
            float const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{

            auto aj = a;
            auto bj = b;
            auto cj = c;
            auto const wa_0 = wa[0] * M;
            auto const wc_0 = wc[0] * M;

            auto i_iter = na[0] / M;
            auto i_rem = na[0] % M;

            for(auto i = i_iter; i > 0 ; --i ){
                SizeType const nta[] = { M, na[1]};

                SizeType const ntb[] = { nb[0], 1};

                SizeType const ntc[] = { M , 1};


                kernel_helper_asm(
                    cj, ntc, wc,
                    aj, nta, wa,
                    bj, ntb, wb
                );
                aj += wa_0;
                cj += wc_0;
            }

            // if( i_rem != 0 ){
                
            //     SizeType const nta[] = { i_rem, na[1]};

            //     SizeType const ntb[] = { nb[0], 1};
            //     SizeType const wtb[] = { 1, 1 };

            //     SizeType const ntc[] = { i_rem , 1};


            //     kernel_helper_asm(
            //         cj, ntc, wc,
            //         aj, nta, wa,
            //         bj, ntb, wtb
            //     );

            //     aj += wa[0] * i_rem;
            //     cj += wc[0] * i_rem;
            // }

        }

    private:
        template<typename SizeType>
        void kernel_helper(
            float* c, SizeType const* nc, SizeType const* wc,
            float const* a, SizeType const* na, SizeType const* wa,
            float const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{
            
            // std::cout<<std::endl;
            // for(auto i = 0; i < nc[0]; ++i){
            //     for(auto j = 0; j < nc[1]; ++j){
            //         std::cout<<c[i * wc[0] + j * wc[1]]<<' ';
            //     }
            //     std::cout<<std::endl;
            // }
            
            // std::cout<<std::endl;
            // for(auto i = 0; i < na[0]; ++i){
            //     for(auto j = 0; j < na[1]; ++j){
            //         std::cout<<a[i * wa[0] + j * wa[1]]<<' ';
            //     }
            //     std::cout<<std::endl;
            // }
            
            // std::cout<<std::endl;
            // for(auto i = 0; i < nb[0]; ++i){
            //     for(auto j = 0; j < nb[1]; ++j){
            //         std::cout<<b[i * wb[0] + j * wb[1]]<<' ';
            //     }
            //     std::cout<<std::endl;
            // }

            __m256 ymm[16] = {_mm256_setzero_ps()};

            auto k_iter = na[1] / 8;
            auto k_rem = na[1] % 8;

            prefetch(c,wc[0]);
            for( auto k = 0ul; k < k_iter; ++k ){
                ymm[0] = _mm256_load_ps(a + wa[1] * 0);
                ymm[1] = _mm256_load_ps(a + wa[1] * 1);
                ymm[2] = _mm256_load_ps(a + wa[1] * 2);
                ymm[3] = _mm256_load_ps(a + wa[1] * 3);
                ymm[4] = _mm256_load_ps(a + wa[1] * 4);
                ymm[5] = _mm256_load_ps(a + wa[1] * 5);
                ymm[6] = _mm256_load_ps(a + wa[1] * 6);
                ymm[7] = _mm256_load_ps(a + wa[1] * 7);
                ymm[8] = _mm256_load_ps(b + wb[1] * 0);
                ymm[9] = _mm256_moveldup_ps(ymm[8]);
                ymm[10] = _mm256_movehdup_ps(ymm[8]);

                // _mm_prefetch(a + wa[1] * 8, _MM_HINT_T0);
                // _mm_prefetch(b + wa[1] * 8, _MM_HINT_T0);

                ymm[15] = _mm256_fmadd_ps(ymm[0], _mm256_permute2f128_ps(ymm[9],ymm[9],_MM_SHUFFLE(0,0,0,0)), ymm[15]);
                ymm[15] = _mm256_fmadd_ps(ymm[1], _mm256_permute2f128_ps(ymm[10],ymm[10],_MM_SHUFFLE(0,0,0,0)), ymm[15]);
                ymm[15] = _mm256_fmadd_ps(ymm[2], _mm256_permute2f128_ps(ymm[9],ymm[9],_MM_SHUFFLE(1,1,1,1)), ymm[15]);
                ymm[15] = _mm256_fmadd_ps(ymm[3], _mm256_permute2f128_ps(ymm[10],ymm[10],_MM_SHUFFLE(1,1,1,1)), ymm[15]);
                ymm[15] = _mm256_fmadd_ps(ymm[4], _mm256_permute2f128_ps(ymm[9],ymm[9],_MM_SHUFFLE(2,2,2,2)), ymm[15]);
                ymm[15] = _mm256_fmadd_ps(ymm[5], _mm256_permute2f128_ps(ymm[10],ymm[10],_MM_SHUFFLE(2,2,2,2)), ymm[15]);
                ymm[15] = _mm256_fmadd_ps(ymm[6], _mm256_permute2f128_ps(ymm[9],ymm[9],_MM_SHUFFLE(3,3,3,3)), ymm[15]);
                ymm[15] = _mm256_fmadd_ps(ymm[7], _mm256_permute2f128_ps(ymm[10],ymm[10],_MM_SHUFFLE(3,3,3,3)), ymm[15]);
                

                b += wb[1] * 8;
                a += wa[1] * 8;
            }
            // std::cout<<na[1]<<'\n';
            // for(auto k = 0; k < k_rem; ++k){
            //     ymm[k] = _mm256_load_ps(a);
            //     ymm[k + 1] = _mm256_broadcast_ss(b);
            //     _mm_prefetch(a + wa[1], _MM_HINT_T0);
            //     _mm_prefetch(b + wb[1], _MM_HINT_T0);

            //     ymm[8] = _mm256_fmadd_ps(ymm[k], ymm[k + 1], ymm[8]);
            //     // PRINT(k);
            //     // PRINT(k + 1);
            //     // PRINT(8);
            //     a += wa[1];
            //     b += wb[1];
            // }

            _mm256_storeu_ps(c,ymm[15]);
        }
        template<typename SizeType>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void kernel_helper_asm(
            float* c, SizeType const* nc, SizeType const* wc,
            float const* a, SizeType const* na, SizeType const* wa,
            float const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{
            
            // std::cout<<"C Address: "<<c<<'\n';
            // std::cout<<std::endl;
            // for(auto i = 0; i < nc[0]; ++i){
            //     for(auto j = 0; j < nc[1]; ++j){
            //         std::cout<<std::fixed<<c[i * wc[0] + j * wc[1]]<<' ';
            //     }
            //     std::cout<<std::endl;
            // }
            
            // std::cout<<std::endl;
            // for(auto i = 0; i < na[0]; ++i){
            //     for(auto j = 0; j < na[1]; ++j){
            //         std::cout<<a[i * wa[0] + j * wa[1]]<<' ';
            //     }
            //     std::cout<<std::endl;
            // }
            
            // std::cout<<std::endl;
            // for(auto i = 0; i < nb[0]; ++i){
            //     for(auto j = 0; j < nb[1]; ++j){
            //         std::cout<<b[i * wb[0] + j * wb[1]]<<' ';
            //     }
            //     std::cout<<std::endl;
            // }

            auto k_iter = na[1] / 8;
            auto k_rem = na[1] % 8;

            _mm_prefetch(c,_MM_HINT_T0);
            
            auto ak = a;
            auto bk = b;
            auto ck = c;
            // __m256 ymm0, ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9;

            asm volatile (R"(
                .align 5
                movq %[a], %%rax
                movq %[b], %%rbx
                movq %[c], %%rcx
                movq %[wa], %%r9
                movq %[k_it], %%r10
                movq %[k_r], %%r13
                movq %[wa], %%r11
                movq %[wa], %%r12
                imul $32, %%r11
                imul $12, %%r12

                testq $0,%%r10
                jne LKERNEL8x8LOOPZERO%=
                vmovaps (%%rcx), %%ymm15
            LKERNEL8x8LOOP%=:
                prefetcht0 (%%rax,%%r11)
                prefetcht0 32(%%rbx)

                vmovaps (%%rbx), %%ymm8
                
                vperm2f128 $0, %%ymm8, %%ymm8, %%ymm9
                vperm2f128 $0x11, %%ymm8, %%ymm8, %%ymm10

                vpermilps $0, %%ymm9, %%ymm11
                vpermilps $0x55, %%ymm9, %%ymm12
                vpermilps $0xAA, %%ymm9, %%ymm13
                vpermilps $0xFF, %%ymm9, %%ymm14

                vfmadd231ps (%%rax), %%ymm11, %%ymm15
                vfmadd231ps (%%rax, %%r9, 4), %%ymm12, %%ymm15
                vfmadd231ps (%%rax, %%r9, 8), %%ymm13, %%ymm15
                
                addq %%r12, %%rax
                vfmadd231ps (%%rax), %%ymm14, %%ymm15

                vpermilps $0, %%ymm10, %%ymm11
                vpermilps $0x55, %%ymm10, %%ymm12
                vpermilps $0xAA, %%ymm10, %%ymm13
                vpermilps $0xFF, %%ymm10, %%ymm14

                vfmadd231ps (%%rax, %%r9, 4), %%ymm11, %%ymm15
                vfmadd231ps (%%rax, %%r9, 8), %%ymm12, %%ymm15
                
                addq %%r12, %%rax
                vfmadd231ps (%%rax), %%ymm13, %%ymm15
                vfmadd231ps (%%rax, %%r9, 4), %%ymm14, %%ymm15
                
                leaq (%%rax, %%r9, 8), %%rax
                addq $32, %%rbx

                decq %%r10
                jne LKERNEL8x8LOOP%=
                
                #testq $0,%%r13
                #jne LKERNEL8x8LOOPZERO%=

            LKERNEL8x8LOOPREM%=:



                #jne LKERNEL8x8LOOPREM%=

                #jmp LKERNEL8x8LOOPZERO%=
            LKERNEL8x8LOOPZERO%=:
                vmovaps %%ymm15, (%%rcx)

            )"
                : [c] "=m" (c)
                // , [y0] "=m" (ymm0)
                // , [y1] "=m" (ymm1)
                // , [y2] "=m" (ymm2)
                // , [y3] "=m" (ymm3)
                // , [y4] "=m" (ymm4)
                // , [y5] "=m" (ymm5)
                // , [y6] "=m" (ymm6)
                // , [y7] "=m" (ymm7)
                // , [y8] "=m" (ymm8)
                // , [cym] "=m" (ymm9)
                : [a] "m" (a)
                , [b] "m" (b)
                , [k_it] "r" (k_iter)
                , [k_r] "r" (k_rem)
                , [wa] "m" (wa[1])
                : "%ymm0", "%ymm1", "%ymm2", "%ymm3"
                , "%ymm4", "%ymm5", "%ymm6", "%ymm7"
                , "%ymm8", "%ymm9", "%ymm10", "%ymm11"
                , "%ymm12", "%ymm13", "%ymm14", "%ymm15"
                , "%rax", "%rbx", "%rcx", "%r9", "%r10", "%r11", "%r12", "%r13"
            );

            // std::cout<<"----------------------\n";

            // PRINT(0);
            // PRINT(1);
            // PRINT(2);
            // PRINT(3);
            // PRINT(4);
            // PRINT(5);
            // PRINT(6);
            // PRINT(7);
            // PRINT(8);
            // PRINT(9);

            // std::cout<<"----------------------\n";

        }
    };

    template<typename KernelType = Kernel, typename PartitionType = partition, typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void mtv2(
        float* c, SizeType const* nc, SizeType const* wc,
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb,
        KernelType ker = {}, PartitionType par = {}
    ) noexcept{

        auto ak = a;
        auto bk = b;
        auto ck = c;

        float* packedA = align_allocator<float>{}.allocate( par.k() * par.m() );

        for(auto k = 0ul; k < nb[0]; k += par.k()){
            auto kb = std::min(nb[0] - k, par.k());

            auto ai = ak;
            auto bi = bk;
            auto ci = ck;

            for( auto i = 0ul; i < na[0]; i += par.m() ){
                auto ib = std::min(na[0] - i, par.m());

                auto aii = ai;
                auto bii = bi;
                auto cii = ci;
                auto ppackedA = packedA;
                
                for( auto ii = 0; ii < ib; ii += KernelType::M ){
                    auto iib = std::min( ib - ii, KernelType::M );
                    SizeType const nt[] = {kb, iib};
                    SizeType const wt[] = {1ul, iib};

                    pack_align(
                        ppackedA, nt, wt,
                        aii, nt, wa
                    );

                    aii += wa[0] * iib;
                    ppackedA += kb * iib;
                }

                SizeType const nta[] = { ib, kb};

                SizeType const ntb[] = { kb, 1};
                SizeType const wtb[] = { 1, 1 };

                SizeType const ntc[] = { ib, 1};
                ker(
                    ci, ntc, wc,
                    packedA, nta, wa,
                    bi, ntb, wtb
                );

                ai += wa[0] * par.m();
                ci += wa[0] * par.m();
            }

            ak += wa[1] * par.k();
            bk += wb[0] * par.k();
        }

        align_allocator<float>{}.deallocate(packedA,0);

    }

    template<typename KernelType = Kernel, typename PartitionType = partition, typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void mtv(
        float* c, SizeType const* nc, SizeType const* wc,
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb,
        KernelType ker = {}, PartitionType par = {}
    ) noexcept{

        auto ai = a;
        auto bi = b;
        auto ci = c;

        

        auto wa_1 = wa[1] * par.k();
        auto wa_0 = wa[0] * par.m();
        auto wb_0 = wb[0] * par.k();
        auto wc_0 = wc[0] * par.m();

        auto const K = par.k();
        auto const M = par.m();

        #pragma omp parallel for schedule(dynamic,1)
        for( auto i = 0ul; i < na[0]; i += M ){
            auto const ib = std::min(na[0] - i, M);

            auto ak = a + i * wa[0];
            auto bk = b;
            auto ck = c + i * wc[0];


            for(auto k = 0ul; k < nb[0]; k += K){
                auto const kb = std::min(nb[0] - k, K);
                auto aii = ak;
                auto bii = bk;
                auto cii = ck;

                SizeType const nta[] = { ib, kb};

                SizeType const ntb[] = { kb, 1};

                SizeType const ntc[] = { ib, 1};
                {
                    ker(
                        ck, ntc, wc,
                        ak, nta, wa,
                        bk, ntb, wb
                    );
                }
                ak += wa_1;
                bk += wb_0;
            }

            ai += wa_0;
            ci += wc_0;
        }

    }
    

} // namespace boost::numeric::ublas::simd


#endif // VECTOR_MULT_HPP
