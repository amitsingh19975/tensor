#if !defined(VECTOR_MULT_HPP)
#define VECTOR_MULT_HPP

#include <boost/numeric/ublas/tensor/simd/detail/simd_helper.hpp>
#include <omp.h>

template<typename T>
void print(T d, const char* N ){
    std::cout<<N<<"[ ";
    std::cout<<d[0]<<' '<<d[1]<<' '<<d[2]<<' '<<d[3]<<' ';
    std::cout<<d[4]<<' '<<d[5]<<' '<<d[6]<<' '<<d[7]<<" ]\n";
}

#define PRINT(N) print(N.y,#N);

namespace boost::numeric::ublas::simd{
    
    struct partition{
        inline constexpr size_t k(size_t K) const noexcept{
            if( K < 2000 ){
                return 96;
            }else{
                return 8;
            }
        }
        inline constexpr size_t m(size_t M) const noexcept{
            if( M > 2000 ){
                auto nm = M / 4;
                auto rem = nm % 8;
                return nm - rem;
            }else{
                return 88;
            }
        }
        inline constexpr size_t n(size_t) const noexcept{
            return 1;
        }
    };

    struct Kernel{
        static constexpr size_t M = 8;
        static constexpr size_t N = 8;

        template<typename SizeType>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void operator()(
            float* __restrict__ c, SizeType const* __restrict__ nc, SizeType const* __restrict__ wc,
            float const* __restrict__ a, SizeType const* __restrict__ na, SizeType const* __restrict__ wa,
            float const* __restrict__ b, SizeType const* __restrict__ nb, SizeType const* __restrict__ wb
        ) const noexcept{

            auto ai = a;
            auto bi = b;
            auto ci = c;
            auto const wa_0 = wa[0] * M;
            auto const wc_0 = wc[0] * M;
            
            auto const wa_1 = wa[1] * N;
            auto const wc_1 = wc[1] * N;

            auto i_iter = na[0] / M;
            auto i_rem = na[0] % M;

            if( i_iter ){
                for(auto i = i_iter; i > 0 ; --i ){

                    SizeType const nta[] = { M, na[1]};

                    SizeType const ntb[] = { nb[0], 1};

                    SizeType const ntc[] = { M , 1};

                    auto aj = ai;
                    auto bj = bi;
                    auto cj = ci;

                    kernel_helper_8xn(
                        cj, ntc, wc,
                        aj, nta, wa,
                        bj, ntb, wb
                    );

                    ai += wa_0;
                    ci += wc_0;
                }
            }

            if( i_rem ){
                SizeType const nta[] = { i_rem, na[1]};

                SizeType const ntb[] = { nb[0], 1};

                SizeType const ntc[] = { i_rem , 1};

                auto aj = ai;
                auto bj = bi;
                auto cj = ci;

                kernel_helper_nxn(
                    cj, ntc, wc,
                    aj, nta, wa,
                    bj, ntb, wb
                );
            }


        }

    private:
        
        template<typename SizeType>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void kernel_helper_8xn(
            float* __restrict__ c, SizeType const* __restrict__ nc, SizeType const* __restrict__ wc,
            float const* __restrict__ a, SizeType const* __restrict__ na, SizeType const* __restrict__ wa,
            float const* __restrict__ b, SizeType const* __restrict__ nb, SizeType const* __restrict__ wb
        ) const noexcept{
            
            FReg r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, res, r10, r11;
            auto k_iter = na[1] / 8;
            auto k_rem = na[1] % 8;
            

            _mm_prefetch(c,_MM_HINT_T0);
            
            auto ak = a;
            auto bk = b;
            auto ck = c;

            res.y = _mm256_load_ps(ck);
            if( k_iter ){
                for( auto k = k_iter; k > 0 ; --k ){
                    r0.y = _mm256_load_ps(ak + wa[1] * 0);
                    r1.y = _mm256_load_ps(ak + wa[1] * 1);
                    r2.y = _mm256_load_ps(ak + wa[1] * 2);
                    r3.y = _mm256_load_ps(ak + wa[1] * 3);
                    r4.y = _mm256_load_ps(ak + wa[1] * 4);
                    r5.y = _mm256_load_ps(ak + wa[1] * 5);
                    r6.y = _mm256_load_ps(ak + wa[1] * 6);
                    r7.y = _mm256_load_ps(ak + wa[1] * 7);

                    r9.y = _mm256_load_ps(bk);

                    r10.y = _mm256_permute2f128_ps(r9.y,r9.y,0);
                    r11.y = _mm256_permute2f128_ps(r9.y,r9.y,0x11);

                    r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(0,0,0,0));
                    res.y = _mm256_fmadd_ps(r0.y,r8.y,res.y);

                    r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(1,1,1,1));
                    res.y = _mm256_fmadd_ps(r1.y,r8.y,res.y);

                    r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(2,2,2,2));
                    res.y = _mm256_fmadd_ps(r2.y,r8.y,res.y);

                    r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(3,3,3,3));
                    res.y = _mm256_fmadd_ps(r3.y,r8.y,res.y);

                    r8.y = _mm256_permute_ps(r11.y,_MM_SHUFFLE(0,0,0,0));
                    res.y = _mm256_fmadd_ps(r4.y,r8.y,res.y);

                    r8.y = _mm256_permute_ps(r11.y,_MM_SHUFFLE(1,1,1,1));
                    res.y = _mm256_fmadd_ps(r5.y,r8.y,res.y);

                    r8.y = _mm256_permute_ps(r11.y,_MM_SHUFFLE(2,2,2,2));
                    res.y = _mm256_fmadd_ps(r6.y,r8.y,res.y);

                    r8.y = _mm256_permute_ps(r11.y,_MM_SHUFFLE(3,3,3,3));
                    res.y = _mm256_fmadd_ps(r7.y,r8.y,res.y);

                    ak += wa[1] * 8;
                    bk += 8;
                }
            }
             
            if( k_rem ){
                k_iter = k_rem / 4;
                k_rem = k_rem % 4;
                
                if( k_iter ){
                        
                    r0.y = _mm256_load_ps(ak + wa[1] * 0);
                    r1.y = _mm256_load_ps(ak + wa[1] * 1);
                    r2.y = _mm256_load_ps(ak + wa[1] * 2);
                    r3.y = _mm256_load_ps(ak + wa[1] * 3);

                    r8.y = _mm256_broadcast_ss(bk);
                    res.y = _mm256_fmadd_ps(r0.y,r8.y,res.y);

                    r8.y = _mm256_broadcast_ss(bk + 1);
                    res.y = _mm256_fmadd_ps(r1.y,r8.y,res.y);

                    r8.y = _mm256_broadcast_ss(bk + 2);
                    res.y = _mm256_fmadd_ps(r2.y,r8.y,res.y);

                    r8.y = _mm256_broadcast_ss(bk + 3);
                    res.y = _mm256_fmadd_ps(r3.y,r8.y,res.y);

                    ak += wa[1] * 4;
                    bk += 4;
                }
                if( k_rem ){
                    for(auto k = k_rem; k > 0 ; --k){
                        r0.y = _mm256_load_ps(ak);
                        r8.y = _mm256_broadcast_ss(bk);
                        res.y = _mm256_fmadd_ps(r0.y,r8.y,res.y);
                        ak += wa[1];
                        ++bk;
                    }
                }

            }

            _mm256_store_ps(ck,res.y);
        }
        
        template<typename SizeType>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void kernel_helper_nxn(
            float* __restrict__ c, SizeType const* __restrict__ nc, SizeType const* __restrict__ wc,
            float const* __restrict__ a, SizeType const* __restrict__ na, SizeType const* __restrict__ wa,
            float const* __restrict__ b, SizeType const* __restrict__ nb, SizeType const* __restrict__ wb
        ) const noexcept{
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
            
            FReg r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, res;
            auto k_iter = na[1] / 8;
            auto k_rem = na[1] % 8;
            
            auto m_iter = na[0] / 8;
            auto m_rem = na[0] % 8;

            _mm_prefetch(c,_MM_HINT_T0);
            
            auto ak = a;
            auto bk = b;
            auto ck = c;

            if( m_iter){
                kernel_helper_8xn(c,nc,wc,a,na,wa,b,nb,wb);
                ck += 8;
                ak = a + wa[0] * 8;
            }
            if( m_rem >= 4 ){
                m_iter = m_rem / 4;
                m_rem = m_rem % 4;

                if( m_iter ){
                    kernel_helper_4xn(c,nc,wc,a,na,wa,b,nb,wb);
                    ak = a + wa[0] * 4;
                    ck += 4;
                }
            }


            if( m_rem ){
                __m128i mask;
                if( m_rem == 1 ) mask = _mm_setr_epi32(-1,1,1,1);
                else if( m_rem == 2 ) mask = _mm_setr_epi32(-1,-1,1,1);
                else if( m_rem == 3 ) mask = _mm_setr_epi32(-1,-1,-1,1);
                
                res.x[0] = _mm_maskload_ps(ck,mask);

                if( k_iter ){
                    for( auto i = 0ul; i < k_iter; ++i ){
                        r0.x[0] = _mm_maskload_ps(ak + wa[1] * 0,mask);
                        r1.x[0] = _mm_maskload_ps(ak + wa[1] * 1,mask);
                        r2.x[0] = _mm_maskload_ps(ak + wa[1] * 2,mask);
                        r3.x[0] = _mm_maskload_ps(ak + wa[1] * 3,mask);
                        r4.x[0] = _mm_maskload_ps(ak + wa[1] * 4,mask);
                        r5.x[0] = _mm_maskload_ps(ak + wa[1] * 5,mask);
                        r6.x[0] = _mm_maskload_ps(ak + wa[1] * 6,mask);
                        r7.x[0] = _mm_maskload_ps(ak + wa[1] * 7,mask);

                        r9.y = _mm256_load_ps(bk);

                        r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(0,0,0,0));
                        res.x[0] = _mm_fmadd_ps(r8.x[0],r0.x[0],res.x[0]);

                        r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(1,1,1,1));
                        res.x[0] = _mm_fmadd_ps(r8.x[0],r1.x[0],res.x[0]);

                        r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(2,2,2,2));
                        res.x[0] = _mm_fmadd_ps(r8.x[0],r2.x[0],res.x[0]);

                        r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(3,3,3,3));
                        res.x[0] = _mm_fmadd_ps(r8.x[0],r3.x[0],res.x[0]);

                        r8.x[0] = _mm_permute_ps(r9.x[1],_MM_SHUFFLE(0,0,0,0));
                        res.x[0] = _mm_fmadd_ps(r8.x[0],r4.x[0],res.x[0]);

                        r8.x[0] = _mm_permute_ps(r9.x[1],_MM_SHUFFLE(1,1,1,1));
                        res.x[0] = _mm_fmadd_ps(r8.x[0],r5.x[0],res.x[0]);

                        r8.x[0] = _mm_permute_ps(r9.x[1],_MM_SHUFFLE(2,2,2,2));
                        res.x[0] = _mm_fmadd_ps(r8.x[0],r6.x[0],res.x[0]);

                        r8.x[0] = _mm_permute_ps(r9.x[1],_MM_SHUFFLE(3,3,3,3));
                        res.x[0] = _mm_fmadd_ps(r8.x[0],r7.x[0],res.x[0]);

                        ak += wa[1] * 8;
                        bk += 8;
                    }
                }

                if( k_rem ){
                    for( auto k = 0ul; k < k_rem; ++k ){
                        r0.x[0] = _mm_maskload_ps(ak,mask);
                        r8.x[0] = _mm_broadcast_ss(bk);
                        res.x[0] = _mm_fmadd_ps(r0.x[0],r8.x[0],res.x[0]);
                        
                        ak += wa[1];
                        ++bk;
                    }
                }
                _mm_maskstore_ps(ck,mask,res.x[0]);

            }
        }
        
        template<typename SizeType>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void kernel_helper_4xn(
            float* __restrict__ c, SizeType const* __restrict__ nc, SizeType const* __restrict__ wc,
            float const* __restrict__ a, SizeType const* __restrict__ na, SizeType const* __restrict__ wa,
            float const* __restrict__ b, SizeType const* __restrict__ nb, SizeType const* __restrict__ wb
        ) const noexcept{

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

            FReg r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, res;
            auto k_iter = na[1] / 8;
            auto k_rem = na[1] % 8;

            _mm_prefetch(c,_MM_HINT_T0);
            
            auto ak = a;
            auto bk = b;
            auto ck = c;

            res.x[0] = _mm_load_ps(ck);
            if( k_iter ){
                for( auto k = k_iter; k > 0 ; --k ){
                    
                    r0.x[0] = _mm_load_ps(ak + wa[1] * 0);
                    r1.x[0] = _mm_load_ps(ak + wa[1] * 1);
                    r2.x[0] = _mm_load_ps(ak + wa[1] * 2);
                    r3.x[0] = _mm_load_ps(ak + wa[1] * 3);
                    r4.x[0] = _mm_load_ps(ak + wa[1] * 4);
                    r5.x[0] = _mm_load_ps(ak + wa[1] * 5);
                    r6.x[0] = _mm_load_ps(ak + wa[1] * 6);
                    r7.x[0] = _mm_load_ps(ak + wa[1] * 7);

                    r9.y = _mm256_load_ps(bk);

                    r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(0,0,0,0));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r0.x[0],res.x[0]);

                    r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(1,1,1,1));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r1.x[0],res.x[0]);

                    r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(2,2,2,2));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r2.x[0],res.x[0]);

                    r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(3,3,3,3));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r3.x[0],res.x[0]);

                    r8.x[0] = _mm_permute_ps(r9.x[1],_MM_SHUFFLE(0,0,0,0));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r4.x[0],res.x[0]);

                    r8.x[0] = _mm_permute_ps(r9.x[1],_MM_SHUFFLE(1,1,1,1));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r5.x[0],res.x[0]);

                    r8.x[0] = _mm_permute_ps(r9.x[1],_MM_SHUFFLE(2,2,2,2));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r6.x[0],res.x[0]);

                    r8.x[0] = _mm_permute_ps(r9.x[1],_MM_SHUFFLE(3,3,3,3));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r7.x[0],res.x[0]);

                    ak += wa[1] * 8;
                    bk += 8;
                }
            }
            if( k_rem ){
                k_iter = k_rem / 4;
                k_rem = k_rem % 4;
                if( k_iter ){
                    r0.x[0] = _mm_load_ps(ak + wa[1] * 0);
                    r1.x[0] = _mm_load_ps(ak + wa[1] * 1);
                    r2.x[0] = _mm_load_ps(ak + wa[1] * 2);
                    r3.x[0] = _mm_load_ps(ak + wa[1] * 3);

                    r9.x[0] = _mm_load_ps(bk);

                    r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(0,0,0,0));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r0.x[0],res.x[0]);

                    r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(1,1,1,1));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r1.x[0],res.x[0]);

                    r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(2,2,2,2));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r2.x[0],res.x[0]);

                    r8.x[0] = _mm_permute_ps(r9.x[0],_MM_SHUFFLE(3,3,3,3));
                    res.x[0] = _mm_fmadd_ps(r8.x[0],r3.x[0],res.x[0]);

                    ak += wa[1] * 4;
                    bk += 4;
                }

                if( k_rem ){
                    for(auto k = k_rem; k > 0 ; --k){
                        r0.x[0] = _mm_load_ps(ak);
                        r8.x[0] = _mm_broadcast_ss(bk);
                        res.x[0] = _mm_fmadd_ps(r0.x[0],r8.x[0],res.x[0]);
                        ak += wa[1];
                        ++bk;
                    }
                }

            }

            _mm_store_ps(ck,res.x[0]);

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
                jne LKERNEL8x8LOOPREM%=
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
                
                vmovaps %%ymm15, (%%rcx)
            LKERNEL8x8LOOPREM%=:
                testq $4,%%r13
                jne LKERNEL8x8LOOPREM_MUL4%=
                
                testq $0,%%r13
                jne LKERNEL8x8LOOPZERO%=

            LKERNEL8x8LOOPREM_MUL%=:
                leaq 32(%%rcx), (%%rcx)

                

                decq %%r13
                jne LKERNEL8x8LOOPREM_MUL%=

                jmp LKERNEL8x8LOOPZERO%=

            LKERNEL8x8LOOPREM_MUL4%=:
                leaq 32(%%rcx), (%%rcx)

                movaps (%rcx), %%xmm15

                vbroadcastss %%xmm0, (%%rbx)
                vfmadd231ps (%%rax), %%xmm0, %%xmm15
                
                vbroadcastss %%xmm0, 4(%%rbx)
                vfmadd231ps (%%rax, %%r9,4), %%xmm0, %%xmm15
                
                vbroadcastss %%xmm0, 8(%%rbx)
                vfmadd231ps (%%rax, %%r9,8), %%xmm0, %%xmm15

                leaq (%%rax, %%r9,8), (%%rax)
                
                vbroadcastss %%xmm0, 12(%%rbx)
                vfmadd231ps (%%rax, %%r9,4), %%xmm0, %%xmm15

                movaps %%xmm15, (%%rcx)

            LKERNEL8x8LOOPZERO%=:

            )"
                : [c] "=m" (c)
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
        float* __restrict__ c, SizeType const* __restrict__ nc, SizeType const* __restrict__ wc,
        float const* __restrict__ a, SizeType const* __restrict__ na, SizeType const* __restrict__ wa,
        float const* __restrict__ b, SizeType const* __restrict__ nb, SizeType const* __restrict__ wb,
        KernelType ker = {}, PartitionType par = {}
    ) noexcept{

        auto ai = a;
        auto bi = b;
        auto ci = c;
        
        // SizeType wta[2] = {wa[0],wa[1]};
        // SizeType wtb[2] = {wb[0],wb[1]};
        // SizeType wtc[2] = {wc[0],wc[1]};

        // if( wa[1] == 1ul ){
        //     std::swap(wta[0],wta[1]);
        // }

        // if( wb[1] == 1ul ){
        //     std::swap(wtb[0],wtb[1]);
        // }

        // if( wc[1] == 1ul ){
        //     std::swap(wtc[0],wtc[1]);
        // }
        
        auto const K = par.k(na[1]);
        auto const M = par.m(na[0]);

        auto wa_1 = wa[1] * K;
        auto wa_0 = wa[0] * M;
        auto wb_0 = wb[0] * K;
        auto wc_0 = wc[0] * M;


        #pragma omp parallel for schedule(dynamic)
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
