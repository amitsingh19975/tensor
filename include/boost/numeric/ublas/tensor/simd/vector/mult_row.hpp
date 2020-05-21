#if !defined(BOOST_TENSOR_VECTOR_MULT_ROW_HPP)
#define BOOST_TENSOR_VECTOR_MULT_ROW_HPP

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
    
    template<>
    struct kernel<last_order>{
        using order_type = last_order;
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

            auto i_iter = na[1] / M;
            auto i_rem = na[1] % M;

            if( i_iter ){
                for(auto i = i_iter; i > 0 ; --i ){

                    SizeType const nta[] = { M , na[0]};

                    SizeType const ntb[] = { 1, na[0]};

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
                assert(0);
                SizeType const nta[] = { i_rem, na[0]};

                SizeType const ntb[] = { na[0], 1};

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
            
            FReg r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, res, r10, r11, r12;
            auto k_iter = na[1] / 8;
            auto k_rem = na[1] % 8;
            
            _mm_prefetch(c,_MM_HINT_T0);
            
            auto ak = a;
            auto bk = b;
            auto ck = c;

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
            // for(auto i = 0; i < nb[0] * nb[1]; ++i){
            //         std::cout<<b[i]<<' ';
            // }
            // std::cout<<std::endl;

            res.y = _mm256_load_ps(ck);
            if( k_iter ){
                for( auto k = k_iter; k > 0 ; --k ){
                    r0.y = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ak + wa[0] * 0 + 0)), _mm_load_ps(ak + wa[0] * 4 + 0),1);
                    r1.y = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ak + wa[0] * 1 + 0)), _mm_load_ps(ak + wa[0] * 5 + 0),1);
                    r2.y = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ak + wa[0] * 2 + 0)), _mm_load_ps(ak + wa[0] * 6 + 0),1);
                    r3.y = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ak + wa[0] * 3 + 0)), _mm_load_ps(ak + wa[0] * 7 + 0),1);
                    r4.y = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ak + wa[0] * 0 + 4)), _mm_load_ps(ak + wa[0] * 4 + 4),1);
                    r5.y = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ak + wa[0] * 1 + 4)), _mm_load_ps(ak + wa[0] * 5 + 4),1);
                    r6.y = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ak + wa[0] * 2 + 4)), _mm_load_ps(ak + wa[0] * 6 + 4),1);
                    r7.y = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ak + wa[0] * 3 + 4)), _mm_load_ps(ak + wa[0] * 7 + 4),1);

                    r8.y = _mm256_unpacklo_ps(r0.y,r1.y);
                    r9.y = _mm256_unpackhi_ps(r0.y,r1.y);
                    r10.y = _mm256_unpacklo_ps(r2.y,r3.y);
                    r11.y = _mm256_unpackhi_ps(r2.y,r3.y);

                    r12.y = _mm256_shuffle_ps(r8.y,r10.y,0x4E);
                    r0.y = _mm256_blend_ps(r8.y,r12.y,0xCC);
                    r1.y = _mm256_blend_ps(r10.y,r12.y,0x33);

                    r12.y = _mm256_shuffle_ps(r9.y,r11.y,0x4E);
                    r2.y = _mm256_blend_ps(r9.y,r12.y,0xCC);
                    r3.y = _mm256_blend_ps(r11.y,r12.y,0x33);

                    r8.y = _mm256_unpacklo_ps(r4.y,r5.y);
                    r9.y = _mm256_unpackhi_ps(r4.y,r5.y);
                    r10.y = _mm256_unpacklo_ps(r6.y,r7.y);
                    r11.y = _mm256_unpackhi_ps(r6.y,r7.y);

                    r12.y = _mm256_shuffle_ps(r8.y,r10.y,0x4E);
                    r4.y = _mm256_blend_ps(r8.y,r12.y,0xCC);
                    r5.y = _mm256_blend_ps(r10.y,r12.y,0x33);

                    r12.y = _mm256_shuffle_ps(r9.y,r11.y,0x4E);
                    r6.y = _mm256_blend_ps(r9.y,r12.y,0xCC);
                    r7.y = _mm256_blend_ps(r11.y,r12.y,0x33);

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
                
                if(k_iter){
                    
                    #pragma omp simd
                    for( auto i = 0ul; i < 4; ++i ){
                        auto aj = ak;
                        for( auto j = 0ul; j < 8; ++j ){
                            *ck += *aj * *bk;
                            ++bk;
                            aj += wa[0];
                        }
                        ++ck;
                        ak += wa[1];
                    }

                }
                
                if(k_rem){
                    
                    for( auto i = 0ul; i < k_rem; ++i ){
                        auto aj = ak;
                        #pragma omp simd
                        for( auto j = 0ul; j < 8; ++j ){
                            *ck += *aj * *bk;
                            ++bk;
                            aj += wa[0];
                        }
                        ++ck;
                        ak += wa[1];
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
    };

} // namespace boost::numeric::ublas::simd


#endif // BOOST_TENSOR_VECTOR_MULT_ROW_HPP
