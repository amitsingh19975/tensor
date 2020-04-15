#ifndef _BOOST_UBLAS_TENSOR_DETAIL_SIMD_DIVISION_IMPL_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_SIMD_DIVISION_IMPL_HPP

#include "simd_multiplication_impl.hpp"

namespace boost::numeric::ublas::simd::detail{

    template< typename T >
    struct divides{
        template<typename U = T, typename = std::enable_if_t< std::numeric_limits<U>::is_integer , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<128,U> const& a, simd_type_t<128,U> const& b) const noexcept{
            if constexpr(std::is_same_v<T,int64_t> || std::is_same_v<T,long long>){
                auto na = cast_to_simd_double<128>(a);
                auto nb = cast_to_simd_double<128>(b);
                auto ans = _mm_div_pd(na,nb);
                return cast_to_simd_int64<128>(ans); 
            }else if constexpr(std::is_same_v<T,int32_t> || std::is_same_v<T,int>){
                auto na = _mm_cvtepi32_ps(a);
                auto nb = _mm_cvtepi32_ps(b);
                auto ans = _mm_div_ps(na,nb);
                return _mm_cvtps_epi32(ans);
            }else if constexpr(std::is_same_v<T,int16_t> || std::is_same_v<T,short>){
                auto na = _mm_load_si128(&a);
                auto nb = _mm_load_si128(&b);
                
                auto al = _mm_cvtepi32_ps( _mm_cvtepi16_epi32(na) );
                auto bl = _mm_cvtepi32_ps( _mm_cvtepi16_epi32(nb) );
                
                na = _mm_srli_si128(na,2);
                nb = _mm_srli_si128(nb,2);
                auto ah = _mm_cvtepi32_ps( _mm_cvtepi16_epi32(na) );
                auto bh = _mm_cvtepi32_ps( _mm_cvtepi16_epi32(nb) );
                
                auto rl = _mm_cvtps_epi32( _mm_div_ps(al,bl) );
                auto rh = _mm_cvtps_epi32( _mm_div_ps(ah,bh) );
                auto* resl = reinterpret_cast<int32_t*>( &rl );
                auto* resh = reinterpret_cast<int32_t*> ( &rh );
                return _mm_setr_epi16(resl[0], resl[1], resl[2], resl[3], resh[0], resh[1], resh[2], resh[3]);
            }else if constexpr(std::is_same_v<T,int8_t> || std::is_same_v<T,char>){
                auto na = _mm_load_si128(&a);
                auto nb = _mm_load_si128(&b);
                
                auto a0 = _mm_cvtepi32_ps( _mm_cvtepi8_epi32(na) );
                auto b0 = _mm_cvtepi32_ps( _mm_cvtepi8_epi32(nb) );
                
                na = _mm_srli_si128(na,1);
                nb = _mm_srli_si128(nb,1);
                auto a1 = _mm_cvtepi32_ps( _mm_cvtepi8_epi32(na) );
                auto b1 = _mm_cvtepi32_ps( _mm_cvtepi8_epi32(nb) );
                
                na = _mm_srli_si128(na,1);
                nb = _mm_srli_si128(nb,1);
                auto a2 = _mm_cvtepi32_ps( _mm_cvtepi8_epi32(na) );
                auto b2 = _mm_cvtepi32_ps( _mm_cvtepi8_epi32(nb) );
                
                na = _mm_srli_si128(na,1);
                nb = _mm_srli_si128(nb,1);
                auto a3 = _mm_cvtepi32_ps( _mm_cvtepi8_epi32(na) );
                auto b3 = _mm_cvtepi32_ps( _mm_cvtepi8_epi32(nb) );
                
                auto r0 = _mm_cvtps_epi32( _mm_div_ps(a0,b0) );
                auto r1 = _mm_cvtps_epi32( _mm_div_ps(a1,b1) );
                auto r2 = _mm_cvtps_epi32( _mm_div_ps(a2,b2) );
                auto r3 = _mm_cvtps_epi32( _mm_div_ps(a3,b3) );

                auto* res0 = reinterpret_cast<int32_t*>( &r0 );
                auto* res1 = reinterpret_cast<int32_t*> ( &r1 );
                auto* res2 = reinterpret_cast<int32_t*>( &r2 );
                auto* res3 = reinterpret_cast<int32_t*> ( &r3 );

                return _mm_setr_epi8(
                    res0[0], res0[1], res0[2], res0[3], 
                    res1[0], res1[1], res1[2], res1[3], 
                    res2[0], res2[1], res2[2], res2[3],
                    res3[0], res3[1], res3[2], res3[3]);
            }else{
                //fallback
                auto* na = reinterpret_cast<T const*>(&a);
                auto* nb = reinterpret_cast<T const*>(&b);
                auto bsz = block_size_v<128,T>;
                float res[bsz];
                std::transform(na, na + bsz, nb, res, std::divides{} );
                return *reinterpret_cast<simd_type_t<128,T>*>(res);
            }
        }
        
        template<typename U = T, typename = std::enable_if_t< std::numeric_limits<U>::is_integer , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE simd_type_t<256,T>  operator()(simd_type_t<256,U> const& a, simd_type_t<256,U> const& b) const noexcept{
            if constexpr(std::is_same_v<T,int64_t> || std::is_same_v<T,long long>){
                auto na = cast_to_simd_double<256>(a);
                auto nb = cast_to_simd_double<256>(b);
                auto ans = _mm256_div_pd(na,nb);
                return cast_to_simd_int64<256>(ans); 
            }else if constexpr(std::is_same_v<T,int32_t> || std::is_same_v<T,int>){
                auto na = _mm256_cvtepi32_ps(a);
                auto nb = _mm256_cvtepi32_ps(b);
                auto ans = _mm256_div_ps(na,nb);
                return _mm256_cvtps_epi32(ans);
            }else if constexpr(std::is_same_v<T,int16_t> || std::is_same_v<T,short>){
                auto na = _mm256_load_si256(&a);
                auto nb = _mm256_load_si256(&b);

                auto na_low = _mm256_extracti128_si256(na, 0);
                auto nb_low = _mm256_extracti128_si256(nb, 0);
                auto na_high = _mm256_extracti128_si256(na, 1);
                auto nb_high = _mm256_extracti128_si256(nb, 1);

                auto a0 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_low) );
                auto b0 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_low) );

                na_low = _mm_srli_si128(na_low,2);
                nb_low = _mm_srli_si128(nb_low,2);
                auto a1 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_low) );
                auto b1 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_low) );
                

                auto a2 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_high) );
                auto b2 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_high) );
                
                na_high = _mm_srli_si128(na_high,2);
                nb_high = _mm_srli_si128(nb_high,2);
                auto a3 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_high) );
                auto b3 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_high) );
                
                auto r0 = _mm256_cvtps_epi32( _mm256_div_ps(a0,b0) );
                auto r1 = _mm256_cvtps_epi32( _mm256_div_ps(a1,b1) );
                auto r2 = _mm256_cvtps_epi32( _mm256_div_ps(a2,b2) );
                auto r3 = _mm256_cvtps_epi32( _mm256_div_ps(a3,b3) );

                auto* res0 = reinterpret_cast<int32_t*>( &r0 );
                auto* res1 = reinterpret_cast<int32_t*> ( &r1 );
                auto* res2 = reinterpret_cast<int32_t*>( &r2 );
                auto* res3 = reinterpret_cast<int32_t*> ( &r3 );
                
                return _mm256_setr_epi16(
                    res0[0], res0[1], res0[2], res0[3], 
                    res1[0], res1[1], res1[2], res1[3], 
                    res2[0], res2[1], res2[2], res2[3],
                    res3[0], res3[1], res3[2], res3[3]);
            }else if constexpr(std::is_same_v<T,int8_t> || std::is_same_v<T,char>){
                auto na = _mm256_load_si256(&a);
                auto nb = _mm256_load_si256(&b);
                
                auto na_low = _mm256_extracti128_si256(na, 0);
                auto nb_low = _mm256_extracti128_si256(nb, 0);
                auto na_high = _mm256_extracti128_si256(na, 1);
                auto nb_high = _mm256_extracti128_si256(nb, 1);

                auto a0 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_low) );
                auto b0 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_low) );
                
                na_low = _mm_srli_si128(na_low,1);
                nb_low = _mm_srli_si128(nb_low,1);
                auto a1 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_low) );
                auto b1 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_low) );
                
                na_low = _mm_srli_si128(na_low,1);
                nb_low = _mm_srli_si128(nb_low,1);
                auto a2 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_low) );
                auto b2 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_low) );
                
                na_low = _mm_srli_si128(na_low,1);
                nb_low = _mm_srli_si128(nb_low,1);
                auto a3 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_low) );
                auto b3 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_low) );


                auto a4 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_high) );
                auto b4 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_high) );
                
                na_high = _mm_srli_si128(na_high,1);
                nb_high = _mm_srli_si128(nb_high,1);
                auto a5 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_high) );
                auto b5 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_high) );
                
                na_high = _mm_srli_si128(na_high,1);
                nb_high = _mm_srli_si128(nb_high,1);
                auto a6 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_high) );
                auto b6 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_high) );
                
                na_high = _mm_srli_si128(na_high,1);
                nb_high = _mm_srli_si128(nb_high,1);
                auto a7 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(na_high) );
                auto b7 = _mm256_cvtepi32_ps( _mm256_cvtepi8_epi32(nb_high) );
                
                auto r0 = _mm256_cvtps_epi32( _mm256_div_ps(a0,b0) );
                auto r1 = _mm256_cvtps_epi32( _mm256_div_ps(a1,b1) );
                auto r2 = _mm256_cvtps_epi32( _mm256_div_ps(a2,b2) );
                auto r3 = _mm256_cvtps_epi32( _mm256_div_ps(a3,b3) );
                auto r4 = _mm256_cvtps_epi32( _mm256_div_ps(a4,b4) );
                auto r5 = _mm256_cvtps_epi32( _mm256_div_ps(a5,b5) );
                auto r6 = _mm256_cvtps_epi32( _mm256_div_ps(a6,b6) );
                auto r7 = _mm256_cvtps_epi32( _mm256_div_ps(a7,b7) );

                auto* res0 = reinterpret_cast<int32_t*> ( &r0 );
                auto* res1 = reinterpret_cast<int32_t*> ( &r1 );
                auto* res2 = reinterpret_cast<int32_t*> ( &r2 );
                auto* res3 = reinterpret_cast<int32_t*> ( &r3 );
                auto* res4 = reinterpret_cast<int32_t*> ( &r4 );
                auto* res5 = reinterpret_cast<int32_t*> ( &r5 );
                auto* res6 = reinterpret_cast<int32_t*> ( &r6 );
                auto* res7 = reinterpret_cast<int32_t*> ( &r7 );
                
                return _mm256_setr_epi8(
                    res0[0], res0[1], res0[2], res0[3], 
                    res1[0], res1[1], res1[2], res1[3], 
                    res2[0], res2[1], res2[2], res2[3],
                    res3[0], res3[1], res3[2], res3[3],
                    res4[0], res4[1], res4[2], res4[3], 
                    res5[0], res5[1], res5[2], res5[3], 
                    res6[0], res6[1], res6[2], res6[3],
                    res7[0], res7[1], res7[2], res7[3]
                    );
            }else{
                //fallback
                auto* na = reinterpret_cast<T const*>(&a);
                auto* nb = reinterpret_cast<T const*>(&b);
                auto bsz = block_size_v<256,T>;
                float res[bsz];
                std::transform(na, na + bsz, nb, res, std::divides{} );
                return *reinterpret_cast<simd_type_t<256,T>*>(res);
            }
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,float> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto  operator()(simd_type_t<128,float> const& a, simd_type_t<128,float> const& b) const noexcept{
            static_assert(std::is_same_v<T,float>, "boost::numeric::ublas::simd::detail::divides::operator()(simd_type_t<128,float>,simd_type_t<128,float>) : invalid type, expected float");
            return _mm_div_ps(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,float> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto  operator()(simd_type_t<256,float> const& a, simd_type_t<256,float> const& b) const noexcept{
            static_assert(std::is_same_v<T,float>, "boost::numeric::ublas::simd::detail::divides::operator()(simd_type_t<256,float>,simd_type_t<256,float>) : invalid type, expected float");
            return _mm256_div_ps(a,b);
        }

        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,float> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto  operator()(simd_type_t<512,float> const& a, simd_type_t<512,float> const& b) const noexcept{
            static_assert(std::is_same_v<T,float>, "boost::numeric::ublas::simd::detail::divides::operator()(simd_type_t<512,float>,simd_type_t<512,float>) : invalid type, expected float");
            return _mm512_div_ps(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,double> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto  operator()(simd_type_t<128,double> const& a, simd_type_t<128,double> const& b) const noexcept{
            static_assert(std::is_same_v<T,double>, "boost::numeric::ublas::simd::detail::divides::operator()(simd_type_t<128,double>,simd_type_t<128,double>) : invalid type, expected double");
            return _mm_div_pd(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,double> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto  operator()(simd_type_t<256,double> const& a, simd_type_t<256,double> const& b) const noexcept{
            static_assert(std::is_same_v<T,double>, "boost::numeric::ublas::simd::detail::divides::operator()(simd_type_t<256,double>,simd_type_t<256,double>) : invalid type, expected double");
            return _mm256_div_pd(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,double> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto  operator()(simd_type_t<512,double> const& a, simd_type_t<512,double> const& b) const noexcept{
            static_assert(std::is_same_v<T,double>, "boost::numeric::ublas::simd::detail::divides::operator()(simd_type_t<512,double>,simd_type_t<512,double>) : invalid type, expected double");
            return _mm512_div_pd(a,b);
        }

    }; 

} // namespace boost::numeric::ublas::detail

#endif 
