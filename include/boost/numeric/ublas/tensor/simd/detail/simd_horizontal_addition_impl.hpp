#ifndef _BOOST_UBLAS_TENSOR_DETAIL_SIMD_HORIZONTAL_ADDITION_IMPL_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_SIMD_HORIZONTAL_ADDITION_IMPL_HPP

#include "simd_assignment_impl.hpp"
#include <cassert>

namespace boost::numeric::ublas::simd::detail{

    template<typename T>
    struct horizontal_addition{
        template<typename U = T, typename = std::enable_if_t< std::numeric_limits<U>::is_integer , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()( simd_type_t<128,U> const& a, simd_type_t<128,U> const& b ) const noexcept{
            if constexpr( std::is_same_v<T,int16_t> || std::is_same_v<T,short> ){
                return _mm_hadd_epi16(a,b);
            }else if constexpr(std::is_same_v<T,int32_t> || std::is_same_v<T,int>){
                return _mm_hadd_epi32(a,b);
            }else if constexpr( std::is_same_v<T,int64_t> || std::is_same_v<T,long long>){
                auto na = _mm_cvtepi64_pd( a );
                auto nb = _mm_cvtepi64_pd( b );
                auto res = _mm_hadd_pd(na,nb);
                return assignment<128,int64_t>{}(res[0],res[1]);
            }else if constexpr( std::is_same_v<T, char> || std::is_same_v<T,int8_t>){
                // TODO: come up with alternative
                boost::numeric::ublas::simd::tool::crash("Not Implemented Yet");
            }
        }

        template<typename U = T, typename = std::enable_if_t< std::numeric_limits<U>::is_integer , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()( simd_type_t<256,U> const& a, simd_type_t<256,U> const& b ) const noexcept{
            if constexpr( std::is_same_v<T,int16_t> || std::is_same_v<T,short> ){
                return _mm256_hadd_epi16(a,b);
            }else if constexpr(std::is_same_v<T,int32_t> || std::is_same_v<T,int>){
                return _mm256_hadd_epi32(a,b);
            }else if constexpr( std::is_same_v<T,int64_t> || std::is_same_v<T,long long>){
                auto na = _mm256_cvtepi64_pd( a );
                auto nb = _mm256_cvtepi64_pd( b );
                auto res = _mm256_hadd_pd(na,nb);
                return assignment<256,int64_t>{}(res[0],res[1], res[2], res[3]);
            }else if constexpr( std::is_same_v<T, char> || std::is_same_v<T,int8_t>){
                // TODO: come up with alternative
                boost::numeric::ublas::simd::tool::crash("Not Implemented Yet");
            }
        }

        template<typename U = T, typename = std::enable_if_t< std::numeric_limits<U>::is_integer , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()( simd_type_t<512,U> const& a, simd_type_t<512,U> const& b ) const noexcept{
            // TODO: write the simd for avx512
        }

        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,float> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<128,float> const& a, simd_type_t<128,float> const& b) const noexcept{
            static_assert(std::is_same_v<T,float>, "simd_hadd_operation::operator()(simd_type_t<128,float>,simd_type_t<128,float>) : invalid type, expected float");
            return _mm_hadd_ps(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,float> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<256,float> const& a, simd_type_t<256,float> const& b) const noexcept{
            static_assert(std::is_same_v<T,float>, "simd_hadd_operation::operator()(simd_type_t<256,float>,simd_type_t<256,float>) : invalid type, expected float");
            return _mm256_hadd_ps(a,b);
        }

        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,float> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<512,float> const& a, simd_type_t<512,float> const& b) const noexcept{
            static_assert(std::is_same_v<T,float>, "simd_hadd_operation::operator()(simd_type_t<512,float>,simd_type_t<512,float>) : invalid type, expected float");
            // TODO: adding support for avx512
            boost::numeric::ublas::simd::tool::crash("Not Implemented Yet");
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,double> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<128,double> const& a, simd_type_t<128,double> const& b) const noexcept{
            static_assert(std::is_same_v<T,double>, "simd_hadd_operation::operator()(simd_type_t<128,double>,simd_type_t<128,double>) : invalid type, expected double");
            return _mm_hadd_pd(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,double> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<256,double> const& a, simd_type_t<256,double> const& b) const noexcept{
            static_assert(std::is_same_v<T,double>, "simd_hadd_operation::operator()(simd_type_t<256,double>,simd_type_t<256,double>) : invalid type, expected double");
            return _mm256_hadd_pd(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,double> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<512,double> const& a, simd_type_t<512,double> const& b) const noexcept{
            static_assert(std::is_same_v<T,double>, "simd_hadd_operation::operator()(simd_type_t<512,double>,simd_type_t<512,double>) : invalid type, expected double");
            // TODO: adding support for avx512
            boost::numeric::ublas::simd::tool::crash("Not Implemented Yet");
        }

    };

} // namespace boost::numeric::ublas::detail

#endif 
