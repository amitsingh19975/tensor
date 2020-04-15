#ifndef _BOOST_UBLAS_TENSOR_DETAIL_SIMD_ADDITION_IMPL_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_SIMD_ADDITION_IMPL_HPP

#include "simd_assignment_impl.hpp"

namespace boost::numeric::ublas::simd::detail{

    template<typename T>
    struct addition{
        static_assert(std::numeric_limits<T>::is_signed, "boost::numeric::ublas::simd::detail::assignment: tensor supports only signed value currently");

        template<typename U = T, typename = std::enable_if_t< std::numeric_limits<U>::is_integer , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<64,U> const& a, simd_type_t<64,U> const& b) const noexcept{
            if constexpr( std::is_same_v<T, int8_t> || std::is_same_v<T, char> ){
                return _mm_add_pi8(a,b);
            }else if constexpr(std::is_same_v<T, int16_t> || std::is_same_v<T, short>){
                return _mm_add_pi16(a,b);
            }else if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int>){
                return _mm_add_pi32(a,b);
            }
        }
        
        template<typename U = T, typename = std::enable_if_t< std::numeric_limits<U>::is_integer , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<128,U> const& a, simd_type_t<128,U> const& b) const noexcept{
            if constexpr( std::is_same_v<T, int8_t> || std::is_same_v<T, char> ){
                return _mm_add_epi8(a,b);
            }else if constexpr(std::is_same_v<T, int16_t> || std::is_same_v<T, short>){
                return _mm_add_epi16(a,b);
            }else if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int>){
                return _mm_add_epi32(a,b);
            }else if constexpr(std::is_same_v<T, int64_t> || std::is_same_v<T, long long>){
                return _mm_add_epi64(a,b);
            }
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,float> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<128,float> const& a, simd_type_t<128,float> const& b) const noexcept{
            static_assert(std::is_same_v<T,float>, "boost::numeric::ublas::simd::detail::addition::operator()(simd_type_t<128,float>,simd_type_t<128,float>) : invalid type, expected float");;
            return _mm_add_ps(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,float> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<256,float> const& a, simd_type_t<256,float> const& b) const noexcept{
            static_assert(std::is_same_v<T,float>, "boost::numeric::ublas::simd::detail::addition::operator()(simd_type_t<256,float>,simd_type_t<256,float>) : invalid type, expected float");
            return _mm256_add_ps(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,float> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<512,float> const& a, simd_type_t<512,float> const& b) const noexcept{
            static_assert(std::is_same_v<T,float>, "boost::numeric::ublas::simd::detail::addition::operator()(simd_type_t<512,float>,simd_type_t<512,float>) : invalid type, expected float");
            return _mm512_add_ps(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,double> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<128,double> const& a, simd_type_t<128,double> const& b) const noexcept{
            static_assert(std::is_same_v<T,double>, "boost::numeric::ublas::simd::detail::addition::operator()(simd_type_t<128,double>,simd_type_t<128,double>) : invalid type, expected double");
            return _mm_add_pd(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,double> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<256,double> const& a, simd_type_t<256,double> const& b) const noexcept{
            static_assert(std::is_same_v<T,double>, "boost::numeric::ublas::simd::detail::addition::operator()(simd_type_t<256,double>,simd_type_t<256,double>) : invalid type, expected double");
            return _mm256_add_pd(a,b);
        }
        
        template<typename U = T, typename = std::enable_if_t< std::is_same_v<U,double> , U> >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(simd_type_t<512,double> const& a, simd_type_t<512,double> const& b) const noexcept{
            static_assert(std::is_same_v<T,double>, "boost::numeric::ublas::simd::detail::addition::operator()(simd_type_t<512,double>,simd_type_t<512,double>) : invalid type, expected double");
            return _mm512_add_pd(a,b);
        }

    };

} // namespace boost::numeric::ublas::detail

#endif 
