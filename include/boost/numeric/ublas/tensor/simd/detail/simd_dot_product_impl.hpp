#ifndef _BOOST_UBLAS_TENSOR_DETAIL_dot_product_IMPL_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_dot_product_IMPL_HPP

#include "simd_horizontal_addition_impl.hpp"
#include "simd_store_impl.hpp"

namespace boost::numeric::ublas::simd::detail{

    template<size_t N, typename T>
    struct dot_product{
        using type = typename simd_type<N,T>::type;
        constexpr static auto const bsz = block_size_v<N,T>;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()( type const& a, type const& b ){
            T res[bsz];
            auto p = multiplies<T>{}(a,b);
            p = horizontal_addition<T>{}(p,p);
            p = horizontal_addition<T>{}(p,p);
            store<N,T>{}(res,p);
            return bsz <= 4 ? res[0] : res[bsz / 2 - 1] + res[bsz / 2 + 1];
        }
    };

    template<>
    struct dot_product<128,float>{
        constexpr static auto const bsz = block_size_v<128,float>;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()( __m128 const& a, __m128 const& b ){
            float res[bsz];
            auto r = _mm_dp_ps( a, b ,0xff );
            store<128,float>{}(res,r);
            return res[0];
        }
    };

    template<>
    struct dot_product<128,double>{
        constexpr static auto const bsz = block_size_v<128,double>;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()( __m128d const& a, __m128d const& b ){
            double res[bsz];
            auto r = _mm_dp_pd( a, b ,0xff );
            store<128,double>{}(res,r);
            return res[0];
        }
    };

    template<>
    struct dot_product<256,float>{
        constexpr static auto const bsz = block_size_v<256,float>;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()( __m256 const& a, __m256 const& b ){
            float res[bsz];
            auto r = _mm256_dp_ps( a, b ,0xff );
            store<256,float>{}(res,r);
            return res[bsz / 2 - 1] + res[bsz / 2 + 1];
        }
    };


} // namespace boost::numeric::ublas::detail

#endif 
