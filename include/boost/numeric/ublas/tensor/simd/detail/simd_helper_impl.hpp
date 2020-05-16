#ifndef _BOOST_UBLAS_TENSOR_DETAIL_SIMD_HELPER_IMPL_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_SIMD_HELPER_IMPL_HPP

#include "simd_macro.hpp"
#include <cstddef>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <functional>

namespace boost::numeric::ublas::simd::detail{
    
    template< size_t N, typename T > struct is_valid_vector_size;
    template< size_t N, typename T > struct block_size;
    template< size_t N, typename T > struct simd_type;
    template< size_t N, typename T > struct assignment;
    template< size_t N, typename T > struct dot_product;
    template< size_t N, typename T > struct load;
    template< size_t N, typename T > struct store;
    template< typename T > struct multiplies;
    template< typename T > struct addition;
    template< typename T > struct subtraction;
    template< typename T > struct divides;
    template< typename T > struct horizontal_addition;

} // namespace boost::numeric::ublas::detail


namespace boost::numeric::ublas::simd::detail{

    template<size_t N, typename T> 
    struct is_valid_vector_size{
        constexpr static auto const value = ( N == 64 ) || ( N == 128 ) || ( N == 256 ) || ( N == 512 );
    };
    
    template<size_t N> 
    struct is_valid_vector_size<N,float>{
        constexpr static auto const value = ( N == 128 ) || ( N == 256 ) || ( N == 512 );
    };
    
    template<size_t N> 
    struct is_valid_vector_size<N,double>{
        constexpr static auto const value = ( N == 128 ) || ( N == 256 ) || ( N == 512 );
    };

    template<size_t N, typename T>
    struct block_size{
        constexpr static auto const value = N / ( 8 * sizeof(T) );
    };

    template<typename T>
    struct simd_type<64, T>{
        using type = __m64;
    };

    template<typename T>
    struct simd_type<128, T>{
        using type = __m128i;
    };

    template<typename T>
    struct simd_type<256, T>{
        using type = __m256i;
    };

    template<typename T>
    struct simd_type<512, T>{
        using type = __m512i;
    };

    template<>
    struct simd_type<128, float>{
        using type = __m128;
    };

    template<>
    struct simd_type<256, float>{
        using type = __m256;
    };

    template<>
    struct simd_type<512, float>{
        using type = __m512;
    };

    template<>
    struct simd_type<128, double>{
        using type = __m128d;
    };

    template<>
    struct simd_type<256, double>{
        using type = __m256d;
    };

    template<>
    struct simd_type<512, double>{
        using type = __m512d;
    };
} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas::simd::detail{
    
    template< size_t N, typename T > 
    inline constexpr static auto const is_valid_vector_size_v = is_valid_vector_size<N,T>::value;
    template< size_t N, typename T > 
    inline constexpr static auto const block_size_v = block_size<N,T>::value;
    template< size_t N, typename T > 
    using simd_type_t = typename simd_type<N,T>::type;

} // namespace boost::numeric::ublas::detail

#include <sstream>
#include <iostream>

namespace boost::numeric::ublas::simd::tool{
    
    namespace detail{
        template <typename A, typename... R>
        void crash_impl (std::ostringstream &oss, A const& arg, R... rest){
            oss << arg;
            if constexpr( sizeof...(rest) == 0 ){
                oss << std::endl;
                std::cerr<<oss.str();
                exit(1);
            }else{
                crash_impl(oss, std::forward<R>(rest)...);
            }
        }
    } // namespace detail
    

    template <typename... Args>
    void crash(Args&&... args){
        std::ostringstream oss;
        detail::crash_impl(oss, std::forward<Args>(args)...);
    }

} // namespace boost::numeric::ublas::simd::tool


namespace boost::numeric::ublas::simd{
    template<int hint = _MM_HINT_T0>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void prefetch(float const* p, size_t offset) noexcept{
        _mm_prefetch(p, hint);
        p += offset;
        _mm_prefetch(p, hint);
        p += offset;
        _mm_prefetch(p, hint);
        p += offset;
        _mm_prefetch(p, hint);
        p += offset;
        _mm_prefetch(p, hint);
        p += offset;
        _mm_prefetch(p, hint);
        p += offset;
        _mm_prefetch(p, hint);
        p += offset;
        _mm_prefetch(p, hint);
    }

    template<int N, int hint = _MM_HINT_T0>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void prefetchN(float const* p, size_t offset) noexcept{
        auto pi = p;
        for( auto i = N - 1; i >= 0; --i, pi += offset ){
            auto pj = pi;
            for( auto j = N; j >= 0; j -= 16, pj += 16){
                _mm_prefetch(pj, hint);
            }
        }
    }
    
    template<int hint = _MM_HINT_T0>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void prefetch_4(float const* p, size_t offset) noexcept{
        _mm_prefetch(p, hint);
        p += offset;
        _mm_prefetch(p, hint);
        p += offset;
        _mm_prefetch(p, hint);
        p += offset;
        _mm_prefetch(p, hint);
    }

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void dot_prod(__m256& res, __m256 const& a, __m256 const& b) noexcept{
        // res = _mm256_mul_ps(a,b);
        // res = _mm256_hadd_ps(res,res);
        // res = _mm256_hadd_ps(res,res);
        res = _mm256_dp_ps(a,b,0xff);
    }

    union FReg{
        __m256 y;
        __m128 x[2];
    };

    struct VFloat{

        VFloat() 
        {
            reg.y = _mm256_setzero_ps();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void load128(float const* p, int idx, size_t w = 0) noexcept{
            reg.x[idx] = _mm_load_ps(p + w);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void loadu128(float const* p, int idx, size_t w = 0) noexcept{
            reg.x[idx] = _mm_loadu_ps(p + w);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void load256(float const* p, size_t w = 0) noexcept{
            reg.y = _mm256_load_ps(p + w);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void loadu256(float const* p, size_t w = 0) noexcept{
            reg.y = _mm256_loadu_ps(p + w);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void store128(float* p, int idx, size_t w = 0) const noexcept{
            _mm_store_ps(p + w, reg.x[idx]);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void storeu128(float* p, int idx, size_t w = 0) const noexcept{
            _mm_storeu_ps(p + w, reg.x[idx]);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void store256(float* p, size_t w = 0) const noexcept{
            _mm256_store_ps(p + w, reg.y);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void storeu256(float* p, size_t w = 0) const noexcept{
            _mm256_storeu_ps(p + w,reg.y);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void broadcast256(float const* p, size_t w = 0) noexcept{
            reg.y = _mm256_broadcast_ss(p + w);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void broadcast128(float const* p, int idx) noexcept{
            reg.x[idx] = _mm_broadcast_ss(p);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void fmadd256(VFloat const& p1, VFloat const& p2) noexcept{
            reg.y = _mm256_fmadd_ps(p1.reg.y, p2.reg.y, reg.y);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void fmadd128(VFloat const& p1, VFloat const& p2, int idx1, int idx2, int curr_idx) noexcept{
            reg.x[curr_idx] = _mm_fmadd_ps(p1.x(idx1), p2.x(idx2), x(curr_idx));
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        constexpr __m256 const& y() const noexcept{
            return reg.y;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        constexpr __m128 const& x(int idx) const noexcept{
            return reg.x[idx];
        }

        FReg reg;
    };

    template<typename T, size_t Alignment = 64>
    struct align_allocator{

        using size_type = size_t;
        using difference_type = ptrdiff_t;
        using pointer = T*;
        using const_pointer = T const*;
        using reference = T&;
        using const_reference = T const&;
        using value_type = T;

        constexpr align_allocator() = default;

        constexpr pointer address(reference x) const noexcept{
            return std::addressof(x);
        }

        constexpr const_pointer address(const_reference x) const noexcept{
            return std::addressof(x);
        }

        inline pointer allocate(size_type n, std::allocator<void>::const_pointer = 0){
            if( n > max_size() ){
                throw std::length_error("align_allocator: 'n' exceeds maximum supported size");
            }
            auto mem = _mm_malloc(n * sizeof(float),Alignment);
            return new(mem) T();
        }

        inline void deallocate(pointer p, size_type){
            _mm_free(p);
        }

        inline constexpr size_type max_size() const noexcept{
            return size_type(~0) / sizeof(T);
        }

    };
    
}

#endif 
