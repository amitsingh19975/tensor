#ifndef _BOOST_UBLAS_TENSOR_DETAIL_SIMD_MACRO_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_SIMD_MACRO_HPP

#include <immintrin.h>

#ifndef BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    #if __GNUC__ || __has_attribute(always_inline)
        #define BOOST_UBLAS_TENSOR_ALWAYS_INLINE inline __attribute__((always_inline))
    #elif defined(_MSC_VER) && !__INTEL_COMPILER && _MSC_VER >= 1310
        #define BOOST_UBLAS_TENSOR_ALWAYS_INLINE inline __forceinline
    #else
        #define BOOST_UBLAS_TENSOR_ALWAYS_INLINE inline
    #endif
#endif

#endif // _BOOST_UBLAS_TENSOR_DETAIL_SIMD_MACRO_HPP