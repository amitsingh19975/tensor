//
//  Copyright (c) 2018-2019, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef _BOOST_UBLAS_TENSOR_DETAIL_CONFIG_HPP_
#define _BOOST_UBLAS_TENSOR_DETAIL_CONFIG_HPP_

#include <boost/config.hpp>
#include <cassert>

#if defined(__GNUC__) || ( defined( __has_attribute ) && __has_attribute(always_inline) )
    #define BOOST_UBLAS_TENSOR_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER) && !__INTEL_COMPILER && _MSC_VER >= 1310
    #define BOOST_UBLAS_TENSOR_ALWAYS_INLINE inline __forceinline
#else
    #define BOOST_UBLAS_TENSOR_ALWAYS_INLINE inline
#endif

#ifdef BOOST_UBLAS_TENSOR_TESTING
    #define TENSOR_ASSERT(e,comment) assert( (e) && comment )
    inline static constexpr bool TENSOR_ASSERT_NOEXCEPT = true;
#else
    #define TENSOR_ASSERT(e,comment) if (!e) { throw std::runtime_error( comment ); }
    inline static constexpr bool TENSOR_ASSERT_NOEXCEPT = false;
#endif
#if defined(__GNUC__) || defined(__INTEL_COMPILER) ||  ( defined( __has_attribute ) && __has_attribute(__builtin_expect) )
    #define BOOST_UBLAS_TENSOR_LIKLY(x) __builtin_expect(!!(x),1)
    #define BOOST_UBLAS_TENSOR_UNLIKLY(x) __builtin_expect(!!(x),0)
#else
    #define BOOST_UBLAS_TENSOR_LIKLY(x) x
    #define BOOST_UBLAS_TENSOR_UNLIKLY(x) BOOST_UBLAS_TENSOR_LIKLY(x)
#endif

#ifndef BOOST_UBLAS_INLINE
    #define BOOST_UBLAS_TENSOR_INLINE BOOST_UBLAS_TENSOR_ALWAYS_INLINE
#else
    #define BOOST_UBLAS_TENSOR_INLINE BOOST_UBLAS_INLINE
#endif


#endif // _BOOST_UBLAS_TENSOR_DETAIL_CONFIG_HPP_
