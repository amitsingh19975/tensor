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

#if defined( __has_attribute )
    #define has_attribute(ATTR) __has_attribute(ATTR)
#else
    #define has_attribute(ATTR) 0
#endif

#ifndef BOOST_UBLAS_TENSOR_TEST_ENABLE
    #define TENSOR_ASSERT(e,comment) assert( (e) && comment )
    inline static constexpr bool TENSOR_ASSERT_NOEXCEPT = true;
#else
    #define TENSOR_ASSERT(e,comment) if (!e) { throw std::runtime_error( comment ); }
    inline static constexpr bool TENSOR_ASSERT_NOEXCEPT = false;
#endif

#if defined(__GNUC__) || defined(__INTEL_COMPILER) ||  has_attribute(__builtin_expect)
    #define BOOST_UBLAS_TENSOR_LIKLY(x) static_cast<bool>(__builtin_expect(!!(x),1))
    #define BOOST_UBLAS_TENSOR_UNLIKLY(x) static_cast<bool>(__builtin_expect(!!(x),0))
#else
    #define BOOST_UBLAS_TENSOR_LIKLY(x) static_cast<bool>(x)
    #define BOOST_UBLAS_TENSOR_UNLIKLY(x) BOOST_UBLAS_TENSOR_LIKLY(x)
#endif

#ifndef BOOST_UBLAS_INLINE
    #define BOOST_UBLAS_TENSOR_INLINE inline
#else
    #define BOOST_UBLAS_TENSOR_INLINE BOOST_UBLAS_INLINE
#endif


#endif // _BOOST_UBLAS_TENSOR_DETAIL_CONFIG_HPP_
