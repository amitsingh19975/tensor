#ifndef _BOOST_UBLAS_TENSOR_DETAIL_PARALLEL_MACRO_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_PARALLEL_MACRO_HPP

#include <cassert>
#include "type_def.hpp"

// #define BOOST_UBLAS_TENSOR_ALWAYS_INLINE

#ifndef BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    #if __GNUC__ || __has_attribute(always_inline)
        #define BOOST_UBLAS_TENSOR_ALWAYS_INLINE inline __attribute__((always_inline))
    #elif defined(_MSC_VER) && !__INTEL_COMPILER && _MSC_VER >= 1310
        #define BOOST_UBLAS_TENSOR_ALWAYS_INLINE inline __forceinline
    #else
        #define BOOST_UBLAS_TENSOR_ALWAYS_INLINE inline
    #endif
#endif

#define TENSOR_ASSERT(e,comment) assert( (e) && comment )

#ifndef TENSOR_MAX_POOL_SIZE
    #define TENSOR_MAX_POOL_SIZE 8
#endif

#ifndef TENSOR_CACHE_LINE_SIZE
    #define TENSOR_CACHE_LINE_SIZE 64
#endif


#if __GNUC__ || __has_attribute(aligned)
    #define BOOST_UBLAS_TENSOR_ALIGN __attribute__((aligned(TENSOR_CACHE_LINE_SIZE)))
#elif defined(_MSC_VER) && !__INTEL_COMPILER && _MSC_VER >= 1310
    #define BOOST_UBLAS_TENSOR_ALIGN __declspec(align(TENSOR_CACHE_LINE_SIZE))
#else
    #define BOOST_UBLAS_TENSOR_ALIGN
#endif

#if __GNUC__ || __has_attribute(__builtin_expect)
    #define BOOST_UBLAS_TENSOR_LIKLY(x,N) __builtin_expect(!!(x),N)
    #define BOOST_UBLAS_TENSOR_UNLIKLY(x,N) __builtin_expect(!!(x),N)
#elif defined(_MSC_VER) && !__INTEL_COMPILER && _MSC_VER >= 1310
    #define BOOST_UBLAS_TENSOR_LIKLY(x,N) x
    #define BOOST_UBLAS_TENSOR_UNLIKLY(x,N) BOOST_UBLAS_TENSOR_LIKLY(x,N)
#else
    #define BOOST_UBLAS_TENSOR_LIKLY(x,N) x
    #define BOOST_UBLAS_TENSOR_UNLIKLY(x,N) BOOST_UBLAS_TENSOR_LIKLY(x,N)
#endif

#if defined(_MSC_VER) 
 #if defined(_ARM_)

  __forceinline void YieldProcessor() { }
  extern "C" void __emit(const unsigned __int32 opcode);
  #pragma intrinsic(__emit)
  #define MemoryBarrier() { __emit(0xF3BF); __emit(0x8F5F); }

 #elif defined(_ARM64_)

  extern "C" void __yield(void);
  #pragma intrinsic(__yield)
  __forceinline void YieldProcessor() { __yield();}

  extern "C" void __dmb(const unsigned __int32 _Type);
  #pragma intrinsic(__dmb)
  #define MemoryBarrier() { __dmb(_ARM64_BARRIER_SY); }

 #elif defined(_AMD64_)
  
  extern "C" void
  _mm_pause (
      void
      );
  
  extern "C" void
  _mm_mfence (
      void
      );

  #pragma intrinsic(_mm_pause)
  #pragma intrinsic(_mm_mfence)
  
  #define YieldProcessor _mm_pause
  #define MemoryBarrier _mm_mfence

 #elif defined(_X86_)
  
  #define YieldProcessor() __asm { rep nop }
  #define MemoryBarrier() MemoryBarrierImpl()
  __forceinline void MemoryBarrierImpl()
  {
      int32_t Barrier;
      __asm {
          xchg Barrier, eax
      }
  }

 #else // !_ARM_ && !_AMD64_ && !_X86_
  #error Unsupported architecture
 #endif
#else // _MSC_VER

// Only clang defines __has_builtin, so we first test for a GCC define
// before using __has_builtin.

#if defined(__i386__) || defined(__x86_64__)

#if (__GNUC__ > 4 && __GNUC_MINOR > 7) || __has_builtin(__builtin_ia32_pause)
 // clang added this intrinsic in 3.8
 // gcc added this intrinsic by 4.7.1
 #define YieldProcessor __builtin_ia32_pause
#endif // __has_builtin(__builtin_ia32_pause)

#if defined(__GNUC__) || __has_builtin(__builtin_ia32_mfence)
 // clang has had this intrinsic since at least 3.0
 // gcc has had this intrinsic since forever
 #define MemoryBarrier __builtin_ia32_mfence
#endif // __has_builtin(__builtin_ia32_mfence)

// If we don't have intrinsics, we can do some inline asm instead.
#ifndef YieldProcessor
 #define YieldProcessor() asm volatile ("pause")
#endif // YieldProcessor

#ifndef MemoryBarrier
 #define MemoryBarrier() asm volatile ("mfence")
#endif // MemoryBarrier

#endif // defined(__i386__) || defined(__x86_64__)

#ifdef __aarch64__
 #define YieldProcessor() asm volatile ("yield")
 #define MemoryBarrier __sync_synchronize
#endif // __aarch64__

#ifdef __arm__
 #define YieldProcessor()
 #define MemoryBarrier __sync_synchronize
#endif // __arm__

#endif // _MSC_VER


#endif // _BOOST_UBLAS_TENSOR_DETAIL_PARALLEL_MACRO_HPP