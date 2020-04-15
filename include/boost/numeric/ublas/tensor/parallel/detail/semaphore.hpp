#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_SEMAPHORE_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_SEMAPHORE_HPP

#include "../fwd.hpp"
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>

namespace boost::numeric::ublas::parallel::detail{
    
    struct binary_semaphore{
        constexpr explicit binary_semaphore(int d) 
            : m_counter( d > 0 )
        {}

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void release() noexcept {
            m_counter.fetch_add(1,std::memory_order_release);
        }

        void acquire(){
            int old = 1;
            while( !m_counter.compare_exchange_weak(old,0,std::memory_order_acquire,std::memory_order_relaxed) ){
                old = 1;
                std::this_thread::yield();
            }
        }

    private:
        std::atomic_int m_counter;
    };

    struct semaphore_impl{
        explicit semaphore_impl( int c )
            : m_counter(c)
            , m_max_count(c)
        {}

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE bool wait() noexcept{
            std::unique_lock l(m_mu);
            m_cond.wait(l,[this]{ m_counter <= 0; });
            return true;
        }

        template<typename Rep, typename Period>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE bool wait_for(std::chrono::duration<Rep,Period> const& time) noexcept{
            std::unique_lock l(m_mu);
            m_cond.wait_for(l,time,[this]{ m_counter <= 0; });
            return true;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE bool try_wait() noexcept{
            std::unique_lock l(m_mu,std::try_to_lock);
            if ( !l.owns_lock() ) return false;
            m_cond.wait(l,[this]{ m_counter <= 0; });
            return true;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void signal() noexcept{
            std::scoped_lock l(m_mu);
            m_counter = std::min(++m_counter,m_max_count);
            m_cond.notify_one();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void signal(int count) noexcept{
            std::scoped_lock l(m_mu);
            while( count-- > 0 ){
                m_counter = std::min(++m_counter,m_max_count);
                m_cond.notify_one();
            }
        }

    private:
        std::condition_variable m_cond;
        int                     m_counter;
        int                     m_max_count;
        std::mutex              m_mu;
    };
    

} // namespace boost::numeric::ublas::parallel::detail


namespace boost::numeric::ublas::parallel
{
    
    struct semaphore{

        semaphore( ssize_t count = 0 )
            : m_sema(count)
        {}
    
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool try_wait() noexcept{
            auto old = m_counter.load(std::memory_order_relaxed);
            while( old > 0 ){
                if( m_counter.compare_exchange_weak(old, old - 1, std::memory_order_acquire, std::memory_order_relaxed) ) {
                    return true;
                }
            }
            return false;
        }
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool try_wait_many(ssize_t max) noexcept{
            TENSOR_ASSERT(max >= 0, "boost::numeric::ublas::parallel::semaphore::try_wait_many : max cannot be less than 0");
            auto old = m_counter.load(std::memory_order_relaxed);
            while( old > 0 ){
                ssize_t nCount = old > max ? old - max : 0;
                if( m_counter.compare_exchange_weak(old, nCount, std::memory_order_acquire, std::memory_order_relaxed) ) {
                    return old - nCount;
                }
            }

            return false;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool wait(std::chrono::duration<uint32_t,std::micro> const& t) noexcept{
            return try_wait() || wait_with_partial_spinning(t);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool wait() noexcept{
            return try_wait() || wait_with_partial_spinning();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr ssize_t wait_many(ssize_t max, std::chrono::duration<uint32_t,std::micro> const& t) noexcept{
            TENSOR_ASSERT(max >= 0, "boost::numeric::ublas::parallel::semaphore::wait_many : max cannot be less than 0");
            ssize_t result = try_wait_many(max);
            if (result == 0 && max > 0)
                result = wait_many_with_partial_spinning(max, t);
            return result;
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr ssize_t wait_many(ssize_t max) noexcept{
            TENSOR_ASSERT(max >= 0, "boost::numeric::ublas::parallel::semaphore::wait_many : max cannot be less than 0");
            ssize_t result = wait_many(max,std::chrono::microseconds(0));
            return result;
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr ssize_t availableApprox() const noexcept{
            ssize_t count = m_counter.load(std::memory_order_relaxed);
		    return count > 0 ? count : 0;
        }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void signal(ssize_t count = 1) noexcept{
            TENSOR_ASSERT(count >= 0, "boost::numeric::ublas::parallel::semaphore::wait_many : max cannot be less than 0");
            ssize_t old = m_counter.fetch_add(count, std::memory_order_release);
            ssize_t to_release = -old < count ? -old : count;
            if (to_release > 0)
            {
                m_sema.signal( static_cast<int>(to_release) );
            }
        }

    private:

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool wait_with_partial_spinning(std::chrono::duration<uint32_t,std::micro> const& t = std::chrono::microseconds(0), int spin = 10000){
            ssize_t old = 0;
            while(--spin >= 0){
                old = m_counter.load(std::memory_order_relaxed);
                if( ( old > 0 ) && m_counter.compare_exchange_strong(old, old - 1, std::memory_order_acquire, std::memory_order_relaxed) ) {
                    return true;
                }
                std::atomic_signal_fence(std::memory_order_acquire);
            }
            old = m_counter.fetch_sub(1,std::memory_order_acquire);
            if( old > 0 ){
                return true;
            }else if ( t == std::chrono::microseconds(0) ){
                return m_sema.wait();
            }else if ( m_sema.wait_for(t) ){
                return true;
            }

            while(true){
                old = m_counter.load(std::memory_order_acquire);
                if( old >= 0 && m_sema.try_wait() ) return true;
                if( ( old < 0 ) && m_counter.compare_exchange_strong(old, old + 1, std::memory_order_relaxed, std::memory_order_relaxed) ) {
                    return true;
                }
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr ssize_t wait_many_with_partial_spinning(ssize_t max, std::chrono::duration<uint32_t,std::micro> const& t = std::chrono::microseconds(0), int spin = 10000){
            TENSOR_ASSERT(max >= 0, "boost::numeric::ublas::parallel::semaphore::waitManyWithPartialSpining : max cannot be less than 0");

            ssize_t old = 0;
            while(--spin >= 0){
                old = m_counter.load(std::memory_order_relaxed);
                if( old > 0 ){
                    ssize_t nCount = old > max ? old - max : 0;
                    if( m_counter.compare_exchange_strong(old, nCount, std::memory_order_acquire, std::memory_order_relaxed) ) {
                        return old - nCount;
                    }
                }
                std::atomic_signal_fence(std::memory_order_acquire);
            }
            old = m_counter.fetch_sub(1,std::memory_order_acquire);
            
            if( old <= 0 ){
                if ( t == std::chrono::microseconds(0) ){
                    if( !m_sema.wait() ) return 0;
                }else if( !m_sema.wait_for(t) ){
                    while(true){
                        old = m_counter.load(std::memory_order_acquire);
                        if( old >= 0 && m_sema.try_wait() ) return true;
                        if( ( old < 0 ) && m_counter.compare_exchange_strong(old, old + 1, std::memory_order_relaxed, std::memory_order_relaxed) ) {
                            return true;
                        }
                    }
                }
            }

            if (max > 1)
                return 1 + try_wait_many(max - 1);
            return 1;

        }

    private:
        std::atomic<ssize_t>    m_counter;
        detail::semaphore_impl  m_sema;
    };

} // namespace boost::numeric::ublas::parallel


#endif