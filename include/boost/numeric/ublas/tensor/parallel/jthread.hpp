#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_JTHREAD_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_JTHREAD_HPP

#include "stop_token.hpp"

namespace boost::numeric::ublas::parallel{
    
    struct jthread{
        using id = std::thread::id;
        using native_handle_type = std::thread::native_handle_type;

        jthread() noexcept
            : m_stop_source(nostopstate_t)
        {}

        template<typename Callback, typename... Args>
        jthread(Callback&& cb, Args&&... args)
            : m_thread(s_create(m_stop_source, std::forward<Callback>(cb), std::forward<Args>(args)...))
        {}

        jthread(jthread const&) = delete;
        jthread(jthread&&) = delete;

        ~jthread(){
            if( joinable() ){
                request_stop();
                join();
            }
        }

        jthread& operator=(jthread const&) = delete;
        jthread& operator=(jthread&&) = delete;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void swap(jthread& other) noexcept{
            std::swap(m_stop_source, other.m_stop_source);
            std::swap(m_thread,other.m_thread);
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool joinable() const noexcept{
            return m_thread.joinable();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void join() noexcept{
            return m_thread.join();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void detach() noexcept{
            return m_thread.detach();
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE id get_id() const noexcept{
            return m_thread.get_id();
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE native_handle_type native_handle() noexcept{
            return m_thread.native_handle();
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE static unsigned hardware_concurrency() noexcept{
            return std::thread::hardware_concurrency();
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE stop_source get_stop_source() const noexcept{
            return m_stop_source;
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE stop_token get_stop_token() const noexcept{
            return m_stop_source.get_token();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool request_stop() noexcept{
            return m_stop_source.request_stop();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend void swap( jthread& lhs, jthread& rhs ) noexcept{
            return lhs.swap(rhs);
        }

    private:
        template<typename Callback, typename... Args>
        static std::thread s_create( stop_source& ssrc, Callback&& cb, Args&&... args ){
            if constexpr( std::is_invocable_v< std::decay_t<Callback>, stop_token, std::decay_t<Args>... > ){
                return std::thread{
                    std::forward<Callback>(cb), ssrc.get_token(),
                    std::forward<Args>(args)...
                };
            }else{
                static_assert(
                    std::is_invocable_v< std::decay_t<Callback>, std::decay_t<Args>... >,
                    "boost::numeric::ublas::parallel::jthread::s_create() : arguments must be invocable after conversion to rvalues"
                );

                return std::thread{
                    std::forward<Callback>(cb),
                    std::forward<Args>(args)...
                };
            }
        }

    private:
        stop_source m_stop_source;
        std::thread m_thread;
    };

} // namespace boost::numeric::ublas::parallel
#endif