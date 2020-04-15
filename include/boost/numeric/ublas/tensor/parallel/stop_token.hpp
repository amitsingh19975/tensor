#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_STOP_TOKEN_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_STOP_TOKEN_HPP

#include "../fwd.hpp"
#include "detail/stop_token_impl.hpp"
#include <type_traits>

namespace boost::numeric::ublas::parallel{

    struct nostopstate{};

    inline constexpr auto nostopstate_t = nostopstate{};

    struct stop_token{
        constexpr stop_token() noexcept = default;

        constexpr stop_token( stop_token const& ) noexcept = default;
        constexpr stop_token( stop_token && ) noexcept = default;

        explicit stop_token( detail::stop_state_ref const& state ) noexcept 
            : m_state(state)
        {}

        ~stop_token() noexcept = default;

        stop_token& operator=(stop_token const&) = default;
        stop_token& operator=(stop_token &&) = default;

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool stop_possible() const noexcept{
            return m_state && m_state->stop_possible();
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool stop_requested() const noexcept{
            return m_state && m_state->stop_requested();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void swap( stop_token& rhs ) noexcept{
            m_state.swap(rhs.m_state);
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend bool operator==( stop_token& lhs, stop_token const& rhs){
            return lhs.m_state == rhs.m_state;
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend bool operator!=( stop_token& lhs, stop_token const& rhs){
            return !(lhs == rhs);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend void swap( stop_token& lhs, stop_token& rhs){
            lhs.swap(rhs);
        }

    private:
        friend struct stop_source;
        template< typename Callback > friend struct stop_callback;
        detail::stop_state_ref m_state;
    };

    struct stop_source{
        stop_source()
            : m_state(*this)
        {}

        constexpr explicit stop_source(nostopstate) noexcept{}

        constexpr stop_source( stop_source const& other ) noexcept 
            : m_state(other.m_state)
        {
            if( m_state ){
                m_state->add_ssrc();
            }
        }

        constexpr stop_source(stop_source&&) noexcept = default;

        stop_source& operator=( stop_source const& other ) noexcept{
            if( m_state != other.m_state ){
                stop_source sink(std::move(*this));
                m_state = other.m_state;
                if(m_state){
                    m_state->add_ssrc();
                }
            }
            return *this;
        }

        stop_source& operator=( stop_source&& other ) noexcept = default;

        ~stop_source(){
            if( m_state ){
                m_state->sub_ssrc();
            }
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool stop_possible() const noexcept {
            return static_cast<bool>(m_state);
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool stop_requested() const noexcept {
            return static_cast<bool>(m_state) && m_state->stop_requested();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool request_stop() const noexcept {
            if( stop_possible() ){
                return m_state->request_stop();
            }
            return false;
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE stop_token get_token() const noexcept {
            return stop_token{m_state};
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void swap( stop_source& other ) noexcept{
            m_state.swap(other.m_state);
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend bool operator==( stop_source& lhs, stop_source const& rhs){
            return lhs.m_state == rhs.m_state;
        }

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend bool operator!=( stop_source& lhs, stop_source const& rhs){
            return !(lhs == rhs);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend void swap( stop_source& lhs, stop_source& rhs){
            lhs.swap(rhs);
        }

    private:
        detail::stop_state_ref m_state;
    };

    template< typename Callback >
    struct [[nodiscard]] stop_callback{
        static_assert( std::is_nothrow_destructible_v<Callback> );
        static_assert( std::is_invocable_v<Callback> );

        using callback_type = Callback;

        template<typename C, std::enable_if_t< std::is_constructible_v<callback_type,C>, int > = 0 >
        explicit stop_callback( stop_token& token, C&& callback)
            noexcept(std::is_nothrow_constructible_v<callback_type,C>)
            : m_callback(std::forward<C>(callback))
        {
            if( auto state = token.m_state ){
                if(state->register_callback(&m_callback)){
                    m_state.swap(state);
                }
            }
        }

        template<typename C, std::enable_if_t< std::is_constructible_v<callback_type,C>, int > = 0 >
        explicit stop_callback( stop_token&& token, C&& callback)
            noexcept(std::is_nothrow_constructible_v<callback_type,C>)
            : m_callback(std::forward<C>(callback))
        {
            if( auto& state = token.m_state ){
                if(state->register_callback(&m_callback)){
                    m_state.swap(state);
                }
            }
        }

        ~stop_callback(){
            if( m_state ){
                m_state->remove_callback(&m_callback);
            }
        }

        stop_callback(  stop_callback const& ) = delete;
        stop_callback(  stop_callback&& ) = delete;
        stop_callback& operator=(  stop_callback const& ) = delete;
        stop_callback& operator=(  stop_callback&& ) = delete;

    private:

        struct cb_impl : detail::stop_cb{
            using base_type = detail::stop_cb;

            template<typename C>
            constexpr explicit cb_impl(C&& callback)
                : base_type(&s_execute),
                  m_callback(std::forward<C>(callback))
            {}

            static void s_execute( base_type* self ) noexcept{
                auto& cb = static_cast<cb_impl*>(self)->m_callback;
                std::forward<callback_type>(cb)();
            }
            
        private:
            callback_type m_callback;
        };
    private:
        callback_type m_callback;
        detail::stop_state_ref m_state;
    };

    template<typename Callback>
    stop_callback( stop_token, Callback ) -> stop_callback<Callback>;

} // namespace boost::numeric::ublas::parallel


#endif