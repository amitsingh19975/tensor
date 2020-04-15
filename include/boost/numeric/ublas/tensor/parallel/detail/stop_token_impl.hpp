#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_STOP_TOKEN_IMPL_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_STOP_TOKEN_IMPL_HPP

#include "../fwd.hpp"
#include "semaphore.hpp"

namespace boost::numeric::ublas::parallel::detail{
    struct stop_cb{
        using cb_type = void( stop_cb* ) noexcept;

        stop_cb( cb_type* cb )
            : callback(cb)
        {}

        constexpr void run() noexcept { callback(this); }

        cb_type* callback;
        stop_cb* prev{nullptr};
        stop_cb* next{nullptr};
        bool* destroyed{nullptr};
        detail::binary_semaphore done{0};

    };

    struct stop_state_t{
        using value_type = uint32_t;

        static constexpr value_type s_stop_request_bit{1};
        static constexpr value_type s_lock_bit{2};
        static constexpr value_type s_ssrc_counter_inc{4};

        stop_state_t() = default;

        constexpr bool stop_possible() const noexcept{
            return m_value.load(std::memory_order_acquire) & ~s_lock_bit;
        }

        constexpr bool stop_requested() const noexcept{
            return m_value.load(std::memory_order_acquire) & s_stop_request_bit;
        }

        constexpr void add_owner() noexcept{
            m_owner.fetch_add(1,std::memory_order_relaxed);
        }

        constexpr void release_ownership() noexcept{
            if( m_owner.fetch_add(1,std::memory_order_release) == 1 ){
                delete this;
            }
        }

        constexpr void add_ssrc() noexcept{
            m_value.fetch_add(s_ssrc_counter_inc,std::memory_order_relaxed);
        }

        constexpr void sub_ssrc() noexcept{
            m_value.fetch_sub(s_ssrc_counter_inc,std::memory_order_release);
        }

        constexpr void lock() noexcept{
            auto old = m_value.load(std::memory_order_relaxed);
            while( !try_lock(old,std::memory_order_relaxed) ){}
        }


        constexpr void unlock() noexcept{
            m_value.fetch_sub(s_lock_bit,std::memory_order_release);
        }


        constexpr bool request_stop() noexcept{
            auto old = m_value.load(std::memory_order_acquire);
            do{
                if( old & s_stop_request_bit ){
                    return false;
                }
            }while(!try_lock_and_stop(old));

            m_requester = std::this_thread::get_id();

            while(m_head){
                bool last_cb{};
                stop_cb* cb = m_head;
                m_head = m_head->next;

                if( m_head ){
                    m_head->prev = nullptr;
                    last_cb = false;
                }else{
                    last_cb = true;
                }

                unlock();

                bool destroyed = false;
                cb->destroyed = &destroyed;

                cb->run();

                if( !destroyed ){
                    cb->destroyed = nullptr;
                    cb->done.release();
                }

                if( last_cb ){
                    return true;
                }
                lock();
            }
            unlock();
            return true;
        }

        constexpr bool register_callback( stop_cb* cb ) noexcept{
            auto old = m_value.load(std::memory_order_acquire);
            do{

                if( old & s_stop_request_bit ){
                    cb->run();
                    return false;
                }
                if( old < s_ssrc_counter_inc ){
                    return false;
                }

            }while(!try_lock(old));

            cb->next = m_head;
            if( m_head ){
                m_head ->prev = cb;
            }
            m_head = cb;
            unlock();
            return true;
        }

        constexpr void remove_callback( stop_cb* cb ) noexcept{
            lock();

            if( cb == m_head ){
                m_head = m_head->next;
                if(m_head){
                    m_head->prev = nullptr;
                }
                unlock();
                return;
            }else if( cb->prev ){
                cb->prev->next = cb->next;
                if( cb->next ){
                    cb->next->prev = cb->prev;
                    unlock();
                    return;
                }
            }
            unlock();

            if( m_requester != std::this_thread::get_id() ){
                cb->done.acquire();
                return;
            }

            if( cb->destroyed ){
                *(cb->destroyed) = true;
            }
        }

        constexpr bool try_lock( value_type& curr_val, std::memory_order failure = std::memory_order_acquire ) noexcept{
            return do_try_lock(curr_val, 0, std::memory_order_acquire, failure);
        }

        constexpr bool try_lock_and_stop( value_type& curr_val ) noexcept{
            return do_try_lock(curr_val, s_stop_request_bit, std::memory_order_acq_rel, std::memory_order_acquire);
        }

        constexpr bool do_try_lock( value_type& curr_val, value_type nbits, std::memory_order success, std::memory_order failure ) noexcept{
            if( curr_val & s_lock_bit ){
                std::this_thread::yield();
                curr_val = m_value.load(failure);
                return false;
            }
            nbits |= s_lock_bit;
            return m_value.compare_exchange_weak(curr_val, curr_val | nbits, success, failure);
        }

    private:
        std::atomic<value_type> m_owner{1};
        std::atomic<value_type> m_value{s_ssrc_counter_inc};
        stop_cb* m_head{nullptr};
        std::thread::id m_requester;
    };


    struct stop_state_ref{
        stop_state_ref() noexcept = default;
        
        explicit stop_state_ref( stop_source const& )
            : m_stop_state(new stop_state_t())
        {}

        constexpr stop_state_ref( stop_state_ref const& other ) noexcept
            : m_stop_state(other.m_stop_state)
        {
            if( m_stop_state ){
                m_stop_state->add_owner();
            }
        }

        constexpr stop_state_ref( stop_state_ref&& other ) noexcept
            : m_stop_state(other.m_stop_state)
        {
            other.m_stop_state = nullptr;
        }

        constexpr stop_state_ref& operator=( stop_state_ref const& other ) noexcept{
            auto p = other.m_stop_state;
            if( p != m_stop_state ){
                if( p ){
                    p->add_owner();
                }
                if( m_stop_state ){
                    m_stop_state->release_ownership();
                }
                m_stop_state = p;
            }
            return *this;
        }

        stop_state_ref& operator=( stop_state_ref&& other ) noexcept{
            stop_state_ref(std::move(other)).swap(*this);
            return *this;
        }

        ~stop_state_ref(){
            if( m_stop_state ) m_stop_state->release_ownership();
        }

        void swap( stop_state_ref& other ) noexcept{
            std::swap(m_stop_state, other.m_stop_state);
        }

        constexpr explicit operator bool() const noexcept { return m_stop_state != nullptr; }
        constexpr stop_state_t* operator->() const noexcept{ return m_stop_state; }

        friend bool operator==( stop_state_ref const& lhs, stop_state_ref const& rhs){
            return lhs.m_stop_state == rhs.m_stop_state;
        };

        friend bool operator!=( stop_state_ref const& lhs, stop_state_ref const& rhs){
            return !(lhs == rhs);
        };

    private:
        stop_state_t* m_stop_state{nullptr};
    };

} // namespace boost::numeric::ublas::parallel::detail


#endif