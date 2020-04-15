#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_QUEUE_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_QUEUE_HPP

#include <list>
#include <mutex>
#include <thread>
#include "detail/queue_impl.hpp"

namespace boost::numeric::ublas::parallel{
    
    template<typename T, typename D = dummy_deleter>
    struct queue{
        using value_type            = T;
        using deleter_type          = D;
        using base_type             = std::list<value_type>;
        using size_type             = typename base_type::size_type;
        using reference_type        = typename base_type::reference;
        using pointer_type          = typename base_type::pointer;
        using const_reference_type  = typename base_type::const_reference;
        using const_pointer_type    = typename base_type::const_pointer;
        using iterator              = typename base_type::iterator;
        using reverse_iterator      = typename base_type::reverse_iterator;
        using const_iterator        = typename base_type::const_iterator;
        using const_reverse_iterator= typename base_type::const_reverse_iterator;

        queue() = default;
        queue(size_type size) 
            : m_data(size)
        {}

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto peek() const noexcept{
            TENSOR_ASSERT(!empty(), "boost::numeric::ublas::parallel::queue::peek() : queue is empty");
            std::scoped_lock l(m_mu);
            return m_data.front();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void push(value_type val) noexcept{
            m_data.push_back(std::move(val));
        }

        template<typename... Args>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE bool emplace(Args&&... val) noexcept{
            try{
                std::scoped_lock l(m_mu);
                m_data.emplace_back( std::forward<Args>(val)... );
                return true;
            }catch(...){
                return false;
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE value_type pop() noexcept{
            TENSOR_ASSERT(!empty(), "boost::numeric::ublas::parallel::queue::peek() : queue is empty");
            auto front = peek();
            std::scoped_lock l(m_mu);
            m_data.pop_front();
            return front;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void pop_and_clean() noexcept{
            TENSOR_ASSERT(!empty(), "boost::numeric::ublas::parallel::queue::peek() : queue is empty");
            auto p = pop();
            std::scoped_lock l(m_mu);
            m_del(p);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE value_type steel() noexcept{
            TENSOR_ASSERT(!empty(), "boost::numeric::ublas::parallel::queue::peek() : queue is empty");
            std::scoped_lock l(m_mu);
            auto back = m_data.back();
            m_data.pop_front();
            return back;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE bool empty() const noexcept{
            std::scoped_lock l(m_mu);
            return m_data.empty();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void reset() noexcept{
            
            while(!empty()) pop_and_clean();
        }

        #if 0
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr iterator begin() noexcept{ return m_data.begin(); }
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr iterator end() noexcept{ return m_data.end(); }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr const_iterator begin() const noexcept{ return m_data.begin(); }
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr const_iterator end() const noexcept{ return m_data.end(); }
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr reverse_iterator rbegin() noexcept{ return m_data.rbegin(); }
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr reverse_iterator rend() noexcept{ return m_data.rend(); }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr const_reverse_iterator rbegin() const noexcept{ return m_data.rbegin(); }
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr const_reverse_iterator rend() const noexcept{ return m_data.rend(); }
        
        #endif

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE size_type size() const noexcept { 
            std::scoped_lock l(m_mu);
            return m_data.size(); 
        }

        ~queue(){
            reset();
        }

    private:
        base_type m_data;
        mutable std::mutex m_mu;
        deleter_type m_del;
    };

} // namespace boost::numeric::ublas::parallel


#endif
