#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_THREAD_POOL_V2_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_THREAD_POOL_V2_HPP

#include <cstddef>
#include <vector>
#include <thread>
#include <boost/numeric/ublas/tensor/parallel_v2/concurrent_queue/concurrentqueue.h>
#include <boost/numeric/ublas/tensor/parallel_v2/detail/parallel_macro.hpp>

namespace boost::numeric::ublas::parallel_v2{

    struct Args{
        float* c;
        float const* a;
        float const* b;
        size_t const* nc;
        size_t const* na;
        size_t const* nb;
        size_t const* wc;
        size_t const* wa;
        size_t const* wb;

        Args() = default;

        Args(float* c, size_t const* nc, size_t const* wc, 
            float const* a, size_t const* na, size_t const* wa, 
            float const* b, size_t const* nb, size_t const* wb)
            : c(c), nc(nc), wc(wc),
              a(c), na(nc), wa(wc),
              b(c), nb(nc), wb(wc)
        {}
    };
    
    struct task{

        task() = default;

        task( void (*fun)(Args&) noexcept, Args& args )
            : args( std::move(args) )
            , fun(fun)
        {}

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void call() noexcept{
            fun(args);
        }

    private:
        Args args;
        void (*fun)(Args&) noexcept {nullptr};
    };

    struct thread_pool{
        
        using queue_type = moodycamel::ConcurrentQueue<task>;
        friend struct chunk_handler;
        friend struct thread_args;

        struct chunk_handler{

            chunk_handler( thread_pool* p )
                : m_tp(p)
            {}

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void wait(){
                m_flag.store(false);
                // auto old = m_c2.load(std::memory_order_acquire);
                while( ( m_c1.load(std::memory_order_acquire) > 0 ) ){
                    auto old = m_c2.exchange(0,std::memory_order_acquire);
                    m_c1.fetch_sub(old,std::memory_order_relaxed);
                    YieldProcessor();
                    MemoryBarrier();
                }
            }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()(task& t) noexcept{
                m_tp->execute(std::move(t));
                add_task();
            }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto invoke_fn(task&& fn, std::atomic<size_t>& sz) noexcept{
                fn.call();
                remove_task();
                sz.fetch_sub(1,std::memory_order_relaxed);
            }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void add_task() noexcept{
                m_c1.fetch_add(1,std::memory_order_relaxed);
            }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE void remove_task() noexcept{
                // std::scoped_lock l(m_mu);
                if( m_flag.load(std::memory_order_acquire) ){
                    m_c2.fetch_add(1,std::memory_order_acquire);
                }else{
                    m_c1.fetch_sub(1, std::memory_order_relaxed );
                }
            }
            
            ~chunk_handler(){
                wait();
            }

        private:
            thread_pool*            m_tp;
            std::atomic< ssize_t >  m_c1{0};
            std::atomic< ssize_t >  m_c2{0};
            std::atomic< bool >     m_flag{true};
        };

    public:

        thread_pool( size_t count )
            : m_threads(count)
            , m_queues(count)
            , m_queues_size(count)
            , m_count(count)
        {
            size_t idx = 0;
            // for( auto& tid : m_threads ){
            //     m_queues_size[idx].store(0);
            //     thread_args* t = new thread_args{this, tid, idx};
            //     pthread_create(&tid, nullptr, &(thread_pool::thread_fn), static_cast<void*>(t) );
            //     ++idx;
            // }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr chunk_handler& get_chunk_handler() noexcept{
            return m_handler;
        } 

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void stop(){
            m_token.store(true);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
        void wait(){
            std::vector<pthread_t> threads(std::move(m_threads));
            if (BOOST_UBLAS_TENSOR_LIKLY ( !threads.empty(), true ) )
            {
                for (auto& t : threads){
                    pthread_join(t,nullptr);
                }
                    
            }
            m_threads.clear();
        }

        ~thread_pool(){
            stop();
            wait();
        }

    private:

        struct thread_args{
            thread_pool* tp;
            pthread_t id;
            size_t q_id;
        };

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void execute(task&& t)
        {
            auto old = m_thread_idx.load(std::memory_order_acquire);
            auto& queue = m_queues[old];
            m_thread_idx.compare_exchange_strong( old, (old + 1) % m_count, std::memory_order_relaxed );

            while( !queue.try_enqueue(std::move(t)) );

            m_queues_size[old].fetch_add(1,std::memory_order_relaxed);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr 
        void try_steel( task& t, queue_type& curr_q, size_t q_idx ) noexcept{
            for(auto& q : m_queues){
                auto& qsz = m_queues_size[q_idx];
                if( qsz.load(std::memory_order_acquire) && q.try_dequeue(t) ){
                    m_handler.invoke_fn(std::move(t),qsz);
                    break;
                }
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        void start(pthread_t id, queue_type& queue, size_t q_idx){
            task t;

            auto& qsz = m_queues_size[q_idx];

            while( true ){
                
                while( !( m_token.load(std::memory_order_acquire) || qsz.load(std::memory_order_acquire) ) ) {
                    try_steel(t,queue,q_idx);
                    YieldProcessor();
                    MemoryBarrier();
                }

                if( m_token.load(std::memory_order_acquire) || ( !qsz.load(std::memory_order_acquire) ) ){ return; };
                
                if( BOOST_UBLAS_TENSOR_LIKLY( queue.try_dequeue(t), true) ){
                    m_handler.invoke_fn(std::move(t),qsz);
                }
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE
        static void* thread_fn(void* args){
            auto& targs = *static_cast<thread_args*>(args);
            
            // std::cout<<targs.q_id<<'\n';

            targs.tp->start( targs.id, targs.tp->m_queues[targs.q_id], targs.q_id );

            delete static_cast<thread_args*>(args);
            pthread_exit(0);
        }

    private:
        std::vector<pthread_t>      m_threads;
        std::vector< queue_type >   m_queues;
        std::vector< std::atomic<size_t> >       m_queues_size;
        std::atomic<bool>           m_token{false};
        std::atomic<size_t>         m_thread_idx{0};
        size_t                      m_count{std::thread::hardware_concurrency()};
        chunk_handler               m_handler{this};
    };

} // namespace boost::numeric::ublas::parallel_v2

namespace boost::numeric::ublas::parallel_v2{
    
    static thread_pool pool( std::thread::hardware_concurrency() );

} // namespace boost::numeric::ublas::parallel_v2


#endif