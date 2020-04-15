#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_CHUNK_THREAD_POOL_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_CHUNK_THREAD_POOL_HPP

#include <thread>
#include "blocking.hpp"
#include "../concurrent_queue/concurrentqueue.h"
#include <list>
#include <new>
#include <condition_variable>
#include <mutex>
#include <future>
#include <boost/core/demangle.hpp>
#include <iostream>
#include <unordered_set>

namespace boost::numeric::ublas::parallel::execution{
    
    struct chunk_thread_pool;

    template<typename ExecuteFunction, typename Function, typename Allocator>
    struct chunk_state{
        using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<chunk_state>;
        chunk_state(ExecuteFunction efn,Function fn, allocator_type const& a, std::function<void()> f)
            : m_ex_fn(std::move(efn))
            , m_fn(std::move(fn))
            , m_alloc(a)
            , m_wait(std::move(f))
        {}

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void wait(){
            m_f.store(false);
            
            // while(  ){
            //     std::this_thread::yield();
            // }
            // std::unique_lock lck(m_mu);
            // m_cond.wait(lck,[this]{ return m_count1.load(std::memory_order_acquire) > 0 && !m_f.load(std::memory_order_acquire); });
            while( m_count1.load(std::memory_order_acquire) > 0 && !m_f.load(std::memory_order_acquire) ){
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
            // std::cout<<m_count1<<' '<<m_count2<<' '<<m_f<<'\n';
            // exit(0);
            // m_wait();
        }
        
        template<typename ArgsStruct>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()(ArgsStruct&& args) noexcept{
            m_ex_fn(std::move(args),*this);
        }
        
        template<typename ArgsStruct>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto invoke_fn(ArgsStruct&& args) noexcept{
            m_fn(std::move(args));
            remove_task();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void add_task(){
            m_count1.fetch_add(1,std::memory_order_acquire);
            // std::scoped_lock l(m_mu2);
            // m_cond.notify_all();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void remove_task(){
            if( m_f.load() == true ){
                m_count2.fetch_add(1,std::memory_order_acquire);
                return;
            }
            // std::cout<<m_count1<<' '<<m_count2<<' '<<m_f<<'\n';
            auto old = m_count2.load(std::memory_order_acquire);
            m_count1.fetch_sub(1 + old,std::memory_order_acquire);
            
            m_count2.store( 0 );
            old = m_count1.load(std::memory_order_acquire);
            if( old <= 0 ){
                m_count1.store( 0 );
            }
            // std::scoped_lock l(m_mu2);
            // m_cond.notify_all();
        }

    private:
        ExecuteFunction m_ex_fn;
        Function m_fn;
        std::atomic< ssize_t > m_count1{0};
        std::atomic< ssize_t > m_count2{0};
        std::atomic< bool > m_f{true};
        allocator_type m_alloc;
        std::function<void()> m_wait;
        std::mutex m_mu;
        std::mutex m_mu2;
        std::condition_variable m_cond;
    };

    struct chunk_thread_pool{
    private:

        template<typename, typename T, typename U> struct dependent_is_same : std::is_same<T,U>{};

        template<typename Interface, typename Blocking, typename Continuation, typename Work, typename ProtoAllocator>
        struct executor_impl{

            using size_type = size_t;

            executor_impl( executor_impl const& other )
                : m_pool(other.m_pool)
            {
                m_pool->work_up(Work{});
            }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr chunk_thread_pool& query(execution::context_t){ return *m_pool; }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static Interface query(execution::executor_concept_t) { return {}; }

            executor_impl<execution::oneway_t,Blocking,Continuation,Work,ProtoAllocator>
            require_concept(execution::oneway_t){ return {m_pool,m_allocator};}
            executor_impl<execution::bulk_oneway_t,Blocking,Continuation,Work,ProtoAllocator>
            require_concept(execution::bulk_oneway_t){ return {m_pool,m_allocator};}

            executor_impl<Interface,execution::blocking_t::never_t,Continuation,Work,ProtoAllocator>
            require(execution::blocking_t::never_t){ return {m_pool,m_allocator};}
            executor_impl<Interface,execution::blocking_t::always_t,Continuation,Work,ProtoAllocator>
            require(execution::blocking_t::always_t){ return {m_pool,m_allocator};}
            executor_impl<Interface,execution::blocking_t::possibly_t,Continuation,Work,ProtoAllocator>
            require(execution::blocking_t::possibly_t){ return {m_pool,m_allocator};}
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr execution::blocking_t query(execution::blocking_t) { return Blocking{}; }

            executor_impl<Interface, Blocking, execution::relationship_t::fork_t, Work, ProtoAllocator>
            require(execution::relationship_t::fork_t) const { return {m_pool, m_allocator}; };
            executor_impl<Interface, Blocking, execution::relationship_t::continuation_t, Work, ProtoAllocator>
            require(execution::relationship_t::continuation_t) const { return {m_pool, m_allocator}; };
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr execution::relationship_t query(execution::relationship_t) { return Continuation{}; }

            executor_impl<Interface, Blocking, Continuation, execution::outstanding_work_t::untracked_t, ProtoAllocator>
            require(execution::outstanding_work_t::untracked_t) const { return {m_pool, m_allocator}; };
            executor_impl<Interface, Blocking, Continuation, execution::outstanding_work_t::tracked_t, ProtoAllocator>
            require(execution::outstanding_work_t::tracked_t) const { return {m_pool, m_allocator}; };
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr execution::outstanding_work_t query(execution::outstanding_work_t) { return Work{}; }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr execution::bulk_guarantee_t query(execution::bulk_guarantee_t) { return execution::bulk_guarantee.parallel; }
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr execution::mapping_t query(execution::mapping_t) { return execution::mapping.thread; }

            executor_impl<Interface, Blocking, Continuation, Work, std::allocator<void>>
            require(const execution::allocator_t<void>&) const { return {m_pool, std::allocator<void>{}}; };
            template<typename NewProtoAllocator>
            executor_impl<Interface, Blocking, Continuation, Work, NewProtoAllocator>
            require(const execution::allocator_t<NewProtoAllocator>& a) const { return {m_pool, a.value()}; }
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE ProtoAllocator query(const execution::allocator_t<ProtoAllocator>&) const noexcept { return m_allocator; }
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE ProtoAllocator query(const execution::allocator_t<void>&) const noexcept { return m_allocator; }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE size_t query(const execution::occupancy_t&) const noexcept { return m_pool->m_threads.size(); }

            bool running_in_this_thread() const noexcept { return m_pool->running_in_this_thread(); }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend bool operator==(const executor_impl& a, const executor_impl& b) noexcept{
             return a.m_pool == b.m_pool;
            }

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend bool operator!=(const executor_impl& a, const executor_impl& b) noexcept{
             return a.m_pool != b.m_pool;
            }

            template<typename F,
                typename std::enable_if<std::is_same<Interface, execution::oneway_t>::value
                >::type* = nullptr>
            auto get_chunk_state(F fn) const
            {
                return m_pool->get_chunk_state(m_allocator, std::move(fn));
            }

        private:
            friend struct chunk_thread_pool;
            
            executor_impl( chunk_thread_pool* pool, ProtoAllocator const& all )
                : m_pool(pool)
                , m_allocator(all)
            {
                m_pool->work_up(Work{});
            }

        private:
            friend struct chunk_thread_pool;
            chunk_thread_pool*      m_pool;
            ProtoAllocator          m_allocator;
        };

    public:

        using executor_type = executor_impl<
            execution::oneway_t,
            execution::blocking_t::possibly_t,
            execution::relationship_t::chunk_t,
            execution::outstanding_work_t::untracked_t,
            std::allocator<void>
        >;

        explicit chunk_thread_pool(size_t count)
            : m_count(count)
        {
            while(count--){
                m_threads.emplace_back([this](){attach();});
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE executor_type executor(){
            
            return executor_type{ this, std::allocator<void>{} };
        } 

        void attach(){
            // thread_private_state prt(this);
            using ptr = task;

            while( !( m_token.load(std::memory_order_acquire) || ( m_work.load(std::memory_order_acquire) == 0 && !m_queue.size_approx() ) ) ){
                while( !(m_token.load(std::memory_order_acquire) || m_work.load(std::memory_order_acquire) == 0 || m_queue.size_approx()) ) {
                        // __asm volatile ("pause" ::: "memory");
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                }

                if( m_token.load(std::memory_order_acquire) || ( m_work.load(std::memory_order_acquire) == 0 && !m_queue.size_approx() ) ){ return; };
                ptr fn;
                if( m_queue.try_dequeue(fn) ){
                    // m_mutex.lock();
                    fn.call();
                }
            }

            // for( std::unique_lock<std::mutex> lck(m_mutex); !m_token.load(std::memory_order_acquire); ){
            //     m_cond.wait(lck,[this]{
            //         return m_token.load(std::memory_order_acquire) || m_work.load(std::memory_order_acquire) == 0 || m_queue.size_approx();
            //     });
            //     if( m_token.load(std::memory_order_acquire) || ( m_work.load(std::memory_order_acquire) == 0 && !m_queue.size_approx() ) ){ return; };
                
            //     // std::cout<<m_token<<' '<<m_queue.size_approx()<<'\n';
            //     lck.unlock();
            //     ptr fn;
            //     if( m_queue.try_dequeue(fn) ){
            //         fn.call();
            //         // fn.destroy();
            //         // std::cout<<"hell[b\n";
            //         // f();
            //     }
            //     lck.lock();
            // }
        }

        void stop(){
            m_token.store(true);
            std::lock_guard<std::mutex> lck(m_mutex);
            m_cond.notify_all();
        }

        void wait()
        {
            std::list<std::thread> threads(std::move(m_threads));
            if (!threads.empty())
            {
                std::unique_lock<std::mutex> lck(m_mutex);
                m_cond.notify_all();
                lck.unlock();
                m_work.fetch_sub(1,std::memory_order_acquire);
                for (auto& t : threads){
                    t.join();
                }
                    
            }
            m_threads.clear();
            // while(m_queue.size_approx()){
            //     task_base::pointer p;
            //     m_queue.wait_dequeue(p);
            //     p.destroy();
            // }
        }

        ~chunk_thread_pool(){
            stop();
            wait();
        }
    
    private:
        template<class Function>
        static void invoke(Function& f) noexcept
        {
            f();
        }
        // struct task_base{
            

        //     virtual ~task_base(){}

        //     struct deleter{
        //         void operator()(task_base* fp){ fp->destroy(); }
        //     };

        //     virtual void destroy(){};
        //     virtual void call(){};
        //     virtual std::function<void()> fn(){ return []{}; };

        //     using pointer = task_base;
        // };

        // template<typename ArgsStruct>
        struct task{

            template<typename ArgsStruct, typename ChunkHandler>
            task( ArgsStruct&& f, ChunkHandler& ch )
            {
                m_fn = [&,args = std::move(f)]{ ch.invoke_fn(std::move(args)); };
            }

            task() = default;

            ~task(){}

            // static task_base::pointer create( ArgsStruct f){
            //     return task(std::move(f));
            // }

            void call() {
                m_fn();
            }

        private:
            std::function<void()> m_fn;
        };

        // template<typename ArgsStruct, typename Allocator, typename ChunkHandler>
        // struct task : task_base{
            
        //     using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<task>;
        //     using pointer = task*;
        //     task() = default;
        //     explicit task( ArgsStruct f, Allocator const& all, ChunkHandler& h )
        //         : m_args(std::move(f))
        //         , m_allocator(all)
        //         , m_ch(h)
        //     {}

        //     ~task(){}

        //     static task_base::pointer create( ArgsStruct f, Allocator const& all, ChunkHandler& h ){
        //         // allocator_type allocator(all);
        //         // pointer fp = allocator.allocate(1);
        //         // try{
        //         //    return task_base::pointer{ new(fp) task(std::move(f),all,h) }; 
        //         // }catch(...){
        //         //     allocator.deallocate(fp,1);
        //         //     throw;
        //         // }
        //         return task(std::move(f),all,h);
        //     }

        //     virtual void destroy() override{
        //         pointer p = this;
        //         p->~task();
        //         // m_allocator.deallocate(p,1);
        //     }

        //     virtual void call() override{
        //         // pointer fp(this);
        //         // ArgsStruct&& args = std::move(m_args);
        //         // fp->destroy();
                
        //         m_ch.invoke_fn(std::move(m_args));
        //     }

        //     virtual std::function<void()> fn() override{
        //         return [args = std::move(m_args), &ch = m_ch]()
        //             mutable {
        //                 return ch.invoke_fn( std::move(args) );
        //             };
        //     }

        //     ArgsStruct m_args;
        //     allocator_type m_allocator;
        //     ChunkHandler& m_ch;
        // };
    
    struct thread_private_state{

        explicit thread_private_state( chunk_thread_pool* pool )
            : m_pool(pool)
        {
            instance() = m_prev_state;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE static thread_private_state*& instance(){
            static thread_local thread_private_state* state;
            return state;
        }

        chunk_thread_pool*                      m_pool;
        thread_private_state*                   m_prev_state{instance()};
    };

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE bool running_in_this_thread() const noexcept{
        auto* inst = thread_private_state::instance();
        if( inst != nullptr && inst->m_pool == this ){
            return true;
        }
        return false;
    }

    template<typename ProtoAllocator, typename Function>
    auto get_chunk_state(ProtoAllocator const& a, Function f){
        auto wraper_fn = [this, &a](auto&& args, auto& ch_handler){
            this->execute(execution::blocking.never, execution::relationship_t::chunk, a, std::move(args), ch_handler);
        };
        return chunk_state<decltype(wraper_fn), decltype(f), ProtoAllocator>(std::move(wraper_fn), std::move(f), a, [this]{ wait(); });
    }

    template<typename Blocking, typename ProtoAllocator, typename ArgsStruct, typename ChunkHandler>
    auto execute(Blocking, execution::relationship_t::chunk_t, ProtoAllocator const& a, ArgsStruct&& args, ChunkHandler& ch)
    {
        auto* inst = thread_private_state::instance();
        if constexpr( std::is_same_v<Blocking, execution::blocking_t::possibly_t> ){
            if( inst != nullptr && this == inst->m_pool ){
                ch.invoke_fn(std::move(args));
                return;
            }
        }else{
            
            auto tp = task(std::move(args),ch);
            m_queue.enqueue(tp);
            if( m_threads.size() != 0 )
                ch.add_task();
            // m_mutex.lock();
            // m_cond.notify_one();
            // m_mutex.unlock();
        }
    }

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void work_up(execution::outstanding_work_t::tracked_t) {
        m_work.fetch_add(1,std::memory_order_acquire);
    }

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE void work_down(execution::outstanding_work_t::tracked_t) {
        if( m_work.fetch_sub(1,std::memory_order_acquire) == 0 ){
            // m_mutex.lock();
            // m_cond.notify_all();
            // m_mutex.unlock();
        }
    }

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void work_up(execution::outstanding_work_t::untracked_t) const noexcept {}

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void work_down(execution::outstanding_work_t::untracked_t) const noexcept {}

    void swap( chunk_thread_pool& other ){
        std::swap(m_threads, other.m_threads);
        std::swap(m_count, other.m_count);
    }

    private:
        std::list<std::thread>                  m_threads;
        std::mutex                              m_mutex;
        moodycamel::ConcurrentQueue<task> m_queue;
        std::condition_variable                 m_cond;
        std::atomic<size_t>                     m_work{1};
        std::atomic<bool>                       m_token{false};
        size_t                                  m_count{std::thread::hardware_concurrency()};
    };

} // namespace boost::numeric::ublas::parallel::execution


#endif
