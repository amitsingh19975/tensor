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

    template<typename Task>
    struct chunk_thread_pool;

    template<typename ExecuteFunction>
    struct chunk_handler{

        chunk_handler( ExecuteFunction ex)
            : m_efn(std::move(ex))
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

        template<typename Function>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()(Function&& fn) noexcept{
            m_efn(std::move(fn),*this);
            add_task();
        }

        template<typename Function>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto invoke_fn(Function&& fn) noexcept{
            fn();
            remove_task();
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
        ExecuteFunction         m_efn;
        std::atomic< ssize_t >  m_c1{0};
        std::atomic< ssize_t >  m_c2{0};
        std::atomic< bool >     m_flag{true};
        std::mutex              m_mu;
    };

    template<typename Allocator>
    struct task;

    template<typename Allocator>
    struct task{
        
        using allocator      = Allocator;
        using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<task>;
        using pointer        = task*;
        using fn_type        = std::function<void()>;
        template<typename Function>
        explicit task(Function fn, Allocator const& a)
            : m_fun(std::move(fn))
            , m_allocator(a)
        {}
        
        template<typename Function, typename ChunkHandler>
        explicit task(Function fn, Allocator const& a, ChunkHandler& ch)
            : m_fun(std::move(fn))
            , m_allocator(a)
            , remove_task([&ch]{ ch.remove_task();})
            , add_task([&ch]{ ch.add_task();})
        {
        }

        template<typename Function, typename ChunkHandler>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE static pointer create(Function fn, Allocator const& a, ChunkHandler& ch) noexcept{
            allocator_type all(a);
            pointer ptr = all.allocate(1);
            try{
                return pointer{ new(ptr) task([&, fn = std::move(fn)]() mutable { 
                        ch.invoke_fn(std::move(fn));
                    }, a, ch) 
                };
            }catch(...){
                all.deallocate(ptr,1);
                TENSOR_ASSERT(0,"boost::numeric::ublas::parallel::execution::task::create : unable to create task");
            }
            
        }

        template<typename Function>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE static pointer create(Function fn, Allocator const& a) noexcept{
            allocator_type all(a);
            pointer ptr = all.allocate(1);
            try{
                return pointer{ new(ptr) task(std::move(fn), a) };
            }catch(...){
                all.deallocate(ptr,1);
                TENSOR_ASSERT(0,"boost::numeric::ublas::parallel::execution::task::create : unable to create task");
            }
            
        }

        void destroy(){
            allocator_type all(m_allocator);
            pointer ptr{this};
            ptr->~task();
            all.deallocate(ptr,1);
        }

        void call(){
            pointer ptr{this};
            fn_type fn(std::move(m_fun));
            ptr->destroy();
            fn();
        }


        fn_type remove_task{ []{} };
        fn_type add_task{ []{} };

    private:
        fn_type     m_fun;
        Allocator   m_allocator;
    };

    template<>
    struct task<void>{

        using pointer        = task;
        using fn_type        = std::function<void()>;
        using allocator      = void*;

        task() = default;
        
        template<typename Function>
        explicit task(Function fn)
            : m_fun(std::move(fn))
        {
        }

        template<typename Function, typename ChunkHandler>
        explicit task(Function fn, ChunkHandler& ch)
            : m_fun(std::move(fn))
            , remove_task([&ch]{ ch.remove_task();})
            , add_task([&ch]{ ch.add_task();})
        {
        }

        task(task const& other)
            : m_fun(other.m_fun)
        {}
        task(task&& other)
            : m_fun(std::move(other.m_fun))
        {}
        task& operator=(task const& other){
            m_fun = other.m_fun;
            return *this;
        };
        task& operator=(task&& other){
            m_fun = std::move(other.m_fun);
            return *this;
        }


        template<typename Function, typename ChunkHandler>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE static pointer create(Function fn, ChunkHandler& ch) noexcept{
            return task([&, fn = std::move(fn)]() mutable { 
                ch.invoke_fn(std::move(fn));
            });
        }

        template<typename Function>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE static pointer create(Function fn) noexcept{
            return task(std::move(fn));
        }

        void destroy(){}

        void call(){
            m_fun();
        }

        fn_type remove_task{ []{} };
        fn_type add_task{ []{} };

    private:
        fn_type     m_fun;
    };

    template<typename T>
    struct MyTraits : public moodycamel::ConcurrentQueueDefaultTraits
    {
        static const size_t BLOCK_SIZE = 256;
    };

    template<typename Task>
    struct chunk_thread_pool{
        using queue_type = moodycamel::ConcurrentQueue<typename Task::pointer, MyTraits<Task> >;
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

            template<typename T = Interface,
                typename std::enable_if<std::is_same<T, execution::bulk_oneway_t>::value
                >::type* = nullptr>
            auto get_chunk_state() const
            {
                return m_pool->get_chunk_state(m_allocator);
            }

            template<typename Function, typename SharedFactory,
                typename std::enable_if<
                std::is_same<Function, Function>::value && std::is_same<Interface, execution::bulk_oneway_t>::value
                >::type* = nullptr>
            void bulk_execute(Function f, size_t n, SharedFactory sf) const
            {
             m_pool->bulk_execute(Blocking{}, Continuation{}, std::allocator<void>{}, std::move(f), n, std::move(sf));
            }

        private:
            friend struct chunk_thread_pool;
            
            executor_impl( chunk_thread_pool* pool, ProtoAllocator const& all )
                : m_pool(pool)
                , m_allocator(all)
            {
                m_pool->work_up(Work{});
            }
            
            executor_impl( chunk_thread_pool* pool )
                : m_pool(pool)
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
        execution::relationship_t::fork_t,
        execution::outstanding_work_t::untracked_t,
        typename Task::allocator
        >;

        explicit chunk_thread_pool(size_t count)
            : m_count(count)
        {
            m_queues.reserve(m_count);
            for(auto i = 0ul; i < m_count; ++i){
                m_queues.emplace_back(1000ul);
            }
            size_t i = 0;
            while(count--){
                m_threads.emplace_back([this,i]() mutable {attach(m_queues[i]);});
                ++i;
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE executor_type executor(){
            
            return executor_type{ this, typename Task::allocator{} };
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void try_steel( typename Task::pointer& fn, queue_type& curr_q ) noexcept{
            for(auto& q : m_queues){
                if( std::addressof(q) != std::addressof(curr_q) ){
                    if( q.size_approx() && q.try_dequeue(fn) ){
                        curr_q.enqueue(std::move(fn));
                        break;
                    }
                }
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void attach( queue_type& queue ){
            using ptr = typename Task::pointer;
            thread_private_state prt(this);
            
            ptr fn;
            while( true ){
                
                while( !(m_token.load(std::memory_order_acquire) || m_work.load(std::memory_order_acquire) == 0 || queue.size_approx() ) ) {
                    try_steel(fn,queue);
                    YieldProcessor();
                    MemoryBarrier();
                }

                if( m_token.load(std::memory_order_acquire) || ( m_work.load(std::memory_order_acquire) == 0 && !queue.size_approx()) ){ return; };
                
                if( BOOST_UBLAS_TENSOR_LIKLY( queue.try_dequeue(fn), true) ){
                    invoke(fn);
                }

                if( prt.m_queue.size_approx() ){
                    if( prt.m_queue.try_dequeue(fn) ){
                        if constexpr( std::is_pointer_v<ptr> ){
                            fn->add_task();
                        }else{
                            fn.add_task();
                        }
                        queue.enqueue(std::move(fn));
                    }
                }

            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void stop(){
            m_token.store(true);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void wait()
        {
            std::list<std::thread> threads(std::move(m_threads));
            if (BOOST_UBLAS_TENSOR_LIKLY ( !threads.empty(), true ) )
            {
                m_work.fetch_sub(1,std::memory_order_acquire);
                for (auto& t : threads){
                    t.join();
                }
                    
            }
            m_threads.clear();
        }

        ~chunk_thread_pool(){
            stop();
            wait();
        }
    
    private:

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

            chunk_thread_pool*      m_pool;
            thread_private_state*   m_prev_state{instance()};
            queue_type              m_queue;
            std::atomic<size_t>     m_queue_size{0};

        };
    
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void invoke(Task* t) noexcept
        {
            t->call();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void invoke(task<void>& t) noexcept
        {
            t.call();
        }
    
    template<typename ProtoAllocator>
    auto get_chunk_state(ProtoAllocator const& a){
        auto wraper_fn = [this, &a](auto&& fn, auto& ch_handler){
            this->execute(execution::blocking.never, execution::relationship_t::chunk, a, std::move(fn), ch_handler);
        };
        return chunk_handler<decltype(wraper_fn)>( std::move(wraper_fn));
    }

    template<typename Blocking, typename ProtoAllocator, typename Function, typename ChunkHandler>
    auto execute(Blocking, execution::relationship_t::chunk_t, ProtoAllocator const& a, Function&& fn, ChunkHandler& ch)
    {
        auto old = m_thread_idx.load(std::memory_order_acquire);
        auto& queue = m_queues[old];
        m_thread_idx.compare_exchange_strong( old, (old + 1) % m_count, std::memory_order_relaxed );

        if constexpr( std::is_same_v<Task, task<void> >){
            auto tp = Task::create(std::move(fn),ch);
            while( !queue.try_enqueue(std::move(tp)) );
        }else{
            auto tp = Task::create(std::move(fn), a, ch);
            while( !queue.try_enqueue(std::move(tp)) );
        }
        m_queue_size.fetch_add(1,std::memory_order_relaxed);
    }

    template<typename Function, typename SharedFactory>
    struct bulk_state{
        
        bulk_state(Function f, SharedFactory ss)
            : m_fn(std::move(f))
            , m_ss(ss())
        {
        }
        
        auto operator()(size_t i){ m_fn(i,m_ss);  }

    // private:
        Function m_fn;
        decltype(std::declval<SharedFactory>()()) m_ss;
    };

    template<typename Blocking, typename Continuation, typename ProtoAllocator, typename Function, typename SharedFactory>
    void bulk_execute(Blocking, Continuation, ProtoAllocator const& a, Function f, size_t n, SharedFactory sf){
        if constexpr( Blocking{} != execution::blocking_t::always ){
            
            typename std::allocator_traits<ProtoAllocator>::template rebind_alloc<char> alloc2(a);
            auto shared_state = std::allocate_shared< bulk_state<Function,SharedFactory> >(alloc2, std::move(f), std::move(sf));
            auto* inst = thread_private_state::instance();
            n = shared_state->m_ss.size();


            if constexpr( std::is_same_v<Continuation, execution::relationship_t::continuation_t> ){
                if( inst != nullptr && this == inst->m_pool ){
                    for( auto i = size_t(0); i < n; ++i ){
                        auto fn = [shared_state, i]() mutable { (*shared_state)(i); };
                        if constexpr( std::is_same_v<Task, task<void> >){
                            auto tp = Task::create(std::move(fn));
                            inst->m_queue.enqueue(std::move(tp));
                        }else{
                            auto tp = Task::create(std::move(fn), a);
                            inst->m_queue.enqueue(std::move(tp));
                        }
                        inst->m_queue_size.fetch_add(1,std::memory_order_relaxed);
                    }
                }
            }else{
                for( auto i = size_t(0); i < n; ++i ){
                    auto fn = [shared_state, i]() mutable { (*shared_state)(i); };
                    
                    auto old = m_thread_idx.load(std::memory_order_acquire);
                    auto& queue = m_queues[old];
                    m_thread_idx.compare_exchange_strong( old, (old + 1) % m_count, std::memory_order_relaxed );

                    if constexpr( std::is_same_v<Task, task<void> >){
                        auto tp = Task::create(std::move(fn));
                        queue.enqueue(std::move(tp));
                    }else{
                        auto tp = Task::create(std::move(fn), a);
                        queue.enqueue(std::move(tp));
                    }
                    m_queue_size.fetch_add(1,std::memory_order_relaxed);
                }
            }
        }else{
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            auto wrapped_f = [f = std::move(f), p = std::move(promise)](std::size_t n, auto& s) mutable { f(n, s);};
            this->bulk_execute(execution::blocking.never, Continuation{}, a, std::move(wrapped_f), n, std::move(sf));
            future.wait();
        }
        
    }

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void work_up(execution::outstanding_work_t::tracked_t) {
        m_work.fetch_add(1,std::memory_order_acquire);
    }

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE void work_down(execution::outstanding_work_t::tracked_t) {
        m_work.fetch_sub(1,std::memory_order_acquire);
    }

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void work_up(execution::outstanding_work_t::untracked_t) const noexcept {}

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void work_down(execution::outstanding_work_t::untracked_t) const noexcept {}

    void swap( chunk_thread_pool& other ){
        std::swap(m_threads, other.m_threads);
        std::swap(m_count, other.m_count);
    }

    private:
        std::list<std::thread>      m_threads;
        std::mutex                  m_mutex;
        std::vector<queue_type>     m_queues;
        std::condition_variable     m_cond;
        std::atomic<size_t>         m_work{1};
        std::atomic<bool>           m_token{false};
        std::atomic<size_t>         m_queue_size{0};
        std::atomic<size_t>         m_thread_idx{0};
        size_t                      m_count{std::thread::hardware_concurrency()};
    };

} // namespace boost::numeric::ublas::parallel::execution


#endif
