#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_STATIC_THREAD_POOL_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_STATIC_THREAD_POOL_HPP

#include <thread>
#include "blocking.hpp"
#include "../queue.hpp"
#include <list>
#include <new>
#include <condition_variable>
#include <mutex>
#include <future>
#include <boost/core/demangle.hpp>
#include <iostream>

namespace boost::numeric::ublas::parallel::execution{

    struct static_thread_pool{
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

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr static_thread_pool& query(execution::context_t){ return *m_pool; }

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

            template<typename Function,
                typename std::enable_if<
                std::is_same<Function, Function>::value && std::is_same<Interface, execution::oneway_t>::value
                >::type* = nullptr>
            auto execute(Function f) const
            {
                if constexpr( Continuation{} == execution::relationship.chunk )
                    return m_pool->execute(execution::blocking.always, execution::relationship.chunk, m_allocator);
                else
                    m_pool->execute(Blocking{}, Continuation{}, m_allocator, std::move(f));
            }

            template<typename Function, typename SharedFactory,
                typename std::enable_if<
                std::is_same<Function, Function>::value && std::is_same<Interface, execution::bulk_oneway_t>::value
                >::type* = nullptr>
            void bulk_execute(Function f, size_t n, SharedFactory sf) const
            {
             m_pool->bulk_execute(Blocking{}, Continuation{}, m_allocator, std::move(f), n, std::move(sf));
            }

        private:
            friend struct static_thread_pool;
            
            executor_impl( static_thread_pool* pool, ProtoAllocator const& all )
                : m_pool(pool)
                , m_allocator(all)
            {
                m_pool->work_up(Work{});
            }

        private:
            friend struct static_thread_pool;
            static_thread_pool*     m_pool;
            ProtoAllocator          m_allocator;
        };

    public:

        using executor_type = executor_impl<
            execution::bulk_oneway_t,
            execution::blocking_t::possibly_t,
            execution::outstanding_work_t::untracked_t,
            std::allocator<void>
        >;

        explicit static_thread_pool(size_t count){
            while(count--){
                m_threads.emplace_back([this](){attach();});
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE executor_type executor(){
            return executor_type{ this, std::allocator<void>{} };
        } 

        void attach(){
            thread_private_state prt(this);
            for( std::unique_lock<std::mutex> lck(m_mutex);; ){
            // while(!m_token){
                m_cond.wait(lck,[this]{
                    return m_token || m_work == 0 || m_head;
                });
                if( m_token || ( m_work == 0 && !m_head ) ){ return; };
                std::cout<<m_token<<' '<<m_head<<'\n';
                func_base* fn = m_head.release();
                m_head = std::move( fn->next );
                m_tail = m_head ? m_tail : &m_head;

                lck.unlock();
                fn->call();
                lck.lock();

                if (prt.m_head)
                {
                    *m_tail = std::move(prt.m_head);
                    m_tail = prt.m_tail;
                    prt.m_tail = &prt.m_head;
                }
            }
        }

        void stop(){
            std::lock_guard<std::mutex> lck(m_mutex);
            m_token = true;
            m_cond.notify_all();
        }

        void wait()
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            std::list<std::thread> threads(std::move(m_threads));
            if (!threads.empty())
            {
                --m_work;
                m_cond.notify_all();
                lock.unlock();
                for (auto& t : threads){
                    // t.request_stop();
                    t.join();
                }
                    
            }
        }

        ~static_thread_pool(){
            stop();
            wait();
        }
    
    private:
        template<class Function>
        static void invoke(Function& f) noexcept
        {
            f();
        }
        struct func_base{
            

            virtual ~func_base(){}

            struct deleter{
                void operator()(func_base* fp){ fp->destroy(); }
            };

            virtual void destroy() = 0;
            virtual void call() = 0;

            using pointer = std::unique_ptr<func_base,deleter>;
            pointer next;
        };

        template<typename Fn, typename Allocator>
        struct func : func_base{
            
            using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<func>;

            explicit func( Fn f, Allocator const& all )
                : m_fn(std::move(f))
                , m_allocator(all)
            {}

            static func_base::pointer create( Fn f, Allocator const& all ){
                allocator_type allocator(all);
                func* fp = allocator.allocate(1);
                try{
                   func* p = new(fp) func(std::move(f),all);
                   return func_base::pointer{p}; 
                }catch(...){
                    allocator.deallocate(fp,1);
                    throw;
                }
            }

            virtual void destroy(){
                func* p = this;
                p->~func();
                m_allocator.deallocate(p,1);
            }

            virtual void call() {
                func_base::pointer fp(this);
                Fn f(std::move(m_fn));
                fp.reset();
                static_thread_pool::invoke(f);
            }
        private:
            Fn m_fn;
            allocator_type m_allocator;
            std::mutex m_mutex;
        };
    
    struct thread_private_state{

        explicit thread_private_state( static_thread_pool* pool )
            : m_pool(pool)
        {
            instance() = m_prev_state;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE static thread_private_state*& instance(){
            static thread_local thread_private_state* state;
            return state;
        }

        static_thread_pool*                     m_pool;
        thread_private_state*                   m_prev_state{instance()};
        func_base::pointer                      m_head;
        func_base::pointer*                     m_tail{&m_head};
    };

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE bool running_in_this_thread() const noexcept{
        auto* inst = thread_private_state::instance();
        if( inst != nullptr && inst->m_pool == this ){
            return true;
        }
        return false;
    }

    template<typename Blocking, typename Continuation, typename ProtoAllocator, typename Function>
    void execute(Blocking, Continuation, ProtoAllocator const& a, Function f){
        
        auto* inst = thread_private_state::instance();
        if constexpr( Blocking{} != execution::blocking_t::always ){
            if constexpr( std::is_same_v<Blocking, execution::blocking_t::possibly_t> ){
                if( inst != nullptr && this == inst->m_pool ){
                    static_thread_pool::invoke(f);
                    return;
                }
            }else{

                auto tp = func<Function,ProtoAllocator>::create(std::move(f),a);

                if constexpr( std::is_same_v<Continuation, execution::relationship_t::continuation_t> ){
                    if( inst != nullptr && this == inst->m_pool ){
                        *inst->m_tail = std::move(tp);
                        inst->m_tail = &(*inst->m_tail)->next;
                        return;
                    }
                }else{
                    std::lock_guard<std::mutex> lck(m_mutex);
                    *m_tail = std::move(tp);
                    m_tail = &(*m_tail)->next;
                    m_cond.notify_all();
                }
            }
        }else{
            if( inst != nullptr && this == inst->m_pool ){
                static_thread_pool::invoke(f);
            }else{
                std::promise<void> promise;
                std::future<void> future = promise.get_future();

                this->execute(execution::blocking.never, Continuation{}, a, [fn = std::move(f), p = std::move(promise)]() mutable { fn();});

                future.wait();
            }
        }

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

            func_base::pointer head;
            func_base::pointer* tail{&head};

            for( auto i = size_t(0); i < n; ++i ){
                auto fn = [shared_state, i]() mutable { (*shared_state)(i); };
                *tail = func<decltype(fn), ProtoAllocator>::create(std::move(fn), a);
                tail = &(*tail)->next;
            }

            if constexpr( std::is_same_v<Continuation, execution::relationship_t::continuation_t> ){
                if( inst != nullptr && this == inst->m_pool ){
                    *inst->m_tail = std::move(head);
                    inst->m_tail = tail;
                }
            }else{
                std::lock_guard<std::mutex> lck(m_mutex);
                *m_tail = std::move(head);
                m_tail = tail;
                m_cond.notify_all();
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
        ++m_work;
    }

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE void work_down(execution::outstanding_work_t::tracked_t) {
        if( --m_work == 0 ){
            std::lock_guard<std::mutex> lck(m_mutex);
            m_cond.notify_all();
        }
    }

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void work_up(execution::outstanding_work_t::untracked_t) const noexcept {}

    BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void work_down(execution::outstanding_work_t::untracked_t) const noexcept {}

    private:
        std::list<std::thread>                  m_threads;
        std::mutex                              m_mutex;
        func_base::pointer                      m_head;
        func_base::pointer*                     m_tail{&m_head};
        std::condition_variable                 m_cond;
        std::atomic<size_t>                     m_work{1};
        bool                                    m_token{false};
    };

} // namespace boost::numeric::ublas::parallel::execution


#endif