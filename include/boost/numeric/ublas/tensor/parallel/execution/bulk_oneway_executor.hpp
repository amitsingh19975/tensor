#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BULK_ONEWAY_EXECUTOR_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BULK_ONEWAY_EXECUTOR_HPP

#include "is_inteface_property.hpp"
#include "impl_helper.hpp"
#include "bad_executor.hpp"
#include <memory>
#include <atomic>

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail::bulk_oneway_executor
    {
        using namespace boost::numeric::ublas::parallel::execution::meta_impl;
        using namespace boost::numeric::ublas::parallel::execution::detail;

        template<typename Function, typename R, typename... Args>
        struct multi_use_func : func_base< R, Args... >{
            
            multi_use_func( Function fn )
                : m_func(std::move(fn))
            {}

            R call(Args&&... args){
                if constexpr(std::is_same_v< R, void >){
                    m_func(std::forward<Args>(args)...);
                }else{
                    return m_func(std::forward<Args>(args)...);
                }
            }

        private:
            Function m_func;
        }; 

        using shared_factory_base = func_base<void>;
        template<typename S> using shared_factory = multi_use_func<S, void>;
        
        using bulk_func_base = func_base<void,size_t,std::shared_ptr<void>&>;
        template<typename F> using bulk_func = multi_use_func<F, void, size_t, std::shared_ptr<void>&>;

        struct impl_base : impl_interface{
            virtual ~impl_base(){}
            virtual impl_base* clone() const noexcept = 0;
            virtual bool equals(const impl_base* e) const noexcept = 0;
            virtual void bulk_execute(std::unique_ptr<bulk_func_base> f, size_t n, std::shared_ptr<shared_factory_base> sf) = 0;
            virtual impl_base* require(std::type_info const&, void const* p) const = 0;
            virtual impl_base* prefer(std::type_info const&, void const* p) const = 0;
        };

        template<typename Executor, typename... SupportableProperties>
        struct impl : impl_base {
            explicit impl(Executor ex) 
                : m_executor(std::move(ex)) 
            {}

            virtual impl_base* clone() const noexcept{
                auto e = const_cast<impl*>(this);
                ++(e->m_ref_count);
                return e;
            }

            virtual void destroy() noexcept{

                if( --m_ref_count == 0 ){
                    delete this;
                }
            }

            virtual void bulk_execute(std::unique_ptr<bulk_func_base> f, size_t n, std::shared_ptr<shared_factory_base> sf){
                m_executor.bulk_execute([fn = std::move(f)](std::size_t i, auto s) mutable { fn.release()->call(i,s); }, 
                        n,
                        [sfn = std::move(sf)]() mutable{ return sfn->call(); }
                    );
            }

            virtual std::type_info const& target_type() const{
                return typeid(m_executor);
            }

            virtual void* target(){
                return static_cast<void*>(&m_executor);
            };

            virtual void const* target() const{
                return static_cast<void const*>(&m_executor);
            };

            virtual bool equals(const impl_base* e) const noexcept{
                if( this == e ){
                    return true;
                }
                if ( this->target_type() != e->target_type() ){
                    return false;
                }
                return m_executor == *static_cast<Executor const*>(e->target());
            };

            virtual impl_base* require(std::type_info const& t, void const* p) const{
                return this->require_helper(property_list<SupportableProperties...>{}, t, p);
            };

            virtual void* require_concept(std::type_info const& t, void const* p) const{
                return this->require_concept_helper(property_list<SupportableProperties...>{}, t, p);
            };
            virtual impl_base* prefer(std::type_info const& t, void const* p) const{
                return this->prefer_helper(property_list<SupportableProperties...>{}, t, p);
            };
            virtual void* query(std::type_info const& t, void const* p) const{
                return this->query_helper(property_list<SupportableProperties...>{}, t, p);
            };  

        private:

            template<typename... Properties>
            impl_base* require_helper(property_list<Properties...>, std::type_info const& t, void const* p ){
                if constexpr( sizeof...(Properties) == 0 ){
                    TENSOR_ASSERT(0,"boost::numeric::ublas::parallel::execution::detail::bulk_oneway_executor::impl_base::require_helper(): property_list is empty");
                    return nullptr;
                }else{
                    using List = property_list<Properties...>;
                    using Head = decltype( peek_front(List{}) );
                    using NewList = decltype( pop_front(List{}) );
                    
                    if constexpr ( Head::is_requirable || !is_interface_property_v<Head> ){
                        if( t == typeid(Head) ){
                            using executor_type = decltype( ::boost::numeric::ublas::parallel::require(m_executor, *static_cast<Head const*>(p)) );
                            return new impl<executor_type,SupportableProperties...>(::boost::numeric::ublas::parallel::require(m_executor, *static_cast<Head const*>(p)));
                        }else{
                            return require_helper(NewList{},t,p);
                        }
                    }else if constexpr( !Head::is_requirable || is_interface_property_v<Head> ) {
                        return require_helper(NewList{},t,p);
                    }else{
                        require_helper(property_list<>{},t,p);
                    }
                }
            }

            template<typename... Properties>
            impl_base* require_conecpt_helper(property_list<Properties...>, std::type_info const& t, void const* p ){
                if constexpr( sizeof...(Properties) == 0 ){
                    TENSOR_ASSERT(0,"boost::numeric::ublas::parallel::execution::detail::bulk_oneway_executor::impl_base::require_conecpt_helper(): property_list is empty");
                    return nullptr;
                }else{
                    using List = property_list<Properties...>;
                    using Head = decltype( peek_front(List{}) );
                    using NewList = decltype( pop_front(List{}) );
                    
                    if constexpr ( is_interface_property_v<Head> ){
                        if( t == typeid(Head) ){
                            using outter_pet = typename Head::template polymorphic_executor_type<>;
                            using inner_pet = typename Head::template polymorphic_executor_type<Properties...>;
                            return new outter_pet( inner_pet(::boost::numeric::ublas::parallel::require_concept(m_executor, *static_cast<Head const*>(p))) );
                        }else{
                            return require_conecpt_helper(NewList{},t,p);
                        }
                    }else if constexpr( !is_interface_property_v<Head> ) {
                        return require_conecpt_helper(NewList{},t,p);
                    }else{
                        require_conecpt_helper(property_list<>{},t,p);
                    }
                }
            }
            
            template<typename... Properties>
            impl_base* prefer_helper(property_list<Properties...>, std::type_info const& t, void const* p ){
                if constexpr( sizeof...(Properties) == 0 ){
                    return clone();
                }else{
                    using List = property_list<Properties...>;
                    using Head = decltype( peek_front(List{}) );
                    using NewList = decltype( pop_front(List{}) );
                    
                    if constexpr ( Head::is_preferable || !is_interface_property_v<Head> ){
                        if( t == typeid(Head) ){
                            using executor_type = decltype( ::boost::numeric::ublas::parallel::prefer(m_executor, *static_cast<Head const*>(p)) );
                            return new impl<executor_type,SupportableProperties...>(::boost::numeric::ublas::parallel::prefer(m_executor, *static_cast<Head const*>(p)));
                        }else{
                            return prefer_helper(NewList{},t,p);
                        }
                    }else if constexpr( !Head::is_preferable || is_interface_property_v<Head> ) {
                        return prefer_helper(NewList{},t,p);
                    }else{
                        prefer_helper(property_list<>{},t,p);
                    }
                }
            }

            template<typename... Properties>
            impl_base* query_helper(property_list<Properties...>, std::type_info const& t, void const* p ){
                if constexpr( sizeof...(Properties) == 0 ){
                    return nullptr;
                }else{
                    using List = property_list<Properties...>;
                    using Head = decltype( peek_front(List{}) );
                    using NewList = decltype( pop_front(List{}) );
                    
                    if constexpr ( can_query_v<Executor,Head> ){
                        if( t == typeid(Head) ){
                            using pqrt = typename Head::template polymorphic_query_result_type<>;
                            return new std::tuple<pqrt>(::boost::numeric::ublas::parallel::query(m_executor, *static_cast<Head const*>(p)));
                        }else{
                            return query_helper(NewList{},t,p);
                        }
                    }else{
                        return query_helper(NewList{},t,p);
                    }
                }
            }

        protected:
            std::atomic_uint64_t m_ref_count{1};
            Executor m_executor;
        };

    } // namespace detail::bulk_oneway_executor
    
    template<typename... SupportableProperties>
    struct bulk_oneway_t::polymorphic_executor_type{
        polymorphic_executor_type() noexcept = default;
        polymorphic_executor_type(std::nullptr_t){}
        polymorphic_executor_type(polymorphic_executor_type const& e)
            : m_impl(e.m_impl ? e.m_impl->clone() : nullptr)
        {}
        polymorphic_executor_type(polymorphic_executor_type&& e)
            : m_impl(e.m_impl)
        {
            e.m_impl = nullptr;
        }

        template<typename Executor>
        polymorphic_executor_type( Executor e, 
            std::enable_if_t< detail::bulk_oneway_executor::is_valid_target_v<
                std::enable_if_t< !std::is_same_v<Executor,polymorphic_executor_type>, Executor >,
                SupportableProperties...
            > >* = 0
        )
        {
            auto new_e = require_concept(std::move(e),oneway);
            m_impl = new detail::bulk_oneway_executor::impl<decltype(new_e),SupportableProperties...>(std::move(new_e));
        }

        template<typename... OtherSupportableProperties>
        polymorphic_executor_type( polymorphic_executor_type<OtherSupportableProperties...> const& e, 
            std::enable_if_t< detail::bulk_oneway_executor::has_exact_property_list_v<
                detail::bulk_oneway_executor::property_list<SupportableProperties...>,
                OtherSupportableProperties...
            > >* = 0
        )
            : m_impl(e.m_impl ? e.m_impl->clone() : nullptr)
        {
        }

        template<typename... OtherSupportableProperties>
        polymorphic_executor_type( polymorphic_executor_type<OtherSupportableProperties...> const& e, 
            std::enable_if_t< !detail::bulk_oneway_executor::has_exact_property_list_v<
                detail::bulk_oneway_executor::property_list<SupportableProperties...>,
                OtherSupportableProperties...
            > >* = 0
        ) = delete;

        polymorphic_executor_type& operator=( polymorphic_executor_type const& other ) noexcept{
            if( m_impl ){
                m_impl->destroy();
            }
            m_impl = other.m_impl ? other.m_impl->clone() : nullptr;
            return *this;
        }

        polymorphic_executor_type& operator=( polymorphic_executor_type&& other ) noexcept{
            if( this != &other ){
                if( m_impl ){
                    m_impl->destroy();
                }
                m_impl = other.m_impl;
                other.m_impl = nullptr;
            }
            return *this;
        }

        polymorphic_executor_type& operator=( std::nullptr_t ) noexcept{
            if( m_impl ){
                m_impl->destroy();
            }
            m_impl = nullptr;
            return *this;
        }

        template<typename Executor>
        polymorphic_executor_type& operator=( Executor e ){
            return this->operator=(polymorphic_executor_type(e));
        }

        template<typename Executor>
        void assign( Executor e ){
            this->operator=(polymorphic_executor_type(e));
        }

        ~polymorphic_executor_type(){
            if( m_impl ) m_impl->destroy();
        }

        void swap(polymorphic_executor_type& rhs){
            std::swap(this->m_impl,rhs.m_impl);
        }

        static constexpr bulk_oneway_t query(executor_concept_t){ return {}; }

        template<typename Property, 
            typename = std::enable_if_t<
                detail::bulk_oneway_executor::find_convertible_property_t<Property,SupportableProperties...>::is_requirable_concept
                && is_interface_property_v<Property>
            > >
        typename Property::template polymorphic_executor_type<SupportableProperties...> require_concept(Property const& p) const{
            
            if( !m_impl ) throw bad_executor{};
            using ptr_type = typename Property::template polymorphic_executor_type<>;
            detail::bulk_oneway_executor::find_convertible_property_t<Property,SupportableProperties...> p1(p);
            return std::unique_ptr<ptr_type>{
                static_cast<ptr_type*>(
                    m_impl->require_concept(typeid(p1), static_cast<void const*>(&p1)) 
                    )->template downcast<SupportableProperties...>()
            };
        }

        template<typename Property, 
            typename = std::enable_if_t<
                detail::bulk_oneway_executor::find_convertible_property_t<Property,SupportableProperties...>::is_requirable
                && !is_interface_property_v<Property>
            > >
        polymorphic_executor_type require(Property const& p) const{
            detail::bulk_oneway_executor::find_convertible_property_t<Property,SupportableProperties...> p1(p);
            return m_impl ? m_impl->require(typeid(p1), static_cast<void const*>(&p1)) : throw bad_executor{};
        }

        template<typename Property, 
            typename = std::enable_if_t<
                detail::bulk_oneway_executor::find_convertible_property_t<Property,SupportableProperties...>::is_preferable >
            >
        friend polymorphic_executor_type prefer(polymorphic_executor_type const& e, Property const& p){
            detail::bulk_oneway_executor::find_convertible_property_t<Property,SupportableProperties...> p1(p);
            return e.get_impl() ? e.get_impl()->prefer(typeid(p1), static_cast<void const*>(&p1)) : throw bad_executor{};
        }

        template<typename Property>
        auto query( Property const& p )
            -> typename detail::bulk_oneway_executor::find_convertible_property_t<Property, SupportableProperties...>::polymorphic_query_result_type
        {
            detail::bulk_oneway_executor::find_convertible_property_t<Property,SupportableProperties...> p1(p);
            using result_type = typename decltype(p1)::polymorphic_query_result_type;
            using tuple_type = std::tuple<result_type>;
            if( !m_impl ) throw bad_executor{};
            std::unique_ptr<tuple_type> res( static_cast<tuple_type*>(m_impl->query(typeid(p1), static_cast<void const*>(&p1) )) );

            if constexpr(
                detail::bulk_oneway_executor::find_convertible_property_t<Property,SupportableProperties...>::is_requirable_concept
                || detail::bulk_oneway_executor::find_convertible_property_t<Property,SupportableProperties...>::is_requirable
                || detail::bulk_oneway_executor::find_convertible_property_t<Property,SupportableProperties...>::is_preferable
            ){
                return res ? std::get<0>(*res) : result_type{};
            }else{
                return std::get<0>(*res);
            }
        }

        template<typename Function, typename SharedFactory>
        void bulk_execute( Function f, std::size_t n, SharedFactory sf ) const{
            auto f_wrap = [f = std::move(f)](std::size_t i, std::shared_ptr<void>& ss) mutable{
                f(i, *std::static_pointer_cast<decltype(sf())>(ss));
            };

            auto sf_wrap = [sf = std::move(sf)]() mutable{
                return std::make_shared<decltype(sf())>(sf());
            };

            std::unique_ptr<detail::bulk_oneway_executor::bulk_func_base> fp(new detail::bulk_oneway_executor::bulk_func<decltype(f_wrap)>(std::move(f_wrap)));
            std::shared_ptr<detail::bulk_oneway_executor::shared_factory_base> sfp(new detail::bulk_oneway_executor::shared_factory<decltype(sf_wrap)>(std::move(sf_wrap)));
            m_impl ? m_impl->bulk_execute(std::move(fp), n, std::move(sfp)) : throw bad_executor();
        }

        constexpr explicit operator bool()const noexcept{
            return !!m_impl;
        }

        template<class Executor> Executor* target() noexcept{
            return m_impl ? static_cast<Executor*>(m_impl->target()) : nullptr;
        }

        template<class Executor> const Executor* target() const noexcept{
            return m_impl ? static_cast<Executor*>(m_impl->target()) : nullptr;
        }

        friend bool operator==(const polymorphic_executor_type& a, const polymorphic_executor_type& b) noexcept{
            if (!a.get_impl() && !b.get_impl())
            return true;
            if (a.get_impl() && b.get_impl())
            return a.get_impl()->equals(b.get_impl());
            return false;
        }

        friend bool operator==(const polymorphic_executor_type& e, std::nullptr_t) noexcept{
            return !e;
        }

        friend bool operator==(std::nullptr_t, const polymorphic_executor_type& e) noexcept{
            return !e;
        }

        friend bool operator!=(const polymorphic_executor_type& a, const polymorphic_executor_type& b) noexcept{
            return !(a == b);
        }

        friend bool operator!=(const polymorphic_executor_type& e, std::nullptr_t) noexcept{
            return !!e;
        }

        friend bool operator!=(std::nullptr_t, const polymorphic_executor_type& e) noexcept{
            return !!e;
        }

        friend void swap(polymorphic_executor_type& a, polymorphic_executor_type& b) noexcept{
            a.swap(b);
        }

        template<typename... OtherSupportableProperties>
        polymorphic_executor_type<OtherSupportableProperties...> downcast() const{
            return polymorphic_executor_type<OtherSupportableProperties...>(m_impl->clone());
        }
    
    private:
        template<typename...> friend struct bulk_oneway_t::polymorphic_executor_type;
        polymorphic_executor_type(detail::bulk_oneway_executor::impl_base* i) noexcept : m_impl(i){}
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE detail::bulk_oneway_executor::impl_base const* get_impl() const noexcept { return m_impl; }
        
    private:
        detail::bulk_oneway_executor::impl_base* m_impl{nullptr};
    };

} // namespace boost::numeric::ublas::parallel::execution





#endif