#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_ALLOCATOR_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_ALLOCATOR_HPP

#include "is_executor.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail::allocator{
        
        template<typename Derived>
        struct property_base
        {
        static constexpr bool is_requirable_concept = false;
        static constexpr bool is_requirable = true;
        static constexpr bool is_preferable = true;

        template<typename Executor, typename Type = decltype(Executor::query(*static_cast<Derived*>(0)))>
        static constexpr Type static_query_v = Executor::query(Derived());
        };

    } // namespace detail::allocator
    
    template<typename ProtoAllocator>
    struct allocator_t : detail::allocator::property_base<allocator_t<ProtoAllocator>>
    {
        constexpr explicit allocator_t(const ProtoAllocator& a) : alloc_(a) {}
        constexpr ProtoAllocator value() const { return alloc_; }
        ProtoAllocator alloc_;
    };

    template<>
    struct allocator_t<void> : detail::allocator::property_base<allocator_t<void>>
    {
        template<typename ProtoAllocator>
        allocator_t<ProtoAllocator> operator()(const ProtoAllocator& a) const
        {
            return allocator_t<ProtoAllocator>(a);
        }
    };

    constexpr allocator_t<void> allocator;

} // namespace boost::numeric::ublas::parallel::execution


namespace boost::numeric::ublas::parallel{
    
    template<class Entity, class T>
    struct is_applicable_property<Entity, execution::allocator_t<T>,
        std::enable_if_t< execution::is_executor_v<Entity>> > : std::true_type {};
    
} // namespace boost::numeric::ublas::parallel


#endif