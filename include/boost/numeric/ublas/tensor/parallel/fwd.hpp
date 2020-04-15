#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_FWD_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_FWD_HPP

#include "detail/parallel_macro.hpp"

namespace boost::numeric::ublas::parallel{

    namespace detail{
        


    } // namespace detail

    namespace properties{
        // template<typename Entity, typename Property, typename = void> struct query_static_member_traits;
        // template<typename Entity, typename Property, typename = void> struct query_member_traits;
        // template<typename Entity, typename Property, typename = void> struct query_static_traits;
        // template<typename Entity, typename Property, typename = void> struct query_free_traits;

        // template<typename Entity, typename Property, typename = void> struct require_member_traits;
        // template<typename Entity, typename Property, typename = void> struct require_static_traits;
        // template<typename Entity, typename Property, typename = void> struct require_free_traits;

        // template<typename Entity, typename Property, typename = void> struct require_concept_member_traits;
        // template<typename Entity, typename Property, typename = void> struct require_concept_static_traits;
        // template<typename Entity, typename Property, typename = void> struct require_concept_free_traits;
        
        // template<typename Entity, typename Property, typename = void> struct prefer_free_traits;


    } // namespace properties
    
    struct default_deleter;
    struct dummy_deleter;
    struct stop_token;
    struct nostopstate;
    struct jthread;
    struct stop_source;

    struct producer_token;
    struct consumer_token;

    template<typename T, typename Traits> struct concurrent_queue;
    struct ConcurrentQueueTests;

    template<typename Callback> struct stop_callback;
    template<typename Partition> struct partition_base;
    template<typename Partition> struct adaptive_mode;
    template<typename T, depth_t MaxCapacity> struct range_vector;

    // template<typename Entity, typename Property, typename = void> struct is_applicable_property;
    template<typename Entity, typename Property> struct can_require_concept;
    template<typename Entity, typename Property> struct can_query;
    template<typename Entity, typename... Properties> struct can_require;
    template<typename Entity, typename... Properties> struct can_prefer;

    template<typename InnerProperty> struct prefer_only;


    namespace execution{
        
        struct context_t;
        struct executor_concept_t;
        struct oneway_t;
        struct bulk_oneway_t;
        struct blocking_t;
        struct blocking_adaptation_t;
        struct relationship_t;
        struct outstanding_work_t;
        struct bulk_guarantee_t;
        struct mapping_t;
        struct occupancy_t;
        class bad_executor;

        template<typename ProtoAllocator> struct allocator_t;
        template<class Executor> struct executor_shape;
        template<class Executor> struct executor_index;

    } // namespace execution
    
    

} // namespace boost::numeric::ublas::parallel

#endif