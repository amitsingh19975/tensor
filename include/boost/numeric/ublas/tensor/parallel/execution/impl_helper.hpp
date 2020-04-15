#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_IMPL_HELPER_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_IMPL_HELPER_HPP

#include "executor_concept.hpp"
#include <typeinfo>

namespace boost::numeric::ublas::parallel::execution{
    
    namespace meta_impl
    {
        
        template<typename... SupportableProperties>
        struct property_list;
        
        template<typename PropetyList>
        struct property_list_size;

        template<>
        struct property_list<>{};

        template<typename Head, typename... Tail>
        struct property_list<Head,Tail...>{};

        template<typename... SupportableProperties>
        struct property_list_size< property_list< SupportableProperties...> >{
            static constexpr auto const value = sizeof...(SupportableProperties);
        };

        template<typename PropetyList>
        inline static constexpr auto const property_list_size_v = property_list_size<PropetyList>::value;

        template<typename Head, typename... Tail>
        constexpr auto peek_front( property_list<Head,Tail...> ) -> Head;

        template<typename Head, typename... Tail>
        constexpr auto pop_front( property_list<Head,Tail...> ) -> property_list<Tail...>;

        template<typename, typename...> struct has_exact_property;

        template<typename Property>
        struct has_exact_property<Property>
            : std::false_type
        {};

        template<typename Property, typename Head, typename... Tail>
        struct has_exact_property<Property,Head,Tail...>
            : std::conditional_t< std::is_same_v<Property,Head>, std::true_type, has_exact_property<Property,Tail...> >
        {};


        template<typename Property, typename... SupportableProperties>
        inline static constexpr auto const has_exact_property_v = has_exact_property<Property,SupportableProperties...>::value;

        template<typename, typename...> struct has_convertible_property;

        template<typename Property, typename Head, typename... Tail>
        struct has_convertible_property<Property,Head,Tail...>
            : std::conditional_t< std::is_convertible_v<Property,Head>, std::true_type, has_convertible_property<Property,Tail...> >
        {};

        template<typename Property>
        struct has_convertible_property<Property>
            : std::false_type
        {};

        template<typename Property, typename... SupportableProperties>
        inline static constexpr auto const has_convertible_property_v = has_convertible_property<Property,SupportableProperties...>::value;

        template<typename, typename... > struct find_convertible_property;

        template<typename Property, typename Head, typename... Tail>
        struct find_convertible_property<Property,Head,Tail...>
            : std::conditional_t< std::is_convertible_v<Property,Head>, std::decay_t<Head>, find_convertible_property<Property,Tail...> >
        {};

        template<typename Property>
        struct find_convertible_property<Property>
        {};

        template<typename Property, typename... SupportableProperties>
        using find_convertible_property_t = typename find_convertible_property<Property,SupportableProperties...>::type;

        template<typename PropertyList, typename... SupportableProperties>
        struct has_exact_property_list;

        template<typename Head, typename... Tail, typename... SupportableProperties>
        struct has_exact_property_list< property_list<Head,Tail...>, SupportableProperties... >
            : std::integral_constant<bool, 
                has_exact_property_v<Head,SupportableProperties...> 
                && has_exact_property_list< property_list<Tail...>, SupportableProperties...>::value  >
        {};

        template<typename... SupportableProperties>
        struct has_exact_property_list< property_list<>, SupportableProperties... >
        {};

        template<typename Property, typename... SupportableProperties>
        inline static constexpr auto const has_exact_property_list_v = has_exact_property_list<Property,SupportableProperties...>::value;


        template<typename Executor, typename... SupportableProperties>
        struct is_valid_target;

        template<typename Executor>
        struct is_valid_target< Executor >
            : std::true_type
        {};

        template<typename Executor, typename Head, typename... Tail>
        struct is_valid_target< Executor, Head, Tail... >
            : std::integral_constant<bool,
                    ( !Head::is_requirable_concept || can_require_concept_v<Executor,Head> )
                &&  ( !Head::is_requirable || can_require_v<Executor,Head> )
                &&  ( !Head::is_preferable || can_prefer_v<Executor,Head> )
                &&  ( Head::is_preferable || Head::is_requirable || Head::is_preferable || can_query_v<Executor,Head> )
                &&  is_valid_target< Executor, Tail... >::value
            >
        {};

        template<typename Property, typename... SupportableProperties>
        inline static constexpr auto const is_valid_target_v = is_valid_target<Property,SupportableProperties...>::value;
        
        struct identity_property{
            static constexpr auto const is_requirable_concept   = true;
            static constexpr auto const is_requirable           = true;
            static constexpr auto const is_preferable           = true;

            template<typename Executor>
            static constexpr bool const static_query_v = true;

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr bool value() noexcept { return true; }
        };

        template<typename Property, typename... SuppoertableProperty>
        using conditional_property_t = std::conditional_t<
            has_exact_property_v<Property,SuppoertableProperty...>,
            Property,
            identity_property
        >;

    } // namespace meta_impl

    namespace detail{
        
        template<typename R, typename... Args>
        struct func_base{
            virtual ~func_base() {}
            virtual R call(Args&&...) = 0;
        };

        struct impl_interface{
            virtual ~impl_interface() {}
            virtual void destroy() noexcept = 0;
            virtual std::type_info const& target_type() const = 0;
            virtual void* target() = 0;
            virtual void const* target() const = 0;
            virtual void* require_concept(std::type_info const&, void const* p) const = 0;
            virtual void* query(std::type_info const&, void const* p) const = 0;  
        };
        
    } // namespace interface
    


} // namespace boost::numeric::ublas::parallel::execution





#endif