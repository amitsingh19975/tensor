#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_PREFER_ONLY_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_PREFER_ONLY_HPP

#include "../fwd.hpp"
#include <type_traits>
#include <tuple>

namespace boost::numeric::ublas::parallel{
    
    namespace properties::prefer_only_impl{
        
        template<typename>
        struct type_check{
            using type = void;
        };

        template<typename InnerProperty, typename = void>
        struct prefer_only_base{};

        template<typename InnerProperty>
        struct prefer_only_base<
            InnerProperty,
            typename type_check< typename InnerProperty::polymorphic_query_result_type >::type
        >
        {
            using polymorphic_query_result_type = typename InnerProperty::polymorphic_query_result_type;
        };

        template<typename, typename InnerProperty>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto property_value( InnerProperty const& property )
            noexcept(noexcept(std::declval<InnerProperty>().value()))
            -> decltype(std::declval<InnerProperty>().value())
        {
            return property.value();
        }

    }

    template<typename InnerProperty>
    struct prefer_only : properties::prefer_only_impl::prefer_only_base<InnerProperty>{
        
        static constexpr auto const is_requirable_concept   = false;
        static constexpr auto const is_requirable           = false;
        static constexpr auto const is_preferable           = InnerProperty::is_preferable;

        template<typename Entity, typename Type = decltype(InnerProperty::template static_query_v<Entity>)>
        static constexpr Type const static_query_v = InnerProperty::template static_query_v<Entity>;

        prefer_only( InnerProperty const& p ) : property(p){} 

        template< typename Dummy = int >
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto value() const
            noexcept(noexcept(properties::prefer_only_impl::property_value<Dummy,InnerProperty>()))
            -> decltype(properties::prefer_only_impl::property_value<Dummy,InnerProperty>())
        {
            return properties::prefer_only_impl::property_value(property);
        }


        template<class Entity, class Property, class = typename std::enable_if<std::is_same<Property, prefer_only>::value>::type>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend auto prefer(Entity ex, const Property& p)
            noexcept(noexcept(prefer(std::move(ex), std::declval<const InnerProperty>())))
            -> decltype(prefer(std::move(ex), std::declval<const InnerProperty>()))
        {
            return ::boost::numeric::ublas::parallel::prefer(std::move(ex), p.property);
        }

        template<class Entity, class Property, class = typename std::enable_if<std::is_same<Property, prefer_only>::value>::type>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE friend constexpr auto query(const Entity& ex, const Property& p)
            noexcept(noexcept(query(ex, std::declval<const InnerProperty>())))
            -> decltype(query(ex, std::declval<const InnerProperty>()))
        {
            return ::boost::numeric::ublas::parallel::query(ex, p.property);
        }

        
        InnerProperty property;
    };

    template<class Entity, class InnerProperty>
    struct is_applicable_property<Entity, prefer_only<InnerProperty>>
    : is_applicable_property<Entity, InnerProperty> {};

} // namespace boost::numeric::ublas::parallel

#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PROPERTIES_PREFER_ONLY_HPP