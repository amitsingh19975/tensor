#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_ENUMERATION_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_ENUMERATION_HPP

#include "../properties/can_query.hpp"
#include "is_executor.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    namespace detail{
        
        template<typename Enumeration, int N, int Value>
        struct enumerator_impl;
        
        template<typename Enumeration, int N, typename Enumerator>
        struct default_enumerator{
            
            template<typename Executor>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr auto static_query()
                -> decltype( Executor::query(*static_cast<Enumerator*>(0)) )
            {
                return Executor::query(Enumerator{});
            }

            template<typename Executor, int I = 1>
            struct use_default_static_query :
                std::conditional_t<
                    can_query<Executor, enumerator_impl<Enumeration, N, I> >::value,
                    std::false_type,
                    std::conditional_t<
                        I + 1 < N,
                        use_default_static_query<Executor, I + 1>,
                        std::true_type
                    >
                >
            {
            };

            template<typename Executor>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr auto static_query()
                -> std::enable_if_t<
                !properties::query_member_traits<Executor, Enumerator>::is_valid
                    && use_default_static_query<Executor>::value,
                Enumerator
                >
            {
                return Enumerator();
            }
        };

        template<typename Enumeration, int N, typename Enumerator>
        struct non_default_enumerator
        {
            template<typename Executor>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr auto static_query()
                -> decltype(Executor::query(*static_cast<Enumerator*>(0)))
            {
                return Executor::query(Enumerator());
            }
        };

        template<typename Enumeration, int N, int Value>
        struct enumerator_impl :
            std::conditional_t<
                Value == 0,
                default_enumerator<Enumeration, N, enumerator_impl<Enumeration, N, Value> >,
                non_default_enumerator<Enumeration, N, enumerator_impl<Enumeration, N, Value> >
            >
            {
            static constexpr bool is_requirable_concept = false;
            static constexpr bool is_requirable = true;
            static constexpr bool is_preferable = true;

            using base_type_helper =
                std::conditional_t<
                Value == 0,
                default_enumerator<Enumeration, N, enumerator_impl<Enumeration, N, Value> >,
                non_default_enumerator<Enumeration, N, enumerator_impl<Enumeration, N, Value> >
                >;

            using polymorphic_query_result_type = Enumeration;

            template<typename Executor,
                typename T = std::enable_if_t<
                (base_type_helper::template static_query<Executor>(), true),
                decltype(base_type_helper::template static_query<Executor>())
                >>
            inline static constexpr T static_query_v = base_type_helper::template static_query<Executor>();

            BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr Enumeration value()
            {
                return Enumeration(Value);
            }
        };

        template<typename Enumeration, int N>
        struct enumeration
        {
            template<int I>
                using enumerator = enumerator_impl<Enumeration, N, I>;

            static constexpr bool is_requirable_concept = false;
            static constexpr bool is_requirable = false;
            static constexpr bool is_preferable = false;

            using polymorphic_query_result_type = Enumeration;

            template<typename Executor>
            static constexpr auto static_query() ->
                decltype(Executor::query(*static_cast<Enumeration*>(0)))
            {
                return Executor::query(Enumeration());
            }

            template<typename Executor, int I = 0>
            struct static_query_type :
                std::conditional_t<
                properties::query_static_traits<Executor, enumerator<I> >::is_valid,
                std::decay<enumerator<I> >,
                std::conditional_t<
                    I + 1 < N,
                    static_query_type<Executor, I + 1>,
                    std::decay<std::enable_if<false> >
                >
                >
            {
            };

            template<typename Executor>
            static constexpr auto static_query()
                -> std::enable_if_t<
                !properties::query_member_traits<Executor, Enumeration>::is_valid,
                decltype(static_query_type<Executor>::type::template static_query_v<Executor>)
                >
            {
                return static_query_type<Executor>::type::template static_query_v<Executor>;
            }

            template<typename Executor,
                typename T = std::enable_if_t<
                (enumeration::static_query<Executor>(), true),
                decltype(enumeration::static_query<Executor>())
                >>
            static constexpr T static_query_v = enumeration::static_query<Executor>();

            constexpr enumeration()
                : m_value(-1)
            {
            }

            template<int I, typename = std::enable_if_t<I < N>>
            constexpr enumeration(enumerator<I>)
                : m_value(enumerator<I>::value().m_value)
            {
            }

            template<typename Executor, int I = 0>
            struct query_type :
                std::conditional_t<
                can_query<Executor, enumerator<I> >::value,
                std::decay<enumerator<I> >,
                std::conditional_t<
                    I + 1 < N,
                    query_type<Executor, I + 1>,
                    std::decay<std::enable_if<false> >
                >
                >
            {
            };

            template<typename Executor, typename Property,
                typename = std::enable_if_t<std::is_same<Property, Enumeration>::value>>
            friend constexpr auto query(const Executor& ex, const Property&)
                noexcept(noexcept(query(ex, typename query_type<Executor>::type())))
                -> decltype(query(ex, typename query_type<Executor>::type()))
            {
                return query(ex, typename query_type<Executor>::type());
            }

            friend constexpr bool operator==(const Enumeration& a, const Enumeration& b) noexcept
            {
                return a.m_value == b.m_value;
            }

            friend constexpr bool operator!=(const Enumeration& a, const Enumeration& b) noexcept
            {
                return a.m_value != b.m_value;
            }

            private:
            template<typename, int, int> friend struct enumerator_impl;

            constexpr enumeration(int v)
                : m_value(v)
            {
            }
        private:
            int m_value;
        };

    } // namespace detail
    


} // namespace boost::numeric::ublas::parallel::execution

namespace boost::numeric::ublas::parallel{
    
    template<typename Entity, typename Enumeration, int N, int I>
    struct is_applicable_property<Entity, execution::detail::enumerator_impl<Enumeration, N, I>>
        : is_applicable_property<Entity, Enumeration> {};
    
} // namespace boost::numeric::ublas::parallel



#endif