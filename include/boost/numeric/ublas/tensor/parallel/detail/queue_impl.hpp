#ifndef _BOOST_UBLAS_TENSOR_DETAIL_PARALLEL_QUEUE_IMPL_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_PARALLEL_QUEUE_IMPL_HPP

#include "../fwd.hpp"
#include <type_traits>

namespace boost::numeric::ublas::parallel{

    struct default_deleter{
        template<typename T>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( T&& resource ) const noexcept{
            if constexpr( std::is_trivially_destructible_v<T> && std::is_pointer_v<T> ){
                delete resource;
            }else if( !std::is_trivially_destructible_v<T> ){
                resource.~T();
            }
        }

        template<typename T>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( T& resource ) const noexcept{
            if constexpr( std::is_trivially_destructible_v<T> && std::is_pointer_v<T> ){
                delete resource;
            }else if( !std::is_trivially_destructible_v<T> ){
                resource.~T();
            }
        }
    };

    struct dummy_deleter{
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto operator()( ... ) const noexcept{}
    };

    struct queue_traits{
        
    };

    namespace detail{
        


    } // namespace detail
    
    

} // namespace boost::numeric::ublas::parallel


#endif // _BOOST_UBLAS_TENSOR_DETAIL_PARALLEL_QUEUE_IMPL_HPP