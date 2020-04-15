#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_NOEXCEPT_FUNCTION_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_NOEXCEPT_FUNCTION_HPP

#include "../fwd.hpp"
#include <type_traits>

namespace boost::numeric::ublas::parallel{
    
    namespace detail{

        // template<typename... Types>
        // struct args_pack{
        //     template<typename R>
        //     args_pack( R(*)(Types...) ){}
        // };

        // template<typename R, typename... Types>
        // struct function_skeleton_helper{
        //     using return_type = R;
        //     using arguments_pack = args_pack<Types...>;
        // };

        // template<typename R, typename... Types>
        // BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto get_function_skeleton_helper( R(*)(Types...) ) -> function_skeleton_helper<R,Types...>;
        
        // template<typename Lambda>
        // BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto get_function_skeleton_helper( Lambda const& lm) -> decltype( get_function_skeleton_helper(+lm) );

        
        // template<typename Function>
        // struct function_skeleton_wraper{
        //     function_skeleton_wraper(Function&& func) noexcept 
        //         : fn(std::move(func))
        //     {}
        //     Function fn;

        //     using function_skeleton = decltype(get_function_skeleton_helper(fn));
        // };

        // template<typename To, typename From>
        // auto&& cast_forward(From&& f){
        //     return std::forward<To>(static_cast<To>(f));
        // }

        // template<typename Function, typename R, typename... RemainingArgs1, typename... RemainingArgs2>
        // BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto static_cast_function_args_helper(function_skeleton_wraper<Function> const& fh,
        //         function_skeleton_helper<R, RemainingArgs1...>, RemainingArgs2&& ... args
        //     )
        // {
        //     if constexpr( std::is_same_v< R, void >){
        //         fh.fn( cast_forward<RemainingArgs1>(args)... );
        //     }else{
        //         return  fh.fn( cast_forward<RemainingArgs1>(args)... );
        //     }
        // }

        // template<typename Function, typename... Args>
        // BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto static_cast_function_args( Function&& fn, Args&& ... args )
        // {
        //     auto fh = function_skeleton_wraper(std::move(fn));
        //     using fh_type = decltype(fh);

        //     if constexpr( std::is_same_v< typename fh_type::function_skeleton::return_type, void >){
        //         static_cast_function_args_helper(fh, typename fh_type::function_skeleton{}, std::forward<Args>(args)...);
        //     }else{
        //         return static_cast_function_args_helper(fh, typename fh_type::function_skeleton{}, std::forward<Args>(args)...);
        //     }
        // }

        template< typename Function >
        struct noexcept_fn{
            template<typename OtherFunction,
                typename = std::enable_if_t<
                std::is_constructible_v<Function,OtherFunction&&>
                >>
            noexcept_fn(OtherFunction&& function) noexcept
                : m_fn(std::forward<OtherFunction>(function))
            {}

            noexcept_fn(const noexcept_fn& other) noexcept
                : m_fn(other.m_fn)
            {}

            noexcept_fn(noexcept_fn&& other) noexcept
                : m_fn(std::move(other.m_fn))
            {}

            ~noexcept_fn() noexcept
            {}

            template<typename... Args>
            BOOST_UBLAS_TENSOR_ALWAYS_INLINE auto operator()(Args&&... args) const noexcept
            {
                return m_fn(std::forward<Args>(args)...);
            }
        private:
            mutable Function m_fn;
        };
        
        template<class Function>
        detail::noexcept_fn<std::decay_t<Function>> make_noexcept_fn(Function&& f)
        {
            return {std::forward<Function>(f)};
        }
    } // namespace detail
    


} // namespace boost::numeric::ublas::parallel


#endif