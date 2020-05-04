#ifndef _BOOST_UBLAS_TENSOR_DETAIL_PARALLEL_UTILITY_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_PARALLEL_UTILITY_HPP

#include "../fwd.hpp"
#include <string>

namespace boost::numeric::ublas::parallel::detail{
    
    template<typename T, typename U> 
    struct cache_aligned_pair{
        static_assert( TENSOR_CACHE_LINE_SIZE / 8 >= sizeof(T) + sizeof(U), "boost::numeric::ublas::parallel::detail::cache_aligned_pair : out of cache line" );

        using first_type = T;
        using second_type = U;

        static constexpr auto const fsz = sizeof(first_type);
        static constexpr auto const ssz = sizeof(second_type);

        constexpr cache_aligned_pair()  noexcept = default;

        constexpr cache_aligned_pair( first_type fst, second_type snd ) noexcept {
            memcpy(m_data, &fst, fsz);
            memcpy(m_data + fsz, &snd, ssz);
        };

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto const& fst() const noexcept{
            return *reinterpret_cast<first_type*>(m_data);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto const& snd() const noexcept{
            return *reinterpret_cast<second_type*>(m_data + fsz);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto& fst() noexcept{
            return *reinterpret_cast<first_type*>(m_data);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto& snd() noexcept{
            return *reinterpret_cast<second_type*>(m_data + fsz);
        }

    private:
        char m_data[fsz + ssz]{0};
    };

} // namespace boost::numeric::ublas::parallel::detail

#endif