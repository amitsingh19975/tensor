#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BLOCK_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BLOCK_HPP

#include "../fwd.hpp"
#include <type_traits>
#include <ostream>
#include <sstream>

namespace boost::numeric::ublas::parallel::execution{
    
    template<typename SizeType = size_t> struct range;

    template<typename SizeType>
    struct range{
        using size_type = SizeType;

        constexpr range() noexcept = default;
        constexpr range( size_type begin, size_type size, size_type stride, size_type subrange = 4 )
            : m_begin(begin)
            , m_size(size)
            , m_stride(stride)
            , m_subrange(subrange)
        {}

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto const begin( size_type k = 0 ) const noexcept{
            auto s = m_begin + k * m_stride * m_subrange;
            return  s >= m_size ? m_size : s ;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto const end( size_type k = 0 ) const noexcept{
            auto e = m_begin + k * m_stride * ( m_subrange ) + m_subrange;
            return e >= m_size ? m_size : e;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto const size( size_type k = 0 ) const noexcept{
            return m_size / m_subrange + (m_size % m_subrange != 0 ? 1 : 0);
        }

        friend auto& operator<<(std::ostream& os, range const& rg){
            std::stringstream ss;
            ss<<"[ Begin: ( "<< rg.m_begin<<" ), Size: ( "<<rg.size()<<" ), Stride: ( "<<rg.m_stride<<" ), Subrange: ( "<<rg.m_subrange<<" ) ]";
            return os<<ss.str();
        }

    private:
        size_type m_begin{0};
        size_type m_size{0};
        size_type m_stride{0};
        size_type m_subrange{ 4 };
    };

} // namespace boost::numeric::ublas::parallel::execution




#endif