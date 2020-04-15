#ifndef _BOOST_UBLAS_TENSOR_SIMD_MAT_STORAGE_HPP
#define _BOOST_UBLAS_TENSOR_SIMD_MAT_STORAGE_HPP

#include "vector_storage.hpp"
namespace boost::numeric::ublas::simd{

    namespace op{
        struct transpose{};
        struct no_op{};
    } // namespace op
    
    template< size_t B, typename T, typename L = op::no_op, size_t... > struct mat_storage;
    

} // namespace boost::numeric::ublas::simd

namespace boost::numeric::ublas::simd{
    
    template<size_t B, typename T, typename O>
    struct mat_storage<B,T,O>{
        using value_type        = T;
        using operation_type    = O;
        using simd_type         = basic_simd_type<B,T>;
        using base_type         = std::vector<simd_type>;
        using iterator          = typename base_type::iterator;
        using const_iterator    = typename base_type::const_iterator;
        using reference         = typename base_type::reference;
        using const_reference   = typename base_type::const_reference;
        using size_type         = typename base_type::size_type;

        constexpr static auto bsz = detail::block_size_v<B,T>;

        mat_storage( ) = default;

        mat_storage( size_type row, size_type col ) 
            : m_row(row), m_col(col)
        {
            if constexpr( std::is_same_v< operation_type, op::transpose > ){
                std::swap(m_row, m_col);
            }
            m_col = detail::get_array_size<B,T>(m_col);
            m_data.resize( m_row * m_col );
        }

        mat_storage( value_type const* data, size_type row, size_type col, size_type wr, size_type wc )
            : mat_storage(row,col)
        {
            m_rstride = wr;
            m_cstride = wc;
            if constexpr ( std::is_same_v< operation_type, op::transpose > ){
                set_data_transpose(data,row,col);
            }else{
                set_data(data,row,col);
            }
        }

        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr reference operator[]( size_type k ) noexcept {
            return m_data[k];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr const_reference operator[]( size_type k ) const noexcept {
            return m_data[k];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr reference at( size_type i, size_type j ) {
            return m_data[ i * m_col + j ];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr const_reference at( size_type i, size_type j ) const {
            return m_data[ i * m_col + j ];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto row() const noexcept{ 
            return m_row ;
        }
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto col() const noexcept{ 
            return m_col;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto& element( size_type k ) noexcept{
            auto block_pos = k / bsz;
            auto pos = k % bsz;
            return m_data[block_pos][pos];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto const& element( size_type k ) const noexcept{
            auto block_pos = k / bsz;
            auto pos = k % bsz;
            return m_data[block_pos][pos];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr size_type size() const noexcept {
            return m_data.size();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto elements() const noexcept{
            return m_data.size() * bsz;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE iterator begin() noexcept{
            return m_data.begin();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE iterator end() noexcept{
            return m_data.end();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE const_iterator begin() const noexcept{
            return m_data.begin();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE const_iterator end() const noexcept{
            return m_data.end();
        }

        friend auto& operator<<( std::ostream& os, mat_storage const& rhs ){
            for( auto i = 0; i < rhs.row(); ++i ){
                for( auto j = 0; j < rhs.col(); ++j ){
                    os<<rhs.at(i,j);
                }
                os<<'\n';
            }
            return os;
        }

    private:

        constexpr void set_data(value_type const* data, size_type row, size_type col) noexcept{
            auto curr_data = reinterpret_cast<value_type*>(m_data.data());
            
            auto rem = col % bsz;
            auto chunck = col / bsz;

            auto pi = data;
            
            if ( rem != 0 ){
                
                for( auto i = size_type(0); i < row; pi += m_rstride, ++i ){
                    auto pj = pi;
                    for( auto j = size_type(0); j < chunck; pj += m_cstride, ++j ){
                        for( auto k = 0ul; k < bsz; ++k, pj += m_cstride, ++curr_data ){
                            *curr_data = *pj;
                        }
                        pj -= m_cstride;
                    }
                    for( auto k = 0ul; k < rem; ++k, ++curr_data, pj += m_cstride ){
                        *curr_data = *pj;
                    }
                    curr_data += bsz - rem;
                }

            }else{
                for( auto i = size_type(0); i < row; pi += m_rstride, ++i ){
                    auto pj = pi;
                    for( auto j = size_type(0); j < chunck; pj += m_cstride, ++j ){
                        for( auto k = 0ul; k < bsz; ++k, pj += m_cstride, ++curr_data ){
                            *curr_data = *pj;
                        }
                        pj -= m_cstride;
                    }
                }
            }
        }

        constexpr void set_data_transpose(value_type const* data, size_type row, size_type col) noexcept{
            auto curr_data = reinterpret_cast<value_type*>(m_data.data());
            
            auto rem = row % bsz;
            auto chunck = row / bsz;
            
            auto pj = data;

            if ( rem != 0 ){
                
                for( auto j = size_type(0); j < col; pj += m_cstride, ++j ){
                    auto pi = pj;
                    for( auto i = size_type(0); i < chunck; pi += m_rstride, ++i ){
                        for( auto k = 0ul; k < bsz; ++k, pi += m_rstride, ++curr_data ){
                            *curr_data = *pi;
                        }
                        pi -= m_rstride;
                    }
                    
                    for( auto k = 0ul; k < rem; ++k, pi += m_rstride, ++curr_data ){
                        *curr_data = *pi;
                    }
                    curr_data += bsz - rem;
                }

            }else{
                for( auto j = size_type(0); j < col; pj += m_cstride, ++j ){
                    auto pi = pj;
                    for( auto i = size_type(0); i < chunck; pi += m_rstride, ++i ){
                        for( auto k = 0ul; k < bsz; ++k, pi += m_rstride, ++curr_data ){
                            *curr_data = *pi;
                        }
                        pi -= m_rstride;
                    }
                }
            }
        }

    private:
        base_type m_data;
        size_type m_row;
        size_type m_col;
        size_type m_rstride;
        size_type m_cstride;
    };

} // namespace boost::numeric::ublas::simd


#endif