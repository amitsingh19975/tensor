#ifndef _BOOST_UBLAS_TENSOR_SIMD_VECTOR_STORAGE_HPP
#define _BOOST_UBLAS_TENSOR_SIMD_VECTOR_STORAGE_HPP

#include "basic_simd_type.hpp"
#include <vector>

namespace boost::numeric::ublas::simd{

    namespace detail{
        template<size_t N, typename T>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE size_t get_array_size( size_t elements ){
            auto bsz = detail::block_size_v<N,T>;
            auto rem = elements % bsz;
            return rem == 0 ? 
                elements / bsz : 
                ( elements + bsz - rem ) / bsz;
        }
    } // namespace detail
    
    template<size_t B, typename T, size_t...> struct vec_storage;

} // namespace boost::numeric::ublas::simd

namespace boost::numeric::ublas::simd{
    
    template<size_t B, typename T>
    struct vec_storage<B,T>{
        using value_type        = T;
        using simd_type         = basic_simd_type<B,T>;
        using base_type         = std::vector<simd_type>;
        using iterator          = typename base_type::iterator;
        using const_iterator    = typename base_type::const_iterator;
        using reference         = typename base_type::reference;
        using const_reference   = typename base_type::const_reference;
        using size_type         = typename base_type::size_type;

        constexpr static auto bsz = detail::block_size_v<B,T>;

        vec_storage( ) = default;

        vec_storage( size_type size ) 
            : m_data(detail::get_array_size<B,T>(size))
        {
        }

        vec_storage( value_type const* data, size_type size, size_type w)
            : vec_storage(size)
        {
            set_data(data, size, w);
        }

        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr reference operator[]( size_type k ) noexcept {
            return m_data[k];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr const_reference operator[]( size_type k ) const noexcept {
            return m_data[k];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr reference at( size_type i ) {
            return m_data[ i ];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr const_reference at( size_type i ) const {
            return m_data[ i ];
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

        friend auto& operator<<( std::ostream& os, vec_storage const& rhs ){
            os<<"[ ";
            for( auto i = 0; i < rhs.size(); ++i ){
                os<<rhs.at(i)<<' ';
            }
            return os<<"]";
        }

    private:

        constexpr void set_data(value_type const* data, size_type size, size_type w) noexcept{
            auto curr_data = reinterpret_cast<value_type*>(m_data.data());
            
            auto rem = size % bsz;
            auto chunck = size / bsz;

            auto pi = data;
            
            if ( rem != 0 ){
                
                for( auto i = size_type(0); i < chunck; pi += w, ++i ){
                    for( auto k = 0ul; k < bsz; ++k, pi += w, ++curr_data ){
                        *curr_data = *pi;
                    }
                    pi -= w;
                }
                for( auto k = 0ul; k < rem; ++k, ++curr_data, pi += w ){
                    *curr_data = *pi;
                }
                curr_data += bsz - rem;

            }else{
                for( auto i = size_type(0); i < chunck; pi += w, ++i ){
                    for( auto k = 0ul; k < bsz; ++k, pi += w, ++curr_data ){
                        *curr_data = *pi;
                    }
                    pi -= w;
                }
            }
        }

    private:
        base_type m_data;
    };
}

#endif