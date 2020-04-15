#ifndef _BOOST_UBLAS_TENSOR_SIMD_BASIC_SIMD_TYPE_HPP
#define _BOOST_UBLAS_TENSOR_SIMD_BASIC_SIMD_TYPE_HPP

#include "detail/simd_helper.hpp"
#include <ostream>

namespace boost::numeric::ublas::simd{
    
    template<size_t N, typename T> struct basic_simd_type;

} // namespace boost::numeric::ublas::simd


namespace boost::numeric::ublas::simd{
    
    template<size_t N, typename T>
    struct basic_simd_type{

        using value_type        = T;
        using simd_type         = detail::simd_type_t<N,value_type>;
        using pointer           = T*;
        using const_pointer     = T const*;
        using reference         = T&;
        using const_reference   = T const&;
        using iterator          = pointer;
        using const_iterator    = const_pointer;
        using size_type         = size_t;

        constexpr static auto bsz = detail::block_size_v<N,T>;
        
        basic_simd_type(){}
        basic_simd_type( simd_type const& data ) { get_simd() = data;}
        basic_simd_type( std::initializer_list<T> const& li ){ std::copy(li.begin(),li.begin() + bsz,m_data); }
        basic_simd_type( const_pointer data ) { std::copy(data, data + bsz, m_data); }
        basic_simd_type( const_pointer data, size_type w ) { 
           set_data(data,w);
        }
        
        template<typename... Args>
        basic_simd_type(T const& a, Args&&... args) { get_simd() = detail::assignment<N,T>{}(a, std::forward<Args>(args)...); }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void set_data( const_pointer data, size_type w ) noexcept{
            auto mp = m_data;
            auto dp = data;
            for(auto i = 0; i < bsz; ++i, ++mp, dp += w ){
                *mp = *dp;
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr reference operator[]( size_type k ) noexcept{
            return m_data[k];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr const_reference operator[]( size_type k ) const noexcept{
            return m_data[k];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr reference at( size_type k ){
            if( k >= bsz ) throw std::out_of_range("boost::numeric::ublas::simd::basic_simd_type::at(size_type): index is greater than block size");
            return this->operator[](k);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr const_reference at( size_type k ) const{
            if( k >= bsz ) throw std::out_of_range("boost::numeric::ublas::simd::basic_simd_type::at(size_type): index is greater than block size");
            return this->operator[](k);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto size() const noexcept {
            return bsz;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE iterator begin() noexcept{
            reinterpret_cast<pointer>(&m_data);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE iterator end() noexcept{
            reinterpret_cast<pointer>(&m_data) + bsz;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE const_iterator begin() const noexcept{
            reinterpret_cast<pointer>(&m_data);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE const_iterator end() const noexcept{
            reinterpret_cast<pointer>(&m_data) + bsz;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto const& data() const noexcept{
            return m_data;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto& data() noexcept{
            return m_data;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto& get_simd() noexcept{
            return *reinterpret_cast< simd_type* >(m_data);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto const& get_simd() const noexcept{
            return *reinterpret_cast< simd_type const* >(m_data);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr basic_simd_type operator*( basic_simd_type const& rhs ) const noexcept {
            basic_simd_type ret{};
            auto& na = get_simd();
            auto& nb = rhs.get_simd();
            ret.get_simd() = detail::multiplies<value_type>{}(na,nb);
            return ret;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr basic_simd_type operator/( basic_simd_type const& rhs ) const noexcept {
            basic_simd_type ret{};
            auto& na = get_simd();
            auto& nb = rhs.get_simd();
            ret.get_simd() = detail::divides<value_type>{}(na,nb);
            return ret;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr basic_simd_type operator+( basic_simd_type const& rhs ) const noexcept {
            basic_simd_type ret{};
            auto& na = get_simd();
            auto& nb = rhs.get_simd();
            ret.get_simd() = detail::addition<value_type>{}(na,nb);
            return ret;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr basic_simd_type operator-( basic_simd_type const& rhs ) const noexcept {
            basic_simd_type ret{};
            auto& na = get_simd();
            auto& nb = rhs.get_simd();
            ret.get_simd() = detail::subtraction<value_type>{}(na,nb);
            return ret;
        }


        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr basic_simd_type& operator*=( basic_simd_type const& rhs ) noexcept {
            auto& na = get_simd();
            auto& nb = rhs.get_simd();
            get_simd() = detail::multiplies<value_type>{}(na,nb);
            return *this;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE basic_simd_type& operator/=( basic_simd_type const& rhs ) noexcept {
            auto& na = get_simd();
            auto& nb = rhs.get_simd();
            get_simd() = detail::divides<value_type>{}(na,nb);
            return *this;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr basic_simd_type& operator+=( basic_simd_type const& rhs ) noexcept {
            auto& na = get_simd();
            auto& nb = rhs.get_simd();
            get_simd() = detail::addition<value_type>{}(na,nb);
            return *this;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr basic_simd_type& operator-=( basic_simd_type const& rhs ) noexcept {
            auto& na = get_simd();
            auto& nb = rhs.get_simd();
            get_simd() = detail::subtraction<value_type>{}(na,nb);
            return *this;
        }

        friend auto& operator<<( std::ostream& os, basic_simd_type const& rhs ){
            if constexpr ( std::is_same_v<typename basic_simd_type::value_type, int8_t> || std::is_same_v<typename basic_simd_type::value_type, char> ){
                os<<"[ ";
                for(auto i = 0; i < rhs.bsz - 1; ++i) os<<(int)rhs[i]<<' ';
                os<<(int)(rhs[rhs.bsz - 1])<<" ]";
            }else{
                os<<"[ ";
                for(auto i = 0; i < rhs.bsz - 1; ++i) os<<rhs[i]<<' ';
                os<<(rhs[rhs.bsz - 1])<<" ]";
            }
            return os;
        }
    private:
        T m_data[bsz]{0};
    };

    template<typename T, typename... Args>
    basic_simd_type(T const&, Args&&... args) -> basic_simd_type< ( (sizeof...(args) + 1 ) * sizeof(T) * 8 ), T>;

    template<size_t N, typename T>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto dot_prod(basic_simd_type<N,T> const& lhs, basic_simd_type<N,T> const& rhs) noexcept{
        auto& na = lhs.get_simd();
        auto& nb = rhs.get_simd();
        return detail::dot_product<N,T>{}(na, nb);
    }

} // namespace boost::numeric::ublas::simd

#endif