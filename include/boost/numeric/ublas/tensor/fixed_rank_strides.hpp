//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//
/// \file strides.hpp Definition for the basic_strides template class


#ifndef _BOOST_UBLAS_TENSOR_FIXED_RANK_STRIDES_HPP_
#define _BOOST_UBLAS_TENSOR_FIXED_RANK_STRIDES_HPP_

#include <boost/numeric/ublas/tensor/fixed_rank_extents.hpp>

namespace boost { 
namespace numeric { 
namespace ublas {

/** @brief Template class for storing tensor strides for iteration with runtime variable size.
 *
 * Proxy template class of std::array<int_type,N>.
 *
 */
template<typename T, std::size_t N, typename L>
class basic_fixed_rank_strides
{
public:

    using base_type = std::array<T, N>;

    static_assert( std::numeric_limits<typename base_type::value_type>::is_integer,
                                 "Static error in boost::numeric::ublas::basic_fixed_rank_strides: type must be of type integer.");
    static_assert(!std::numeric_limits<typename base_type::value_type>::is_signed,
                                "Static error in boost::numeric::ublas::basic_fixed_rank_strides: type must be of type unsigned integer.");
    static_assert(std::is_same<L,first_order>::value || std::is_same<L,last_order>::value,
                                "Static error in boost::numeric::ublas::basic_fixed_rank_strides: layout type must either first or last order");


    using layout_type           = L;
    using value_type            = typename base_type::value_type;
    using reference             = typename base_type::reference;
    using const_reference       = typename base_type::const_reference;
    using size_type             = typename base_type::size_type;
    using const_pointer         = typename base_type::const_pointer;
    using const_iterator        = typename base_type::const_iterator;


    /** @brief Default constructs basic_fixed_rank_strides
     *
     * @code auto ex = basic_fixed_rank_strides<unsigned>{};
     */
    constexpr explicit basic_fixed_rank_strides()
        : _base{}
    {
    }

    /** @brief Constructs basic_fixed_rank_strides from basic_extents for the first- and last-order storage formats
     *
     * @code auto strides = basic_fixed_rank_strides<unsigned>( basic_extents<std::size_t>{2,3,4} );
     *
     */
    template<typename ExtentsType>
    basic_fixed_rank_strides(ExtentsType const& s)
    {
        static_assert( is_extents_v<ExtentsType>, "boost::numeric::ublas::basic_fixed_rank_strides(ExtentsType const&) : "
            "ExtentsType should be tensor extents type" 
        );

        _base.fill(value_type(1));

        if( s.empty() )
            return;

        if( !valid(s) )
            throw std::runtime_error("Error in boost::numeric::ublas::basic_fixed_rank_strides(ExtentsType const&) : shape is not valid.");        

        if( is_vector(s) || is_scalar(s) )
            return;

        if( this->size() < 2 )
            throw std::runtime_error("Error in boost::numeric::ublas::basic_fixed_rank_strides(ExtentsType const&) : size of strides must be greater or equal 2.");


        if constexpr ( std::is_same_v<layout_type,first_order> ){

            size_type k = 1ul, kend = this->size();
            for(; k < kend; ++k)
                _base[k] = _base[k-1] * s[k-1];

        } else if constexpr ( std::is_same_v<layout_type,last_order> ) {

            size_type k = this->size()-2, kend = 0ul;
            for(; k > kend; --k)
                _base[k] = _base[k+1] * s[k+1];
            _base[0] = _base[1] * s[1];

        }
    }

    basic_fixed_rank_strides(basic_fixed_rank_strides const& l)
        : _base(l._base)
    {}

    basic_fixed_rank_strides(basic_fixed_rank_strides && l )
        : _base(std::move(l._base))
    {}

    basic_fixed_rank_strides(base_type const& l )
        : _base(l)
    {}

    basic_fixed_rank_strides(base_type && l )
            : _base(std::move(l))
    {}

    ~basic_fixed_rank_strides() = default;


    basic_fixed_rank_strides& operator=(basic_fixed_rank_strides other)
    {
        swap (*this, other);
        return *this;
    }

    friend void swap(basic_fixed_rank_strides& lhs, basic_fixed_rank_strides& rhs) {
        std::swap(lhs._base   , rhs._base);
    }

    [[nodiscard]] inline
    constexpr const_reference operator[] (size_type p) const{
        return _base[p];
    }

    [[nodiscard]] inline
    constexpr const_pointer data() const{
        return _base.data();
    }

    [[nodiscard]] inline
    constexpr const_reference at (size_type p) const{
        return _base.at(p);
    }

    [[nodiscard]] inline
    constexpr const_reference back () const{
        return _base[N - 1];
    }

    [[nodiscard]] inline
    constexpr reference back (){
        return _base[N - 1];
    }

    [[nodiscard]] inline
    constexpr bool empty() const{
        return _base.empty();
    }

    [[nodiscard]] inline
    constexpr size_type size() const{
        return _base.size();
    }

    [[nodiscard]] inline
    constexpr const_iterator begin() const{
        return _base.begin();
    }

    [[nodiscard]] inline
    constexpr const_iterator end() const{
        return _base.end();
    }
    
    [[nodiscard]] inline
    constexpr base_type const& base() const{
        return this->_base;
    }

protected:
    base_type _base;
};


}
}
}

#endif
