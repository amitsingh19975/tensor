//
//  Copyright (c) 2018-2019, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_HPP_

#include <algorithm>
#include <boost/numeric/ublas/tensor/detail/extents_functions.hpp>
#include <boost/numeric/ublas/tensor/dynamic_extents.hpp>
#include <boost/numeric/ublas/tensor/static_extents.hpp>

namespace boost::numeric::ublas {

    template <class LExtents, class RExtents,
        std::enable_if_t<detail::is_extents<LExtents>::value && detail::is_extents<RExtents>::value, int> = 0
    >
    constexpr bool operator==(LExtents const& lhs, RExtents const& rhs){
        static_assert(detail::is_extents<RExtents>::value && detail::is_extents<LExtents>::value,
            "boost::numeric::ublas::operator==() : invalid type, type should be an extents");
        if( lhs.size() != rhs.size() ){
            return false;
        }else{
            return std::equal(lhs.begin(), lhs.end(), rhs.begin());
        }
    }

    template <class LExtents, class RExtents,
        std::enable_if_t<detail::is_extents<LExtents>::value && detail::is_extents<RExtents>::value, int> = 0
    >
    constexpr bool operator!=(LExtents const& lhs, RExtents const& rhs){
        static_assert(detail::is_extents<RExtents>::value && detail::is_extents<LExtents>::value,
            "boost::numeric::ublas::operator!=() : invalid type, type should be an extents");
        return !(lhs == rhs);
    }

    template <class Extents,
        std::enable_if_t<detail::is_extents<Extents>::value, int> = 0
    >
    std::ostream& operator<<(std::ostream& os, Extents const& e){
        static_assert(detail::is_extents<Extents>::value,
            "boost::numeric::ublas::operator<<() : invalid type, type should be an extents");
        return os<<to_string(e);
    }

}

#endif
