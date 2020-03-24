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

#include <boost/numeric/ublas/tensor/detail/extents_functions.hpp>
#include <boost/numeric/ublas/tensor/dynamic_extents.hpp>
#include <boost/numeric/ublas/tensor/static_extents.hpp>

namespace boost::numeric::ublas {

template <class RExtents, class LExtents,
    typename std::enable_if_t< detail::is_extents<RExtents>::value && detail::is_extents<LExtents>::value, int > = 0
>
constexpr bool operator==(RExtents const& lhs, LExtents const& rhs){
    if( rhs.size() != lhs.size() ){
        return false;
    }

    for(auto i = 0u; i < rhs.size(); i++){
        if(rhs.at(i) != lhs.at(i)){
            return false;
        }
    }
    return true;
}
template <class RExtents, class LExtents,
    typename std::enable_if_t< detail::is_extents<RExtents>::value && detail::is_extents<LExtents>::value, int > = 0
>
constexpr bool operator!=(RExtents const& lhs, LExtents const& rhs){
    return !(lhs == rhs);
}

template <class Extents,
    typename std::enable_if_t< detail::is_extents<Extents>::value, int > = 0
>
std::ostream& operator<<(std::ostream& os, Extents const& e){
    return os<<to_string(e);
}




}

#endif
