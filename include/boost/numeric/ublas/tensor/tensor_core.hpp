//  Copyright (c) 2018-2019
//  Mohammad Ashar Khan
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.

#ifndef BOOST_UBLAS_TENSOR_CORE_HPP
#define BOOST_UBLAS_TENSOR_CORE_HPP

#include "extents.hpp"
#include "strides.hpp"

// Forward the friends;

namespace boost::numeric::ublas{
    
    template <class new_type, class tensor_t>
    auto static_tensor_cast(tensor_t&);
    
    template <class new_type, class tensor_t>
    auto static_tensor_cast(tensor_t&&);
    
    template <class new_type, class tensor_t>
    auto dynamic_tensor_cast(tensor_t&);
    
    template <class new_type, class tensor_t>
    auto dynamic_tensor_cast(tensor_t&&);
    
    template <class new_type, class tensor_t>
    auto reinterpret_tensor_cast(tensor_t&);
    
    template <class new_type, class tensor_t>
    auto reinterpret_tensor_cast(tensor_t&&);
    
    
    template <class TT, class FF, class AA>
    class tensor;
}


namespace boost::numeric::ublas::detail
{

template <class T, class F, class A>
class tensor_core
{
    using array_type = A;
    using layout_type = F;
    using strides_type = basic_strides<std::size_t, layout_type>;
    using extents_type = shape;

public:
    tensor_core() : extents_(), strides_(), data_() {}

    decltype ( auto ) operator() ( size_t index )
    {
        return data_[index];
    }
    
    auto const & extents() const {
        return extents_;
    }

private:
    extents_type extents_;
    strides_type strides_;
    array_type data_;

    template <class TT, class FF, class AA>
    friend class ::boost::numeric::ublas::tensor;
    
    template <boost::yap::expr_kind K, typename Tuple>
    friend struct tensor_expression;
    
    template <class type, typename AB>
    friend auto ::boost::numeric::ublas::static_tensor_cast(AB&);
    
    template <class type, typename AB>
    friend auto ::boost::numeric::ublas::static_tensor_cast(AB&&);
    
    template <class type, typename AB>
    friend auto ::boost::numeric::ublas::dynamic_tensor_cast(AB&);
    
    template <class type, typename AB>
    friend auto ::boost::numeric::ublas::dynamic_tensor_cast(AB&&);
    
    template <class type, typename AB>
    friend auto ::boost::numeric::ublas::reinterpret_tensor_cast(AB&);

    template <class type, typename AB>
    friend auto ::boost::numeric::ublas::reinterpret_tensor_cast(AB&&);

};
}  // namespace boost::numeric::ublas::detail

#endif
