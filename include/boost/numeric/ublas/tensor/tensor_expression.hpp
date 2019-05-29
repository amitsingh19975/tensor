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
//

#ifndef BOOST_UBLAS_TENSOR_YAP_EXPRESSIONS_HPP
#define BOOST_UBLAS_TENSOR_YAP_EXPRESSIONS_HPP

#include <boost/config.hpp>
#include <boost/yap/print.hpp>
#include <boost/yap/yap.hpp>
#include "expression_transforms.hpp"

namespace boost::numeric::ublas::detail
{

template <boost::yap::expr_kind Kind, typename Tuple>
class tensor_expression
{
public:
    const static boost::yap::expr_kind kind = Kind;

    Tuple elements;

    bool is_extent_static = false; /* If true implies this expression is formed of
                                    static extent tensor only. */

    BOOST_UBLAS_INLINE
    decltype ( auto ) operator() ( size_t i )
    {
        return boost::yap::evaluate (
                   boost::yap::transform ( *this, transforms::at_index{i} ) );
    }

    // @Todo(coder3101): Add a method eval(). For Explicit Evaluation like Eigen

};
}  // namespace boost::numeric::ublas::detail

#endif
