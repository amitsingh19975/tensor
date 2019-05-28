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

#ifndef BOOST_UBLAS_TENSOR_EXPRESSION_TRANSFORM_HPP
#define BOOST_UBLAS_TENSOR_EXPRESSION_TRANSFORM_HPP

#include <boost/yap/yap.hpp>

namespace boost {
namespace numeric {
namespace ublas {

template <class element_type, class storage_format, class storage_type>
class tensor;

template <class size_type>
class basic_extents;

namespace detail {

template <boost::yap::expr_kind Kind, typename Tuple>
class tensor_expression;

}

}  // namespace ublas
}  // namespace numeric
}  // namespace boost

namespace boost {
namespace numeric {
namespace ublas {
namespace detail {
/**
 * @brief This namespace contains all the transoforms of
 * that are applied to tensor expressions.
 *
 */
namespace transforms {

struct at_index {
  template <typename T, typename F, typename A>
  decltype(auto) operator()(
      boost::yap::terminal<tensor_expression, tensor<T, F, A>> &expr) {
    boost::yap::make_terminal(boost::yap::value(expr)[index]);
  }
  size_t index;
};

}  // namespace transforms
}  // namespace detail
}  // namespace ublas
}  // namespace numeric
}  // namespace boost

#endif