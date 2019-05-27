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

namespace boost {
namespace numeric {
namespace ublas {

template <class element_type, class storage_format, class storage_type>
class tensor;

template <class size_type>
class basic_extents;

// TODO: put in fwd.hpp
struct tensor_tag {};
}  // namespace ublas
}  // namespace numeric
}  // namespace boost

namespace boost {
namespace numeric {
namespace ublas {
namespace detail {

/** @\brief base class for tensor expressions
 *
 * \note implements crtp - no use of virtual function calls
 *
 * \tparam T type of the tensor
 * \tparam E type of the derived expression (crtp)
 *
 **/
template <boost::yap::expr_kind Kind, typename Tuple>
class tensor_expression {
  const static boost::yap::expr_kind kind = Kind;

  BOOST_UBLAS_INLINE
  decltype(auto) operator()(size_t i) {
    return boost::yap::evaluate(
        boost::yap::transform(*this, transforms::at_index{i}));
  }

 protected:
  explicit tensor_expression() = default;
  tensor_expression(const tensor_expression&) = delete;
  tensor_expression& operator=(const tensor_expression&) = delete;
};

}  // namespace detail
}  // namespace ublas
}  // namespace numeric
}  // namespace boost

#endif
