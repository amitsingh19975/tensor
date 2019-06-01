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

#include "expression_transforms.hpp"
#include "extents.hpp"
#include "strides.hpp"
#include <boost/config.hpp>
#include <boost/yap/print.hpp>
#include <boost/yap/yap.hpp>

namespace boost::numeric::ublas::detail {

template <boost::yap::expr_kind Kind, typename Tuple> struct tensor_expression {
  const static boost::yap::expr_kind kind = Kind;

  Tuple elements;

  BOOST_UBLAS_INLINE
  decltype(auto) operator()(size_t i) {
    return boost::yap::evaluate(
        boost::yap::transform(*this, transforms::at_index{i}));
  }

  // todo(coder3101) : Make eval() and eval_to() based on device.

  template <class T, class F = ::boost::numeric::ublas::first_order,
            class A = std::vector<T, std::allocator<T>>>
  auto eval() {
    using namespace boost::hana::literals;

    ::boost::numeric::ublas::tensor<T, F, A> result;

    auto shape_expr = boost::yap::transform(*this, transforms::get_extents{});
    result.elements[0_c].extents_ = shape_expr;
    result.elements[0_c].strides_ = basic_strides<std::size_t, F>{shape_expr};
    result.elements[0_c].data_.resize(shape_expr.product());

    for (auto i = 0u; i < shape_expr.product(); i++)
      result.elements[0_c].data_[i] = this->operator()(i);

    return result;
  }

  template <class T, class F, class A>
  void eval_to(::boost::numeric::ublas::tensor<T, F, A> &target) {
    auto shape_expr = boost::yap::transform(*this, transforms::get_extents{});
    using namespace boost::hana::literals;

    target.elements[0_c].data_.resize(shape_expr.product());
    target.elements[0_c].extents_ = shape_expr;
    target.elements[0_c].strides_ = basic_strides<std::size_t, F>{shape_expr};

    for (auto i = 0u; i < shape_expr.product(); i++)
      target.elements[0_c].data_[i] = this->operator()(i);
  }
};
} // namespace boost::numeric::ublas::detail

#endif
