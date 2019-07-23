//  Copyright (c) 2019-2020
//  Mohammad Ashar Khan, ashar786khan@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google in producing this work
//  which started as a Google Summer of Code project.

#ifndef BOOST_UBLAS_EXPRESSION_TRANSFORMS_TRAITS_HPP
#define BOOST_UBLAS_EXPRESSION_TRANSFORMS_TRAITS_HPP

#include "ublas_type_traits.hpp"
#include <boost/yap/yap.hpp>
#include <type_traits>

namespace boost::numeric::ublas::detail {
template <boost::yap::expr_kind, typename> struct tensor_expression;
}

namespace boost::numeric::ublas::detail::transforms {
template <class T> struct is_multiply_operand {
  static constexpr bool value = false;
};

template <class operandA, class operandB>
struct is_multiply_operand<boost::numeric::ublas::detail::tensor_expression<
    boost::yap::expr_kind::multiplies,
    boost::hana::tuple<
        boost::numeric::ublas::detail::tensor_expression<
            boost::yap::expr_kind::terminal, boost::hana::tuple<operandA>>,
        boost::numeric::ublas::detail::tensor_expression<
            boost::yap::expr_kind::terminal, boost::hana::tuple<operandB>>>>> {
  static constexpr bool value =
      is_tensor_v<std::remove_reference_t<operandA>> &&
      is_tensor_v<std::remove_reference_t<operandB>>;
};

} // namespace boost::numeric::ublas::detail::transforms

#endif // UBLAS_EXPRESSION_TRANSFORMS_TRAITS_HPP
