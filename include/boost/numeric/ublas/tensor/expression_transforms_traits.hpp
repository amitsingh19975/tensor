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
#include <utility>

/**
 * All these traits are used to check the state of the expression for
 * optimization. Currently we are holding the optimization to tensor only. Maybe
 * in future we extent it to other ublas containers like vector, matrix that are
 * embedded in the tensor expression.
 */

namespace boost::numeric::ublas::detail {
template <boost::yap::expr_kind, typename> struct tensor_expression;
}

namespace boost::numeric::ublas::detail::transforms {
template <class T> struct is_multiply { static constexpr bool value = false; };

template <class operandA, class operandB>
struct is_multiply<boost::numeric::ublas::detail::tensor_expression<
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

template <class T> struct is_addition { static constexpr bool value = false; };

template <class operandA, class operandB>
struct is_addition<boost::numeric::ublas::detail::tensor_expression<
    boost::yap::expr_kind::plus,
    boost::hana::tuple<
        boost::numeric::ublas::detail::tensor_expression<
            boost::yap::expr_kind::terminal, boost::hana::tuple<operandA>>,
        boost::numeric::ublas::detail::tensor_expression<
            boost::yap::expr_kind::terminal, boost::hana::tuple<operandB>>>>> {

  static constexpr bool value =
      is_tensor_v<std::remove_reference_t<operandA>> &&
      is_tensor_v<std::remove_reference_t<operandB>>;
};

template <class T> struct is_scalar_multiply {
  static constexpr bool value = false;
};

template <class A, class B>
struct is_scalar_multiply<boost::numeric::ublas::detail::tensor_expression<
    boost::yap::expr_kind::multiplies,
    boost::hana::tuple<
        boost::numeric::ublas::detail::tensor_expression<
            boost::yap::expr_kind::terminal, boost::hana::tuple<A>>,
        boost::numeric::ublas::detail::tensor_expression<
            boost::yap::expr_kind::terminal, boost::hana::tuple<B>>>>> {

  static constexpr bool first_scalar =
      std::is_arithmetic_v<std::remove_reference_t<A>> &&
      is_tensor_v<std::remove_reference_t<B>>;
  static constexpr bool second_scalar =
      std::is_arithmetic_v<std::remove_reference_t<B>> &&
      is_tensor_v<std::remove_reference_t<A>>;
  static constexpr bool value = first_scalar || second_scalar;
};

template <class T> struct is_terminal { static constexpr bool value = false; };

template <class operand>
struct is_terminal<boost::numeric::ublas::detail::tensor_expression<
    boost::yap::expr_kind::terminal, boost::hana::tuple<operand>>> {
  static constexpr bool value = is_tensor_v<std::remove_reference_t<operand>>;
};

} // namespace boost::numeric::ublas::detail::transforms

#endif // UBLAS_EXPRESSION_TRANSFORMS_TRAITS_HPP
