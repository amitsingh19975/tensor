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

#ifndef BOOST_UBLAS_EXPRESSION_OPTIMIZATION_HPP
#define BOOST_UBLAS_EXPRESSION_OPTIMIZATION_HPP

#include <boost/numeric/ublas/detail/config.hpp>

#include <boost/type_traits/has_multiplies.hpp>
#include <boost/yap/yap.hpp>
#include <type_traits>
#include <variant>
#include "expression_transforms_traits.hpp"

namespace boost::numeric::ublas::detail::transforms {

struct runtime_scalar_optimizer {
  template <class LExpr, class RExpr>
  decltype(auto) operator()(boost::yap::expr_tag<boost::yap::expr_kind::plus>,
                            LExpr&& lhs, RExpr&& rhs) {
    using lhs_t = std::remove_reference_t<LExpr>;
    using rhs_t = std::remove_reference_t<RExpr>;
    constexpr bool are_tensor = boost::numeric::ublas::is_tensor_v<lhs_t> &&
                                boost::numeric::ublas::is_tensor_v<rhs_t>;

    constexpr bool are_optimizable =
        are_tensor &&
        std::is_same_v<typename lhs_t::value_type, typename rhs_t::value_type>;

    if constexpr (are_optimizable) {
      auto unoptimized_expr = boost::yap::make_expression<
          boost::numeric::ublas::detail::tensor_expression,
          boost::yap::expr_kind::plus>(lhs, rhs);

      auto optimized_expr = boost::yap::make_expression<
          boost::numeric::ublas::detail::tensor_expression,
          boost::yap::expr_kind::multiplies>(2, rhs);

      using optimized_t = std::remove_reference_t<decltype(optimized_expr)>;
      using unoptimized_t = std::remove_reference_t<decltype(unoptimized_expr)>;

      if (std::addressof(lhs) == std::addressof(rhs)) {
        std::variant<optimized_t, unoptimized_t> result(
            std::move(optimized_expr));
        return boost::yap::make_terminal<
            boost::numeric::ublas::detail::tensor_expression>(
            std::move(result));
      } else {
        std::variant<optimized_t, unoptimized_t> result(
            std::move(unoptimized_expr));
        return boost::yap::make_terminal<
            boost::numeric::ublas::detail::tensor_expression>(
            std::move(result));
      }

    } else {
      auto left_expr_optimized =
          boost::yap::transform(std::forward<LExpr>(lhs), *this);
      auto right_expr_optimized =
          boost::yap::transform(std::forward<RExpr>(rhs), *this);
      return boost::yap::make_expression<
          boost::numeric::ublas::detail::tensor_expression,
          boost::yap::expr_kind::plus>(std::move(left_expr_optimized),
                                       std::move(right_expr_optimized));
    }
  }
};

}  // namespace boost::numeric::ublas::detail::transforms

#endif  // UBLAS_EXPRESSION_OPTIMIZATION_HPP
