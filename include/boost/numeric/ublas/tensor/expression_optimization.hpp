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

#include "expression_transforms_traits.hpp"
#include <boost/yap/yap.hpp>
#include <boost/type_traits/has_multiplies.hpp>
#include <type_traits>


namespace boost::numeric::ublas::detail::transforms {

//struct apply_scalar_rule {
//  constexpr apply_scalar_rule() = default;
//
//  template <class Expr1, class Expr2>
//  constexpr decltype(auto)
//  operator()(boost::yap::expr_tag<boost::yap::expr_kind::plus>, Expr1 &&e1,
//             Expr2 &&e2) {
//
//    if constexpr (is_terminal<std::remove_reference_t<Expr1>>::value &&
//                  is_terminal<std::remove_reference_t<Expr2>>::value) {
//
//      using tensor_type =
//          std::remove_reference_t<decltype(boost::yap::value(e1))>;
//
//      usable = std::addressof(e1) == std::addressof(e2);
//
//      if constexpr (boost::has_multiplies<
//                        int, typename tensor_type::value_type>::value)
//
//        return boost::yap::make_expression<
//            boost::numeric::ublas::detail::tensor_expression,
//            boost::yap::expr_kind::multiplies>(2, std::forward<Expr1>(e1));
//
//    } else if constexpr (is_scalar_multiply<
//                             std::remove_reference_t<Expr1>>::value &&
//                         is_scalar_multiply<
//                             std::remove_reference_t<Expr2>>::value) {
//    } else {
//      // Add here logic for recursive optimization. As of now forward the expr
//      usable = true;
//      return boost::yap::make_expression()
//    }
//  }
//  bool usable = false;
//};

struct apply_distributive_law {
  constexpr apply_distributive_law() = default;

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(boost::yap::expr_tag<boost::yap::expr_kind::plus>, Expr1 &&e1,
             Expr2 &&e2) {

    if constexpr (is_multiply<std::remove_reference_t<Expr1>>::value &&
                  is_multiply<std::remove_reference_t<Expr2>>::value) {

      auto &operand1 = boost::yap::value(boost::yap::left(e1));
      auto &operand2 = boost::yap::value(boost::yap::right(e1));
      auto &operand3 = boost::yap::value(boost::yap::left(e2));
      auto &operand4 = boost::yap::value(boost::yap::right(e2));

      using Op_1_t = std::remove_reference_t<decltype(operand1)>;
      using Op_2_t = std::remove_reference_t<decltype(operand2)>;
      using Op_3_t = std::remove_reference_t<decltype(operand3)>;
      using Op_4_t = std::remove_reference_t<decltype(operand4)>;

      if constexpr (std::is_same_v<Op_1_t, Op_2_t> &&
                    std::is_same_v<Op_2_t, Op_3_t> &&
                    std::is_same_v<Op_3_t, Op_4_t>) {

        bool eq_1_3 = std::addressof(operand1) == std::addressof(operand3);
        bool eq_1_4 = std::addressof(operand1) == std::addressof(operand4);
        bool eq_2_3 = std::addressof(operand2) == std::addressof(operand3);
        bool eq_2_4 = std::addressof(operand2) == std::addressof(operand4);

        usable = eq_1_3 || eq_1_4 || eq_2_3 || eq_2_4;

        if (eq_1_3) {
          auto inner_op = boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::plus>(operand2, operand4);
          return boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::multiplies>(operand1, std::move(inner_op));
        }
        if (eq_1_4) {
          auto inner_op = boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::plus>(operand2, operand3);
          return boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::multiplies>(operand1, std::move(inner_op));
        }
        if (eq_2_3) {
          auto inner_op = boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::plus>(operand1, operand4);
          return boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::multiplies>(operand2, std::move(inner_op));
        }
        if (eq_2_4) {
          auto inner_op = boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::plus>(operand1, operand3);
          return boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::multiplies>(operand2, std::move(inner_op));
        }
        // This is just to suppress the warning of non-void return end.
        // If this is returned the usable flag will be false and user should not
        // use this returned expression.
        auto inner_op = boost::yap::make_expression<
            boost::numeric::ublas::detail::tensor_expression,
            boost::yap::expr_kind::plus>(operand2, operand4);
        return boost::yap::make_expression<
            boost::numeric::ublas::detail::tensor_expression,
            boost::yap::expr_kind::multiplies>(operand1, std::move(inner_op));

      } else {
        usable = true;
        return boost::yap::make_expression<
            boost::numeric::ublas::detail::tensor_expression,
            boost::yap::expr_kind::plus>(std::forward<Expr1>(e1),
                                         std::forward<Expr2>(e2));
      }
    } else {
#ifndef BOOST_UBLAS_NO_RECURSIVE_OPTIMIZATION
      std::remove_reference_t<decltype(*this)> a, b;

      auto xa = boost::yap::transform(
          boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(
              std::forward<Expr1>(e1)),
          a);
      auto xb = boost::yap::transform(
          boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(
              std::forward<Expr2>(e2)),
          b);
      usable = a.usable && b.usable;
      return boost::yap::make_expression<
          boost::numeric::ublas::detail::tensor_expression,
          boost::yap::expr_kind::plus>(xa, xb);
#else
      usable = true;
      return boost::yap::make_expression<
          boost::numeric::ublas::detail::tensor_expression,
          boost::yap::expr_kind::plus>(std::forward<Expr1>(e1),
                                       std::forward<Expr2>(e2));
#endif
    }
  }

  template <class Expr1, class Expr2>
  constexpr decltype(auto)
  operator()(boost::yap::expr_tag<boost::yap::expr_kind::minus>, Expr1 &&e1,
             Expr2 &&e2) {

    if constexpr (is_multiply<std::remove_reference_t<Expr1>>::value &&
                  is_multiply<std::remove_reference_t<Expr2>>::value) {

      auto &operand1 = boost::yap::value(boost::yap::left(e1));
      auto &operand2 = boost::yap::value(boost::yap::right(e1));
      auto &operand3 = boost::yap::value(boost::yap::left(e2));
      auto &operand4 = boost::yap::value(boost::yap::right(e2));

      using Op_1_t = std::remove_reference_t<decltype(operand1)>;
      using Op_2_t = std::remove_reference_t<decltype(operand2)>;
      using Op_3_t = std::remove_reference_t<decltype(operand3)>;
      using Op_4_t = std::remove_reference_t<decltype(operand4)>;

      if constexpr (std::is_same_v<Op_1_t, Op_2_t> &&
                    std::is_same_v<Op_2_t, Op_3_t> &&
                    std::is_same_v<Op_3_t, Op_4_t>) {

        bool eq_1_3 = std::addressof(operand1) == std::addressof(operand3);
        bool eq_1_4 = std::addressof(operand1) == std::addressof(operand4);
        bool eq_2_3 = std::addressof(operand2) == std::addressof(operand3);
        bool eq_2_4 = std::addressof(operand2) == std::addressof(operand4);

        usable = eq_1_3 || eq_1_4 || eq_2_3 || eq_2_4;

        if (eq_1_3) {
          auto inner_op = boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::minus>(operand2, operand4);
          return boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::multiplies>(operand1, std::move(inner_op));
        }
        if (eq_1_4) {
          auto inner_op = boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::minus>(operand2, operand3);
          return boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::multiplies>(operand1, std::move(inner_op));
        }
        if (eq_2_3) {
          auto inner_op = boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::minus>(operand1, operand4);
          return boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::multiplies>(operand2, std::move(inner_op));
        }
        if (eq_2_4) {
          auto inner_op = boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::minus>(operand1, operand3);
          return boost::yap::make_expression<
              boost::numeric::ublas::detail::tensor_expression,
              boost::yap::expr_kind::multiplies>(operand2, std::move(inner_op));
        }
        // This is just to suppress the warning of non-void return end.
        // If this is returned the usable flag will be false and user should not
        // use this returned expression.
        auto inner_op = boost::yap::make_expression<
            boost::numeric::ublas::detail::tensor_expression,
            boost::yap::expr_kind::minus>(operand2, operand4);
        return boost::yap::make_expression<
            boost::numeric::ublas::detail::tensor_expression,
            boost::yap::expr_kind::multiplies>(operand1, std::move(inner_op));

      } else {
        usable = true;
        return boost::yap::make_expression<
            boost::numeric::ublas::detail::tensor_expression,
            boost::yap::expr_kind::minus>(std::forward<Expr1>(e1),
                                          std::forward<Expr2>(e2));
      }
    } else {
#ifndef BOOST_UBLAS_NO_RECURSIVE_OPTIMIZATION
      std::remove_reference_t<decltype(*this)> a, b;

      auto xa = boost::yap::transform(
          boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(
              std::forward<Expr1>(e1)),
          a);
      auto xb = boost::yap::transform(
          boost::yap::as_expr<boost::numeric::ublas::detail::tensor_expression>(
              std::forward<Expr2>(e2)),
          b);
      usable = a.usable && b.usable;
      return boost::yap::make_expression<
          boost::numeric::ublas::detail::tensor_expression,
          boost::yap::expr_kind::minus>(xa, xb);
#else
      usable = true;
      return boost::yap::make_expression<
          boost::numeric::ublas::detail::tensor_expression,
          boost::yap::expr_kind::minus>(std::forward<Expr1>(e1),
                                        std::forward<Expr2>(e2));
#endif
    }
  }

  bool usable = false;
};
} // namespace boost::numeric::ublas::detail::transforms

#endif // UBLAS_EXPRESSION_OPTIMIZATION_HPP
