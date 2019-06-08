//
// Created by ashar on 8/6/19.
//

#ifndef UBLAS_TENSOR_OPERATOR_MACROS_HPP
#define UBLAS_TENSOR_OPERATOR_MACROS_HPP

#include <boost/yap/user_macros.hpp>
#include <boost/yap/yap.hpp>

#define BOOST_UBLAS_TENSOR_TENSOR_TO_EXPR(op_name)                             \
  template <class T1, class T2, class F1, class F2, class A1, class A2>        \
  constexpr decltype(auto) operator BOOST_YAP_INDIRECT_CALL(op_name)(          \
      boost::numeric::ublas::tensor<T1, F1, A1> const &lhs,                    \
      boost::numeric::ublas::tensor<T2, F2, A2> &&rhs) {                       \
    using lhs_type = boost::numeric::ublas::detail::tensor_expression<         \
        boost::yap::expr_kind::terminal,                                       \
        boost::hana::tuple<                                                    \
            boost::numeric::ublas::detail::tensor<T1, F1, A1>>>;               \
    using rhs_type = boost::numeric::ublas::detail::tensor_expression<         \
        boost::yap::expr_kind::terminal,                                       \
        boost::hana::tuple<boost::numeric::ublas::tensor<T2, F2, A2>>>;        \
    using hana_type = boost::hana::tuple<lhs_type, rhs_type>;                  \
    using result_expr = boost::numeric::ublas::detail::tensor_expression<      \
        boost::yap::expr_kind::op_name, hana_type>;                            \
    return result_expr{hana_type{lhs_type(lhs), rhs_type(std::move(rhs))}};    \
  }                                                                            \
  template <class T1, class T2, class F1, class F2, class A1, class A2>        \
  constexpr decltype(auto) operator BOOST_YAP_INDIRECT_CALL(op_name)(          \
      boost::numeric::ublas::tensor<T1, F1, A1> const &lhs,                    \
      boost::numeric::ublas::tensor<T2, F2, A2> const &rhs) {                  \
    using lhs_type = boost::numeric::ublas::detail::tensor_expression<         \
        boost::yap::expr_kind::terminal,                                       \
        boost::hana::tuple<                                                    \
            boost::numeric::ublas::detail::tensor<T1, F1, A1>>>;               \
    using rhs_type = boost::numeric::ublas::detail::tensor_expression<         \
        boost::yap::expr_kind::terminal,                                       \
        boost::hana::tuple<boost::numeric::ublas::tensor<T2, F2, A2>>>;        \
    using hana_type = boost::hana::tuple<lhs_type, rhs_type>;                  \
    using result_expr = boost::numeric::ublas::detail::tensor_expression<      \
        boost::yap::expr_kind::op_name, hana_type>;                            \
    return result_expr{hana_type{lhs_type(lhs), rhs_type(rhs)}};               \
  }                                                                            \
  template <class T1, class T2, class F1, class F2, class A1, class A2>        \
  constexpr decltype(auto) operator BOOST_YAP_INDIRECT_CALL(op_name)(          \
      boost::numeric::ublas::tensor<T1, F1, A1> &&lhs,                         \
      boost::numeric::ublas::tensor<T2, F2, A2> &&rhs) {                       \
    using lhs_type = boost::numeric::ublas::detail::tensor_expression<         \
        boost::yap::expr_kind::terminal,                                       \
        boost::hana::tuple<                                                    \
            boost::numeric::ublas::detail::tensor<T1, F1, A1>>>;               \
    using rhs_type = boost::numeric::ublas::detail::tensor_expression<         \
        boost::yap::expr_kind::terminal,                                       \
        boost::hana::tuple<boost::numeric::ublas::tensor<T2, F2, A2>>>;        \
    using hana_type = boost::hana::tuple<lhs_type, rhs_type>;                  \
    using result_expr = boost::numeric::ublas::detail::tensor_expression<      \
        boost::yap::expr_kind::op_name, hana_type>;                            \
    return result_expr{                                                        \
        hana_type{lhs_type(std::move(lhs)), rhs_type(std::move(rhs))}};        \
  }                                                                            \
  template <class T1, class T2, class F1, class F2, class A1, class A2>        \
  constexpr decltype(auto) operator BOOST_YAP_INDIRECT_CALL(op_name)(          \
      boost::numeric::ublas::tensor<T1, F1, A1> &&lhs,                         \
      boost::numeric::ublas::tensor<T2, F2, A2> const &rhs) {                  \
    using lhs_type = boost::numeric::ublas::detail::tensor_expression<         \
        boost::yap::expr_kind::terminal,                                       \
        boost::hana::tuple<                                                    \
            boost::numeric::ublas::detail::tensor<T1, F1, A1>>>;               \
    using rhs_type = boost::numeric::ublas::detail::tensor_expression<         \
        boost::yap::expr_kind::terminal,                                       \
        boost::hana::tuple<boost::numeric::ublas::tensor<T2, F2, A2>>>;        \
    using hana_type = boost::hana::tuple<lhs_type, rhs_type>;                  \
    using result_expr = boost::numeric::ublas::detail::tensor_expression<      \
        boost::yap::expr_kind::op_name, hana_type>;                            \
    return result_expr{hana_type{lhs_type(std::move(lhs)), rhs_type(rhs)}};    \
  }

#define BOOST_UBLAS_TENSOR_UNARY_TO_EXPR(op_name)                              \
  template <class T, class F, class A>                                         \
  constexpr decltype(auto) operator BOOST_YAP_INDIRECT_CALL(op_name)(          \
      boost::numeric::ublas::tensor<T, F, A> &&val) {                          \
    using operand_type = boost::numeric::ublas::detail::tensor_expression<     \
        boost::yap::expr_kind::terminal,                                       \
        boost::hana::tuple<boost::numeric::ublas::tensor<T, F, A>>>;           \
    using hana_type = boost::hana::tuple<operand_type>;                        \
    using result_type = boost::numeric::ublas::detail::tensor_expression<      \
        boost::yap::expr_kind::op_name, hana_type>;                            \
    return result_type{hana_type{operand_type(std::move(val))}};               \
  }                                                                            \
  template <class T, class F, class A>                                         \
  constexpr decltype(auto) operator BOOST_YAP_INDIRECT_CALL(op_name)(          \
      boost::numeric::ublas::tensor<T, F, A> const &val) {                     \
    using operand_type = boost::numeric::ublas::detail::tensor_expression<     \
        boost::yap::expr_kind::terminal,                                       \
        boost::hana::tuple<boost::numeric::ublas::tensor<T, F, A>>>;           \
    using hana_type = boost::hana::tuple<operand_type>;                        \
    using result_type = boost::numeric::ublas::detail::tensor_expression<      \
        boost::yap::expr_kind::op_name, hana_type>;                            \
    return result_type{hana_type{operand_type(val)}};                          \
  }
#endif // UBLAS_TENSOR_OPERATOR_MACROS_HPP
