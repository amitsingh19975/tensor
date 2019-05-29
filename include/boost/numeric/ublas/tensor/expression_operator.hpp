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

#ifndef BOOST_UBLAS_EXPRESSION_OPERATOR_HPP
#define BOOST_UBLAS_EXPRESSION_OPERATOR_HPP

#include <boost/yap/yap.hpp>
//#include <algorithm>
//#include <type_traits>
//#include "multi_index_utility.hpp"
//#include "functions.hpp"

namespace boost {
namespace numeric {
namespace ublas {

template <class element_type, class storage_format, class storage_type>
class tensor;

template <class E>
class matrix_expression;

template <class E>
class vector_expression;

namespace detail {
template <boost::yap::expr_kind K, typename A> /* A : Hana Arguments */
class tensor_expression;
}

}  // namespace ublas
}  // namespace numeric
}  // namespace boost

#define BOOST_UBLAS_TENSOR_TENSOR_OPERATOR(op_name)                           \
  template <::boost::yap::expr_kind Kind, typename Tuple, typename Expr>      \
  constexpr auto operator BOOST_YAP_INDIRECT_CALL(op_name)(                   \
      boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> const     \
          &lhs,                                                               \
      Expr &&rhs) {                                                           \
    using lhs_type = ::boost::yap::detail::operand_type_t<                    \
        boost::numeric::ublas::detail::tensor_expression,                     \
        boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> const   \
            &>;                                                               \
    using rhs_type = ::boost::yap::detail::operand_type_t<                    \
        boost::numeric::ublas::detail::tensor_expression, Expr>;              \
    using tuple_type = ::boost::hana::tuple<lhs_type, rhs_type>;              \
    bool s = lhs.is_extent_static && rhs.is_extent_static;                    \
    return boost::numeric::ublas::detail::tensor_expression<                  \
        ::boost::yap::expr_kind::op_name, tuple_type>{                        \
        tuple_type{::boost::yap::detail::make_operand<lhs_type>{}(lhs),       \
                   ::boost::yap::detail::make_operand<rhs_type>{}(            \
                       static_cast<Expr &&>(rhs))},                           \
        s};                                                                   \
  }                                                                           \
  template <::boost::yap::expr_kind Kind, typename Tuple, typename Expr>      \
  constexpr auto operator BOOST_YAP_INDIRECT_CALL(op_name)(                   \
      boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &lhs,     \
      Expr &&rhs) {                                                           \
    using lhs_type = ::boost::yap::detail::operand_type_t<                    \
        boost::numeric::ublas::detail::tensor_expression,                     \
        boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &>;     \
    using rhs_type = ::boost::yap::detail::operand_type_t<                    \
        boost::numeric::ublas::detail::tensor_expression, Expr>;              \
    using tuple_type = ::boost::hana::tuple<lhs_type, rhs_type>;              \
    bool s = lhs.is_extent_static && rhs.is_extent_static;                    \
    return boost::numeric::ublas::detail::tensor_expression<                  \
        ::boost::yap::expr_kind::op_name, tuple_type>{                        \
        tuple_type{::boost::yap::detail::make_operand<lhs_type>{}(lhs),       \
                   ::boost::yap::detail::make_operand<rhs_type>{}(            \
                       static_cast<Expr &&>(rhs))},                           \
        s};                                                                   \
  }                                                                           \
  template <::boost::yap::expr_kind Kind, typename Tuple, typename Expr>      \
  constexpr auto operator BOOST_YAP_INDIRECT_CALL(op_name)(                   \
      boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &&lhs,    \
      Expr &&rhs) {                                                           \
    using lhs_type = ::boost::yap::detail::remove_cv_ref_t<                   \
        boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &&>;    \
    using rhs_type = ::boost::yap::detail::operand_type_t<                    \
        boost::numeric::ublas::detail::tensor_expression, Expr>;              \
    using tuple_type = ::boost::hana::tuple<lhs_type, rhs_type>;              \
    bool s = lhs.is_extent_static && rhs.is_extent_static;                    \
    return boost::numeric::ublas::detail::tensor_expression<                  \
        ::boost::yap::expr_kind::op_name, tuple_type>{                        \
        tuple_type{std::move(lhs),                                            \
                   ::boost::yap::detail::make_operand<rhs_type>{}(            \
                       static_cast<Expr &&>(rhs))},                           \
        s};                                                                   \
  }                                                                           \
  template <typename T, ::boost::yap::expr_kind Kind, typename Tuple>         \
  constexpr auto operator BOOST_YAP_INDIRECT_CALL(op_name)(                   \
      T &&lhs,                                                                \
      boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &&rhs)    \
      ->::boost::yap::detail::free_binary_op_result_t<                        \
          boost::numeric::ublas::detail::tensor_expression,                   \
          ::boost::yap::expr_kind::op_name, T,                                \
          boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &&> { \
    using result_types = ::boost::yap::detail::free_binary_op_result<         \
        boost::numeric::ublas::detail::tensor_expression,                     \
        ::boost::yap::expr_kind::op_name, T,                                  \
        boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &&>;    \
    using lhs_type = typename result_types::lhs_type;                         \
    using rhs_type = typename result_types::rhs_type;                         \
    using tuple_type = ::boost::hana::tuple<lhs_type, rhs_type>;              \
    bool s = lhs.is_extent_static && rhs.is_extent_static;                    \
    return {tuple_type{lhs_type{static_cast<T &&>(lhs)}, std::move(rhs)}, s}; \
  }                                                                           \
  template <typename T, ::boost::yap::expr_kind Kind, typename Tuple>         \
  constexpr auto operator BOOST_YAP_INDIRECT_CALL(op_name)(                   \
      T &&lhs,                                                                \
      boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> const     \
          &rhs)                                                               \
      ->::boost::yap::detail::free_binary_op_result_t<                        \
          boost::numeric::ublas::detail::tensor_expression,                   \
          ::boost::yap::expr_kind::op_name, T,                                \
          boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> const \
              &> {                                                            \
    using result_types = ::boost::yap::detail::free_binary_op_result<         \
        boost::numeric::ublas::detail::tensor_expression,                     \
        ::boost::yap::expr_kind::op_name, T,                                  \
        boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> const   \
            &>;                                                               \
    using lhs_type = typename result_types::lhs_type;                         \
    using rhs_type = typename result_types::rhs_type;                         \
    using tuple_type = ::boost::hana::tuple<lhs_type, rhs_type>;              \
    using rhs_tuple_type = typename result_types::rhs_tuple_type;             \
    bool s = lhs.is_extent_static && rhs.is_extent_static;                    \
    return {tuple_type{lhs_type{static_cast<T &&>(lhs)},                      \
                       rhs_type{rhs_tuple_type{std::addressof(rhs)}}},        \
            s};                                                               \
  }                                                                           \
  template <typename T, ::boost::yap::expr_kind Kind, typename Tuple>         \
  constexpr auto operator BOOST_YAP_INDIRECT_CALL(op_name)(                   \
      T &&lhs,                                                                \
      boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &rhs)     \
      ->::boost::yap::detail::free_binary_op_result_t<                        \
          boost::numeric::ublas::detail::tensor_expression,                   \
          ::boost::yap::expr_kind::op_name, T,                                \
          boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &> {  \
    using result_types = ::boost::yap::detail::free_binary_op_result<         \
        boost::numeric::ublas::detail::tensor_expression,                     \
        ::boost::yap::expr_kind::op_name, T,                                  \
        boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &>;     \
    using lhs_type = typename result_types::lhs_type;                         \
    using rhs_type = typename result_types::rhs_type;                         \
    using tuple_type = ::boost::hana::tuple<lhs_type, rhs_type>;              \
    using rhs_tuple_type = typename result_types::rhs_tuple_type;             \
    bool s = lhs.is_extent_static && rhs.is_extent_static;                    \
    return {tuple_type{lhs_type{static_cast<T &&>(lhs)},                      \
                       rhs_type{rhs_tuple_type{std::addressof(rhs)}}},        \
            s};                                                               \
  }

#define BOOST_UBLAS_UNARY_TENSOR_OPERATOR(op_name)                          \
  template <::boost::yap::expr_kind Kind, typename Tuple>                   \
  auto operator BOOST_YAP_INDIRECT_CALL(op_name)(                           \
      boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> const   \
          &x) {                                                             \
    using lhs_type = ::boost::yap::detail::operand_type_t<                  \
        boost::numeric::ublas::detail::tensor_expression,                   \
        boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> const \
            &>;                                                             \
    using tuple_type = ::boost::hana::tuple<lhs_type>;                      \
    bool s = x.is_extent_static;                                            \
    return boost::numeric::ublas::detail::tensor_expression<                \
        ::boost::yap::expr_kind::op_name, tuple_type>{                      \
        tuple_type{::boost::yap::detail::make_operand<lhs_type>{}(x)}, s};  \
  }                                                                         \
  template <::boost::yap::expr_kind Kind, typename Tuple>                   \
  auto operator BOOST_YAP_INDIRECT_CALL(op_name)(                           \
      boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &x) {   \
    using lhs_type = ::boost::yap::detail::operand_type_t<                  \
        boost::numeric::ublas::detail::tensor_expression,                   \
        boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &>;   \
    using tuple_type = ::boost::hana::tuple<lhs_type>;                      \
    bool s = x.is_extent_static;                                            \
    return boost::numeric::ublas::detail::tensor_expression<                \
        ::boost::yap::expr_kind::op_name, tuple_type>{                      \
        tuple_type{::boost::yap::detail::make_operand<lhs_type>{}(x)}, s};  \
  }                                                                         \
  template <::boost::yap::expr_kind Kind, typename Tuple>                   \
  auto operator BOOST_YAP_INDIRECT_CALL(op_name)(                           \
      boost::numeric::ublas::detail::tensor_expression<Kind, Tuple> &&x) {  \
    using tuple_type = ::boost::hana::tuple<                                \
        boost::numeric::ublas::detail::tensor_expression<Kind, Tuple>>;     \
    bool s = x.is_extent_static;                                            \
    return boost::numeric::ublas::detail::tensor_expression<                \
        ::boost::yap::expr_kind::op_name, tuple_type>{                      \
        tuple_type{std::move(x)}, s};                                       \
  }

// Binary Operators tensor-tensor
BOOST_UBLAS_TENSOR_TENSOR_OPERATOR(plus);
BOOST_UBLAS_TENSOR_TENSOR_OPERATOR(minus);
BOOST_UBLAS_TENSOR_TENSOR_OPERATOR(multiplies);
BOOST_UBLAS_TENSOR_TENSOR_OPERATOR(divides);

// Unary Operator tensor
BOOST_UBLAS_UNARY_TENSOR_OPERATOR(negate);
BOOST_UBLAS_UNARY_TENSOR_OPERATOR(unary_plus);

// Tensor Contraction
// template <class tensor_type_left, class tuple_type_left,
//           class tensor_type_right, class tuple_type_right>
// auto operator*(std::pair<tensor_type_left const &, tuple_type_left> lhs,
//                std::pair<tensor_type_right const &, tuple_type_right> rhs) {
//   using namespace boost::numeric::ublas;

//   auto const &tensor_left = lhs.first;
//   auto const &tensor_right = rhs.first;

//   auto multi_index_left = lhs.second;
//   auto multi_index_right = rhs.second;

//   static constexpr auto num_equal_ind =
//       number_equal_indexes<tuple_type_left, tuple_type_right>::value;

//   if constexpr (num_equal_ind == 0) {
//     return tensor_left * tensor_right;
//   } else if constexpr (num_equal_ind ==
//                            std::tuple_size<tuple_type_left>::value &&
//                        std::is_same<tuple_type_left, tuple_type_right>::value) {
//     return boost::numeric::ublas::inner_prod(tensor_left, tensor_right);
//   } else {
//     auto array_index_pairs =
//         index_position_pairs(multi_index_left, multi_index_right);
//     auto index_pairs = array_to_vector(array_index_pairs);
//     return boost::numeric::ublas::prod(tensor_left, tensor_right,
//                                        index_pairs.first, index_pairs.second);
//   }
// }

#endif