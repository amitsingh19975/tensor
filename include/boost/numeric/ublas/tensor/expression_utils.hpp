
#ifndef BOOST_UBLAS_EXPRESSION_UTILS_HPP
#define BOOST_UBLAS_EXPRESSION_UTILS_HPP

#include "tensor_expression.hpp"
#include <boost/yap/user_macros.hpp>
#include <boost/yap/yap.hpp>

namespace boost::numeric::ublas::detail {
template <class T> struct is_terminal_type_expr {
  constexpr static bool value = false;
};

template <class T>
struct is_terminal_type_expr<
    tensor_expression<boost::yap::expr_kind::terminal, boost::hana::tuple<T>>> {
  constexpr static bool value = true;
  using value_type = typename std::remove_reference_t<T>;
};

template <class T>
struct is_terminal_type_expr<boost::yap::expression<
    boost::yap::expr_kind::terminal, boost::hana::tuple<T>>> {
  constexpr static bool value = true;
  using value_type = typename std::remove_reference_t<T>;
};

template <class T> struct function_return;
template <class R, class A> struct function_return<R (*)(A)> {
  using type = R;
};

template <class Expr> decltype(auto) get_type(Expr &&e) {

  auto expr = boost::yap::as_expr<tensor_expression>(std::forward<Expr>(e));
  using Expr_t = decltype(expr);

  if constexpr (is_terminal_type_expr<std::remove_reference_t<Expr_t>>::value) {
    using type = typename is_terminal_type_expr<
        std::remove_reference_t<Expr_t>>::value_type;
    if constexpr (is_tensor_v<type> || is_vector_v<type> || is_matrix_v<type>) {
      typename type::value_type s{};
      return s;
    } else
      return type{};
  }

  else if constexpr (Expr_t::kind != boost::yap::expr_kind::call) {

    auto left_t = get_type(boost::yap::left(expr));
    auto right_t = get_type(boost::yap::right(expr));

    return boost::yap::evaluate(
        boost::yap::make_expression<Expr_t::kind>(left_t, right_t));
  } else {
    using namespace boost::hana::literals;
    using ret_t = typename function_return<std::remove_reference_t<decltype(
        boost::yap::value(boost::yap::get(expr, 0_c)))>>::type;
    return ret_t{};
  }
}

} // namespace boost::numeric::ublas::detail

#endif // UBLAS_EXPRESSION_UTILS_HPP
