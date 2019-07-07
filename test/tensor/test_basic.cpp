
#include <algorithm>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/yap/print.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <complex>

#include "expression_transforms_traits.hpp"

/**
 *
 * This file is not run as test it is just for my local developement quick
 * testings
 *
 */

template <class T> constexpr std::string_view type_name() {
  using namespace std;
#ifdef __clang__
  string_view p = __PRETTY_FUNCTION__;
  return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
  string_view p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
  return string_view(p.data() + 36, p.size() - 36 - 1);
#else
  return string_view(p.data() + 49, p.find(';', 49) - 49);
#endif
#elif defined(_MSC_VER)
  string_view p = __FUNCSIG__;
  return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}


int main() {
  using namespace boost::numeric::ublas;

  using tensor_type = tensor<int>;
  std::vector<int> sa(50*500), sb(25000);
  std::iota(sa.begin(), sa.end(), 1);
  std::iota(sb.begin(), sb.end(), 1);

  tensor_type a{shape{50, 500}, sa}, b{shape{50, 500}, sb}, c{shape{50,500}, 1};

  auto expr = (a*b + a*c) + 1;

  auto sub_expr = boost::yap::transform(expr, detail::transforms::at_index{2});

  //static_assert(detail::transforms::is_multiply_operand<decltype(sub_expr)>::value);

  boost::yap::print(std::cout, boost::yap::transform(sub_expr, detail::transforms::apply_distributive_law{}));

  //std::cout<<type_name<decltype(sub_expr)>();


  //expr(3);
}
