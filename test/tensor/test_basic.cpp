#include <boost/numeric/ublas/tensor.hpp>
#include <boost/yap/print.hpp>

/**
 *
 * This file is not run as test it is just for my local developement quick
 * testings
 *
 */

int main() {
  using namespace boost::numeric::ublas;

  using tensor_type = tensor<int>;
  std::vector<int> sa(5000 * 500);
  std::vector<int> sb(2500000);
  std::iota(sa.begin(), sa.end(), 1);
  std::iota(sb.begin(), sb.end(), 1);

  tensor_type a{shape{500, 5000}, sa};
  tensor_type b{shape{500, 5000}, sb};
  tensor_type c{shape{500, 5000}, 1};

  auto expr = a * b + a * c;

  auto l = [](auto r) { return r + 1; };
  auto new_expr = for_each(expr, [](auto const &s) { return 5; });

  // auto new_expr2 = boost::yap::transform(new_expr,
  // detail::transforms::at_index{5});

  boost::yap::print(std::cout, new_expr);


  // std::cout<<boost::yap::evaluate(new_expr2);

  tensor_type z = new_expr;

  // std::cout<<z;
}
