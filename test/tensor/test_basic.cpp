#include <boost/numeric/ublas/tensor.hpp>
#include <boost/yap/print.hpp>

/**
 *
 * This file is not run as test it is just for my local development quick
 * testings
 *
 */

int main() {
  using namespace boost::numeric::ublas;

  using tensor_type = tensor<int>;
  std::vector<int> sa(50 * 50);
  std::vector<int> sb(2500);
  std::iota(sa.begin(), sa.end(), 1);
  std::iota(sb.begin(), sb.end(), 1);

  tensor_type a{shape{50, 50}, sa};
  tensor_type b{shape{50, 50}, sb};
  tensor_type c{shape{50, 50}, 1};

  auto expr = a * b + a * c;

  auto l = [](auto r) { return r + 1; };
  auto new_expr = for_each(a, [](auto const &s) { return 5.0f + s; });

  // auto new_expr2 = boost::yap::transform(new_expr,
  // detail::transforms::at_index{5});

  // boost::yap::print(std::cout, new_expr);


  // std::cout<<boost::yap::evaluate(new_expr2);

  tensor_type z = new_expr;
  assert((bool)(z == 5+a));
  z = for_each(a + b, [](auto const &e){return sqrt(e);});

//  auto kl = boost::yap::make_terminal(l);
//  auto e = kl(56);
//  boost::yap::print(std::cout, e);

  // std::cout<<z;
}
