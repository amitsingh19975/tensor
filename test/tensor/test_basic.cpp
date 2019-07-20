#include <boost/numeric/ublas/tensor.hpp>
#include <boost/yap/print.hpp>

/**
 *
 * This file is not run as test it is just for my local development quick
 * testings
 *
 */

template <class T>
auto func(T s){
  return s;
}

auto foobar(int const &e){
    return e+1;
}

int main() {
  using namespace boost::numeric::ublas;

  using tensor_type = tensor<int>;
  using value_type = int;
//  std::vector<int> sa(50 * 50);
//  std::vector<int> sb(2500);
//  std::iota(sa.begin(), sa.end(), 1);
//  std::iota(sb.begin(), sb.end(), 1);
//
//  tensor_type a{shape{50, 50}, sa};
//  tensor_type b{shape{50, 50}, sb};
//  tensor_type c{shape{50, 50}, 1};
//
//  auto expr = a * b + a * c;

  auto e = shape{{5,5,3}};
  auto t = tensor_type(e);
  auto v = value_type{0};

  for (auto &tt : t) {
    tt = v;
    v += value_type{1};
  }

  auto t_copy1 = t;
  auto t_copy2 = t;

  auto d = t;
  std::reverse(d.begin(), d.end());

  auto terminal_tensor = boost::yap::make_terminal(d);

//  auto transformed_expr1 = for_each(terminal_tensor, [](auto const& ep){return 5.0f;});
//  auto transformed_expr2 = for_each(terminal_tensor, [](auto const& ep){return 5.0f+ep;});
//  auto transformed_expr3 = for_each(terminal_tensor, [](auto const& ep){return ep*ep;});
//  auto transformed_expr4 = for_each(terminal_tensor, [](auto const& ep){return sqrt(ep);});

  auto transformed_expr5 = for_each(d, func<int>);
//  auto transformed_expr6 = for_each(d, [](auto const& ep){return 5.0f+ep;});
//  auto transformed_expr7 = for_each(d, [](auto const& ep){return ep*ep;});
//  auto transformed_expr8 = for_each(d, [](auto const& ep){return sqrt(ep);});

tensor_type  sas = transformed_expr5;



}
