#include <boost/numeric/ublas/tensor.hpp>
#include <boost/yap/print.hpp>
#include "expression_utils.hpp"
#include <boost/numeric/ublas/matrix.hpp>
/**
 *
 * This file is not run as test it is just for my local development quick
 * testings
 *
 */

template <class T>
void preety(T e){
  std::cout<<__PRETTY_FUNCTION__;
}

int main() {
  using namespace boost::numeric::ublas;

  using tensor_type = tensor<int>;
  using value_type = int;
  std::vector<int> sa(50 * 50);
  std::vector<int> sb(2500);
  std::iota(sa.begin(), sa.end(), 1);
  std::iota(sb.begin(), sb.end(), 1);

  tensor_type a{shape{50, 50}, sa};
  tensor_type b{shape{50, 50}, sb};
  tensor_type c{shape{50, 50}, 1};
  matrix<double> sd{50,50,2};


  auto expr = for_each(a * b + a * c, [](auto const &e){return 5.0f;}) - sd;

  using namespace boost::hana::literals;
  auto res = boost::numeric::ublas::detail::get_type(expr);
  //boost::yap::print(std::cout, boost::yap::value(boost::yap::get(expr, 0_c)));
  preety(res);
  tensor_type z = expr;

}
