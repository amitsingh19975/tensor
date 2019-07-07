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
  std::vector<int> sa(5000*500), sb(2500000);
  std::iota(sa.begin(), sa.end(), 1);
  std::iota(sb.begin(), sb.end(), 1);

  tensor_type a{shape{500, 5000}, sa}, b{shape{500, 5000}, sb}, c{shape{500,5000}, 1};

  auto expr = a*b + a*c;

  tensor_type x = expr;
}
