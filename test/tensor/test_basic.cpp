//#define BOOST_UBLAS_NO_RECURSIVE_OPTIMIZATION

#include "expression_utils.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/yap/print.hpp>
/**
 *
 * This file is not run as test it is just for my local development quick
 * testings
 *
 */

template <class T> void preety(T e) { std::cout << __PRETTY_FUNCTION__; }

int main() {
  using namespace boost::numeric::ublas;

  using tensor_type = tensor<int>;
  using value_type = int;
  std::vector<int> sa(50 * 50);
  std::vector<int> sb(2500);
  std::iota(sa.begin(), sa.end(), 1);
  std::iota(sb.begin(), sb.end(), 1);

  tensor_type a{shape{50, 50}, sa};
  tensor_type b{shape{50, 50}, 5};
  tensor_type c{shape{50, 50}, 1};
  matrix<double> sd{50, 50, 2};

  auto expr = (a * b +  c * a) + ( a * b +  b * 1);
  auto xform = detail::transforms::apply_distributive_law{};
  auto optimized = boost::yap::transform(expr, xform);
  if (xform.usable)
    boost::yap::print(std::cout, optimized);
  else
    std::cout << "Optimize Failed: Falling back to use old expression";
  return 0;
}
