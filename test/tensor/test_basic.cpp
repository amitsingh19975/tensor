//#define BOOST_UBLAS_NO_RECURSIVE_OPTIMIZATION

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/yap/print.hpp>
/**
 *
 * This file is not run as test it is just for my local development quick
 * testings
 *
 */

template <class T>
void preety(T e) {
  std::cout << __PRETTY_FUNCTION__;
}

int incremented(const int &input) { return input + 1; }

int decremented(const int &input) { return input - 1; }

int main() {
  using namespace boost::numeric::ublas;

  using tensor_type = tensor<int>;
  using value_type = int;
  std::vector<int> sa(50 * 50);
  std::vector<int> sb(2500);

  std::iota(sa.begin(), sa.end(), 1);
  std::iota(sb.begin(), sb.end(), 1);

  tensor_type a{shape{50, 50, 30}, 3};
  tensor_type b{shape{50, 50}, 5};
  tensor_type c{shape{50, 50}, 1};

  using namespace boost::numeric::ublas::index;
  
  auto expr = c(_i, _j) * b(_i, _k) * a(_i, _j, _k);
  
  boost::yap::print(std::cout, expr);
  //tensor_type result = expr;
  return 0;
}
