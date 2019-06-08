
#include <boost/core/demangle.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/yap/print.hpp>
#include <boost/yap/yap.hpp>
#include <iostream>
#include <typeinfo>

using namespace boost::numeric::ublas;

int main() {
  tensor<int> t1{shape{5, 5}, 45}, t2{shape{5, 5}, 55};
  auto expr = -tensor<int>{shape{5, 5}, 100} + t1 + t2 + tensor<float>{shape{5, 5}, 4};
  boost::yap::print(std::cout, expr);
  tensor<int> e(expr);
  std::cout << "Extent is :" << e.extents().to_string();
  return 0;
}
