
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

  std::cout << "******************* TENSOR TEST********************\n";

  tensor<int> s{shape{5, 1}, 5};
  tensor<int> a{shape{5, 1}, 5};

  auto ten_expr = s - a;

  // boost::yap::print(std::cout, ten_expr);

  std::cout << "******************* MAT-VEC TEST********************\n";
  // Compatibility check
  vector<int> v1{5, 5}, v2{5, 5};

  auto vec_expr = 4 * v1 + v2;

  // std::cerr<<"VECTOR Expr : "<< vector<int>{vec_expr}.data()[0]<<"\n";
  // std::cerr<<"TENSOR Expr : "<<tensor<int>{ten_expr}.data()[0]<<"\n";
  matrix<int> m1{5, 1, 1}, m2{5, 1, 1};

  auto mat_expr = m1 + 3 * m2;

  // todo(coder3101)
  auto mixed_expr = ten_expr + mat_expr - vec_expr;

  //    tensor<int> p(modified);
  //  vector<int> v{vec_expr};
  //  tensor<int> t{ten_expr};
  //  tensor<int> ms{//
  //  std::cout<<t.extents().to_string()<< " " <<ms.extents().to_string()<<"\n";

  std::cout << "******************* XFORM ********************\n";
  auto exx = boost::yap::transform(mixed_expr, detail::transforms::evaluate_ublas_expr<int>{});
   boost::yap::print(std::cerr, exx);
  std::cerr << exx(0)<<" \n";
//  tensor<int> result{mixed_expr};
//  std::cerr << "Extent is " << result.extents().to_string() << "\n";
//  for (auto e : result)
//    std::cerr << " " << e;
//  std::cout << "\n";
  return 0;
}
