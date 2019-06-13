
#include <boost/core/demangle.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/yap/print.hpp>
#include <boost/yap/yap.hpp>
#include <iostream>
#include <typeinfo>

using namespace boost::numeric::ublas;

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
  tensor<int> t1{shape{5, 5}, 45}, t2{shape{5, 5}, 55};
  matrix<int> m1{5, 5, 44}, m2{5, 6, 9};
  vector<int> v1(5,9), v2(10,8);



  //  // Unary Operators
  //  auto expr = -t1;
  //  auto expr1 = +t2;
  //  auto expr2 = -tensor<int>{shape{5, 5}};
  //  auto expr3 = -tensor<int>{shape{5, 5}};
  //
  //  auto expr4 = -(expr1);
  //  auto expr5 = +expr2;
  //  auto expr6 = -(-t1);
  //  auto expr7 = -(+tensor<int>{shape{5, 5}});
  //
  //  // Binary Operators;
  //  auto expr8 = t1 + t2;
  //  auto expr9 = t1 + tensor<int>{shape{5, 5}, 5};
  //  auto expr10 = tensor<float>{shape{5, 5}, 5.0} + t1;
  //  auto expr11 = tensor<int>{shape{5, 5}, 5} + tensor<float>{shape{5,
  //  5}, 5.0}; auto expr12 = expr5 + tensor<int>{shape{5, 5}}; auto expr13 =
  //  -t1 + t1; auto expr14 = expr3 + tensor<int>{shape{5, 5}, 5}; auto expr15 =
  //  -(t1 + t2) + tensor<int>{shape{5, 5}}; auto expr16 = t1 + expr6; auto
  //  expr17 = t1 + -t1; auto expr18 = tensor<int>{shape{5, 5}} + expr7; auto
  //  expr19 = tensor<int>{shape{5, 5}} + -t1;
  //
  //  auto expr20 = t1 * 20;
  //  auto expr21 = tensor<int>{shape{7, 7}} + 7;
  //  auto i = 78;
  //  auto expr22 = tensor<int>{shape{7, 7}} + i;
  //  auto expr23 = t1 - i;

  //
  //  // boost::yap::print(std::cerr, expr);
  //
  //  tensor<int> t10{expr};
  //  tensor<int> t11{expr1};
  //  tensor<int> t12{expr2};
  //  tensor<int> t13{expr3};
  //  tensor<int> t14{expr4};
  //  tensor<int> t15{expr5};
  //
  //  tensor<int> t16{expr6};
  //  tensor<int> t17{expr7};
  //  tensor<int> t18{expr8};
  //  tensor<int> t19{expr9};
  //  tensor<int> t20{expr10};
  //
  //  tensor<int> t21{expr11};
  //  tensor<int> t22{expr12};
  //  tensor<int> t23{expr13};
  //  tensor<int> t24{expr14};
  //  tensor<int> t25{expr15};
  //
  //  tensor<int> t26{expr16};
  //  tensor<int> t27{expr17};
  //  tensor<int> t28{expr18};
  //  tensor<int> t29{expr19};
  //  tensor<int> t30{expr20};
  //
  //  tensor<int> t31{expr22};
  //  tensor<int> t32{expr21};
  //  tensor<int> t33 = expr23;
}
