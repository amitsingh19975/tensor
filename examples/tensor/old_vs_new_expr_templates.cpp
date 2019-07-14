//  Copyright (c) 2019-2020
//  Mohammad Ashar Khan, ashar786khan@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google in producing this work
//  which started as a Google Summer of Code project.

#include <boost/numeric/ublas/tensor.hpp>
#include <cmath>
#include <iostream>
#include <type_traits>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCDFAInspection"

template<class T1, class T2>
decltype(auto) make_squared_difference(T1 &&t1, T2 &&t2) {
  static_assert(boost::numeric::ublas::is_tensor_v<std::remove_reference_t<T1>> &&
      boost::numeric::ublas::is_tensor_v<std::remove_reference_t<T2>>, "Arguments must be tensor type");

  return t1 * t1 - t2 * t2; // squared_difference;
}

#pragma clang diagnostic pop

int main() {

  using tensor_int = boost::numeric::ublas::tensor<int>;
  using tensor_float = boost::numeric::ublas::tensor<float>;
  using shape = boost::numeric::ublas::shape;
  using boost::numeric::ublas::for_each;

  {
    /*
     * Freedom to copy and move expression objects. This was not allowed in the old expression templates
     * as the copy and move of tensor_expression was deleted implicitly.
     * This allows us to make functions return tensor_expressions.
     */

    tensor_int a{shape{15, 15, 4}, 6};
    tensor_int b{shape{15, 15, 4}, 4};

    auto expr = make_squared_difference(a, b);

    std::cout << "Squared Difference at index 5 is : " << expr(5) << "\n";
    assert(expr(5) == 20);
  }

  {

    /*
     * Freedom of scalars is yet another new flexibility that comes with the new expression templates. It was not
     * possible in the old templates. Also we do not give a hard error unless you try to evaluate an expression that cannot
     * be evaluated.
     */

    struct zero_like{
      auto operator+(const int s){
        return 0+s;
      }
    };

    struct one_like{}; // Does not have any overload with int. one_like + int is not defined.


    tensor_int a {shape{4,2}, 5};
    tensor_int b {shape{4,2}, 6};
    zero_like c;
    one_like d;

    auto expr = c + a; // zero like has an overload with int hence compiles and runs.

    auto expr2 = b + d; // one like has no overload with int yet compiles and runs as long as this is not evaluated.

    tensor_int x = expr; // finds + overload and compiles.

    // tensor_int y = expr2; // compilation fails as no overload for one_like+int exist.
    // Error Message:
    // error: no match for ‘operator+’ (operand types are ‘const int’ and ‘main()::one_like’)

    assert(x(0) == 5);
  }

  {
    /*
     * Freedom of lazily applying lamdas to an expression. While older expression template did had such a capability,
     * It was not very accessible since, it was wrapped into detail namespace and did not had any documentations governing 
     * the use of it. In this new expression templates we however give the end-user a flexibility to do such thing in a modern goodlooking interface. 
     */

    tensor_int a = tensor_int{shape{5,5,3}, 56};
    tensor_int b = tensor_int{shape{5,5,3}, 44};
    auto expr = for_each( a+b ,[] (auto const& val) { return sqrt(val); }); // ==> sqrt(a+b); All Lazy
    tensor_float x = expr;

    assert(x(0) == 10.0f);
    assert(x == 10.0f);

  }

  {
    /*
     * Lazy relational operators. The new expression templates does not only provides lazy arithematic but even relational operators are lazy
     * All the 6 relational operators are lazy and only evaluates to an boolean type once required. They all are implicitly converted to bool type
     * as long as they have at least one relational operator.
     */

     tensor_int a = tensor_int{shape{5,5,3}, 56};
     tensor_int b = tensor_int{shape{5,5,3}, 44};

     auto expr = a - 12 == b;
     // expr is still not evaluated. Maybe we don't need to know the complete result. So why should we waste evaluating everything.
    
    if(expr) // implicitly converted to bool. Causing an evaluation
      std::cout<<"Yah.. Expr evaluated to true\n";

    // The whole point of lazying the relational operator is to give user the freedom to only use unevaluated bool results in the expression to write effificent code.
    // Consider this example:

    auto iotafill = [](tensor_int &e, int v){ std::iota(e.begin(), e.end(), v); };

    tensor_int c = tensor_int{shape{5,5,3}, 96};
    
    iotafill(a,1);
    iotafill(b,1);
    iotafill(c,46);

    tensor_int d = (a == b) * c;
    // d is the tensor that has:
    // d(x) == c(x) iff a(x) == b(x); for all valid x; else d(x) == 0

    // Making the relational operator lazy makes us evaluate this expression in the most efficient possible way.
  }

}

