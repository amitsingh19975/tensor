
#include <boost/numeric/ublas/tensor.hpp>
#include <iostream>
#include <typeinfo>
#include <boost/core/demangle.hpp>
#include <boost/yap/print.hpp>

using namespace boost::numeric::ublas;

int main()
{
    tensor<int> s{shape{3, 3}, 5};
    tensor<float> a{shape{3,3}, 5.2};
    
    // Mix and match allowed. Unless there is a +, -, * and / operator overload. Float + int is valid
    auto expr = s * s + (s + 52 / s) - s * 2;
    
    //Evalutate to tensor.
    tensor<float> ten1(expr);
    
    auto casted_tensor = boost::numeric::ublas::static_tensor_cast<int>(expr); 
    // We have dynamic and reinterpret cast as well.
    // This cast is eager and is only performed on tensor l-value or r-value
    // A Work is in progress for the implementation of lazy cast that takes an expression.
    
    
    auto ss = expr.eval<int>(); 
    // This is similar to Eigen. It retuns a new tensor after evaluating the expression.
    // I just implemented it because Eigen has something like this.
    // This eval<..> is a template function that takes all three parameters and returns a tensor build out of those parameters.
    // This therefore can be used to change layout of a tensor or dtype as well.
    
    boost::yap::print(std::cout, expr);
    // Just a proof that expr is really a YAP expression
    
    std::cout<<boost::core::demangle(typeid(casted_tensor.at(1,1)).name())<<" \n";
    // We did really changed the type to int from double with the cast.
    
    return 0;
    
}

/*
 * Areas of Work in Progress.
 * Last expression templates were inherited from ublas_expression has hence was easily interchangiably used with martix and vector expression.
 * This new YAP expression however does things in complete different way and hence it is still WIP on how to make YAP expression work with matrix
 * and vector expression.
 */
