#include <boost/numeric/ublas/tensor.hpp>
#include <iostream>
#include <typeinfo>
#include <boost/core/demangle.hpp>
#include <boost/yap/print.hpp>

using namespace boost::numeric::ublas;

int main(){
    tensor<int> s{shape{3,5}, 1};
    auto res = s + s + boost::yap::as_expr(s);
    //auto res = s+s;
    boost::yap::print(std::cout, res);
    //std::cout<<boost::core::demangle(typeid(res).name());
    //std::cout<<s.at(1,1,1) + s.at(2,2);
    return 0;
}