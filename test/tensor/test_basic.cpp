#include <boost/numeric/ublas/tensor.hpp>
#include <iostream>
#include <typeinfo>
#include <type_traits>
#include <boost/core/demangle.hpp>
#include <boost/yap/print.hpp>

using namespace boost::numeric::ublas;
using namespace boost::hana::literals;

struct S{
    S() = default;

};

int main(){
    tensor<int> s{shape{3,5,8}, 5};
    s.is_extent_static = true;
    auto res = s + s;
    std::cout<<std::boolalpha<<res.is_extent_static<<"\n";
    // std::cout<<s.elements[0_c].extents_.product();
    // boost::yap::print(std::cout, res);
    // std::cout<<std::boolalpha;
    std::cout<<res(2);
    // std::cout<<std::is_lvalue_reference<decltype(res)>::value<<" "<<std::is_rvalue_reference<decltype(res)>::value<<" "<<std::is_const<decltype(res)>::value;    //auto r = res(2);
    //std::cout<<boost::core::demangle(typeid(r).name())<<" "<<r;
    //std::cout<<s.at(1,1,1) + s.at(2,2);
    return 0;
}