//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//
//  And we acknowledge the support from all contributors.


#include <iostream>
#include <algorithm>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <boost/test/unit_test.hpp>

#include "utility.hpp"

// BOOST_AUTO_TEST_SUITE ( test_tensor_functions, * boost::unit_test::depends_on("test_tensor_contraction") )
BOOST_AUTO_TEST_SUITE ( test_tensor_functions)


using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::first_order>;


struct fixture
{
    template<size_t... E>
    using static_extents_type = boost::numeric::ublas::static_extents<E...>;
    
    using dynamic_extents_type = boost::numeric::ublas::extents<>;

    fixture()
      : extents {
          dynamic_extents_type{1,1}, // 1
          dynamic_extents_type{2,3}, // 2
          dynamic_extents_type{2,3,1}, // 3
          dynamic_extents_type{4,2,3}, // 4
          dynamic_extents_type{4,2,3,5}} // 5
    {
    }

    std::tuple<
        static_extents_type<1,1>, // 1
        static_extents_type<2,3>, // 2
        static_extents_type<2,3,1>, // 3
        static_extents_type<4,2,3>, // 4
        static_extents_type<4,2,3,5> // 5
    > static_extents{};


    std::vector<dynamic_extents_type> extents;
};



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_prod_vector, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    for_each_tuple(static_extents,[](auto const&, auto& n){                                   
        using extents_type = typename std::decay<decltype(n)>::type;              
        using tensor_type = ublas::static_tensor<value_type, extents_type, layout_type>; 
        using vector_type = typename tensor_type::vector_type;                    
        auto a = tensor_type(n,value_type{2});
        
        for (auto m = 0u; m < n.size(); ++m) {                                    
                                                                                
        auto b = vector_type(n[m], value_type{1});                               
                                                                                
        auto c = ublas::prod(a, b, m + 1);                                       
                                                                                
        for (auto i = 0u; i < c.size(); ++i)                                     
            BOOST_CHECK_EQUAL(c[i], value_type( static_cast< inner_type_t<value_type> >(n[m]) ) * a[i]);                     
        }                                   
    });

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_inner_prod, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    auto const body = [&](auto const& a, auto const& b){
        auto c = ublas::inner_prod(a, b);
        auto r = std::inner_product(a.begin(),a.end(), b.begin(),value_type(0));

        BOOST_CHECK_EQUAL( c , r );
    };

    for_each_tuple(static_extents,[&](auto const&, auto & n){
        using extents_type_1 = typename std::decay<decltype(n)>::type;             
        using extents_type_2 = typename std::decay<decltype(n)>::type;             
        using tensor_type_1 = ublas::static_tensor<value_type, extents_type_1, layout_type>;
        using tensor_type_2 = ublas::static_tensor<value_type, extents_type_2, layout_type>;
        auto a  = tensor_type_1(n,value_type(2));
        auto b  = tensor_type_2(n,value_type(1));
        body(a,b);

    });

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_outer_prod, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    for_each_tuple(static_extents,[&](auto const&, auto const& n1){
        using extents_type_1 = typename std::decay<decltype(n1)>::type;             
        using tensor_type_1 = ublas::static_tensor<value_type, extents_type_1, layout_type>;
        auto a  = tensor_type_1(n1,value_type(2));
        for_each_tuple(static_extents,[&](auto const& J, auto const& n2){
            using extents_type_2 = typename std::decay<decltype(n2)>::type;             
            using tensor_type_2 = ublas::static_tensor<value_type, extents_type_2, layout_type>;
            auto b  = tensor_type_2(n2,value_type(1));
            auto c  = ublas::outer_prod(a, b);

            for(auto const& cc : c)
                BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
            
        });

    });

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_trans, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    auto fac = [](auto const& p){
        auto f = 1ul;
        for(auto i = 1u; i <= p; ++i)
            f *= i;
        return f;
    };

    auto inverse = [](auto const& pi){
        auto pi_inv = pi;
        for(auto j = 0u; j < pi.size(); ++j)
            pi_inv[pi[j]-1] = j+1;
        return pi_inv;
    };

    for_each_tuple(static_extents,[&](auto const&, auto & n){
        using extents_type = typename std::decay<decltype(n)>::type;
        using tensor_type  = ublas::static_tensor<value_type, extents_type,layout_type>;
        auto const p = n.size();
        auto const s = product(n);
        auto aref = tensor_type();
        auto v    = value_type{};
        for(auto i = 0u; i < s; ++i, v+=1)
            aref[i] = v;

        auto pi = std::vector<std::size_t>(p);
        std::iota(pi.begin(), pi.end(), 1);
        
        auto a = ublas::trans( aref, pi );
        
        for(auto i = 0ul; i < a.size(); i++){
            BOOST_CHECK( a[i] == aref[i]  );
        }


        auto const pfac = fac(p);
        auto i = 0u;
        for(; i < pfac-1; ++i) {
            std::next_permutation(pi.begin(), pi.end());
            a = ublas::trans( a, pi );
        }
        std::next_permutation(pi.begin(), pi.end());
        for(; i > 0; --i) {
            std::prev_permutation(pi.begin(), pi.end());
            auto pi_inv = inverse(pi);
            a = ublas::trans( a, pi_inv );
        }

        for(auto j = 0ul; j < a.size(); j++){
            BOOST_CHECK( a[j] == aref[j]  );
        }
    });

}

BOOST_AUTO_TEST_SUITE_END()
