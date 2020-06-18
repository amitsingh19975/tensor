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
BOOST_AUTO_TEST_SUITE ( test_fixed_rank_tensor_functions)


using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::first_order>;


struct fixture
{
    template<size_t R>
    using fixed_rank_extents_type = boost::numeric::ublas::extents<R>;
    
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
        fixed_rank_extents_type<2>,
        fixed_rank_extents_type<2>,
        fixed_rank_extents_type<3>,
        fixed_rank_extents_type<3>,
        fixed_rank_extents_type<4>
    > fixed_rank_extents{
        {1,1}, // 1
        {2,3}, // 2
        {2,3,1}, // 3
        {4,2,3}, // 4
        {4,2,3,5} // 5
    };


    std::vector<dynamic_extents_type> extents;

};



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_prod_vector, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    for_each_tuple(fixed_rank_extents,[](auto const&, auto & n){                                   
        using extents_type = typename std::decay<decltype(n)>::type;              
        using tensor_type = ublas::fixed_rank_tensor<value_type, extents_type::_size, layout_type>; 
        using vector_type = typename tensor_type::vector_type;                    
        auto a = tensor_type(n, value_type{2});
        
        for (auto m = 0u; m < n.size(); ++m) {
            auto b = vector_type(n[m], value_type{1});                               
                                                                                    
            auto c = ublas::prod(a, b, m + 1);                                       
                                                                                    
            for (auto i = 0u; i < c.size(); ++i)                                     
                BOOST_CHECK_EQUAL(c[i], value_type( static_cast< inner_type_t<value_type> >(n[m]) ) * a[i]);                     
        }                                   
    });

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_prod_matrix, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    for_each_tuple(fixed_rank_extents,[](auto const&, auto & n){                                                                   
        using extents_type = typename std::decay<decltype(n)>::type;             
        using tensor_type = ublas::fixed_rank_tensor<value_type, extents_type::_size, layout_type>; 
        using matrix_type = typename tensor_type::matrix_type;                   
        auto a = tensor_type(n, value_type{2});                     
        for (auto m = 0u; m < n.size(); ++m) {                                   
                                                                                
            auto b = matrix_type  ( n[m], n[m], value_type{1} );                     
                                                                                    
            auto c = ublas::prod(a, b, m + 1);                                       
                                                                                    
            for (auto i = 0u; i < c.size(); ++i)                                     
                BOOST_CHECK_EQUAL(c[i], value_type( static_cast< inner_type_t<value_type> >(n[m]) ) * a[i]);                    
        }                                   
    });

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_prod_tensor_1, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    auto const body1 = [](auto const& a, auto const& b){

        using lhs_tensor_type = std::decay_t<decltype(a)>;
        using extents_type = typename lhs_tensor_type::extents_type;
        constexpr auto const pa = extents_type::_size;

        boost::mp11::mp_for_each< boost::mp11::mp_iota_c< pa > >([&](auto iter_i){
            constexpr auto const I = decltype(iter_i)::value;

            auto phi = std::array<std::size_t,I>{};
            std::iota(phi.begin(), phi.end(), 1ul);

            auto c = ublas::prod(a, b, phi);                                         

            auto acc = value_type(1);
            for(auto j = 0; j < I; ++j ){
                acc *= value_type( static_cast< inner_type_t<value_type> >( a.extents().at(phi.at(j) - 1) ) );
            }
                                                                                    
            for (auto i = 0ul; i < c.size(); ++i)                                    
                BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);

        });       
    };

    auto const body2 = [](auto const& a, auto const& b){

        using rhs_tensor_type = std::decay_t<decltype(b)>;
        using extents_type = typename rhs_tensor_type::extents_type;
        constexpr auto const pa = extents_type::_size;

        boost::mp11::mp_for_each< boost::mp11::mp_iota_c< pa > >([&](auto iter_i){
            constexpr auto const I = decltype(iter_i)::value;

            auto phi = std::array<std::size_t,I>{};
            std::iota(phi.begin(), phi.end(), 1ul);

            auto c = ublas::prod(a, b, phi);                                         

            auto acc = value_type(1);
            for(auto j = 0; j < I; ++j ){
                acc *= value_type( static_cast< inner_type_t<value_type> >( a.extents().at(phi.at(j) - 1) ) );
            }
                                                                                    
            for (auto i = 0ul; i < c.size(); ++i)                                    
                BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);

        });       
    };



    for_each_tuple(fixed_rank_extents,[&](auto const&, auto & n){                                                                   
        auto n1 = n;                                                              
        auto n2 = n;                                                              
        using extents_type_1 = typename std::decay<decltype(n1)>::type;           
        using extents_type_2 = typename std::decay<decltype(n2)>::type;           
        using tensor_type_1 =                                                     
            ublas::fixed_rank_tensor<value_type, extents_type_1::_size, layout_type>;               
        using tensor_type_2 =                                                     
            ublas::fixed_rank_tensor<value_type, extents_type_2::_size, layout_type>;               
        auto a = tensor_type_1(n1, value_type{2});                                
        auto b = tensor_type_2(n2, value_type{3});                                
        body1(a,b);      
    });

    for_each_tuple(fixed_rank_extents,[&](auto iter, auto & n){                                                                   
        using I = decltype(iter);
        auto n1 = n;
        auto n2 = extents[I::value];
        using extents_type_1 = typename std::decay<decltype(n1)>::type;                      
        using tensor_type_1 =                                                     
            ublas::fixed_rank_tensor<value_type, extents_type_1::_size, layout_type>;               
        using tensor_type_2 =                                                     
            ublas::dynamic_tensor<value_type, layout_type>;               
        auto a = tensor_type_1(n1, value_type{2});                                
        auto b = tensor_type_2(n2, value_type{3});                                
        body1(a,b);      
    });

    for_each_tuple(fixed_rank_extents,[&](auto iter, auto & n){ 
        using I = decltype(iter);
        auto n1 = extents[I::value];                                                              
        auto n2 = n;                                                                      
        using extents_type_2 = typename std::decay<decltype(n2)>::type;           
        using tensor_type_1 =                                                     
            ublas::dynamic_tensor<value_type, layout_type>;               
        using tensor_type_2 =                                                     
            ublas::fixed_rank_tensor<value_type, extents_type_2::_size, layout_type>;               
        auto a = tensor_type_1(n1, value_type{2});                                
        auto b = tensor_type_2(n2, value_type{3});                                
        body2(a,b);      
    });

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_prod_tensor_2, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    auto compute_factorial = [](auto const& p){
        auto f = 1ul;
        for(auto i = 1u; i <= p; ++i)
            f *= i;
        return f;
    };

    auto permute_extents = [](auto const& pi, auto const& na){
        auto nb = na;
        assert(pi.size() == na.size());
        for(auto j = 0u; j < pi.size(); ++j)
            nb[pi[j]-1] = na[j];
        return nb;
    };

    for_each_tuple(fixed_rank_extents,[&](auto const&, auto & n){
        auto na = n;                                                              
        using extents_type_1 = typename std::decay<decltype(na)>::type;             
        
        constexpr auto const pa = extents_type_1::_size;
        
        using tensor_type_1 = ublas::fixed_rank_tensor<value_type, pa, layout_type>; 
        auto a = tensor_type_1(na, value_type{2});                                  
        
        auto pi = std::array<std::size_t,pa>{};
        constexpr auto const fac = compute_factorial(pa);
        std::iota(pi.begin(), pi.end(), 1ul); 

        for (auto f = 0ul; f < fac; ++f) {                                        
            auto nb = permute_extents(pi, na); 
            using tensor_type_2 = ublas::dynamic_tensor<value_type, layout_type>;                                                     
            auto b = tensor_type_2(nb, value_type{3});                                 

            boost::mp11::mp_for_each< boost::mp11::mp_iota_c<pa + 1> >([&](auto iter){
                constexpr auto const q = decltype(iter)::value;

                auto phia = std::array<std::size_t,q>{};                              
                auto phib = std::array<std::size_t,q>{};                              
                                                                                    
                std::iota(phia.begin(), phia.end(), 1ul);
                std::transform(phia.begin(), phia.end(), phib.begin(),
                            [&pi](std::size_t i) { return pi.at(i - 1); });
                                                                                    
                auto c = ublas::prod(a, b, phia, phib);                               
                                                                                    
                auto acc = value_type(1);                                             
                for (auto i = 0ul; i < q; ++i)                                        
                acc *= value_type( static_cast< inner_type_t<value_type> >( a.extents().at(phia.at(i) - 1) ) );
                                                                                    
                for (auto i = 0ul; i < c.size(); ++i)                                 
                BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);

            });
            std::next_permutation(pi.begin(), pi.end());                             
        }  
    });

    for_each_tuple(fixed_rank_extents,[&](auto const&, auto & n){
        auto na = n;                                                              
        using extents_type_1 = typename std::decay<decltype(na)>::type;    

        constexpr auto const pa = extents_type_1::_size;   

        using tensor_type_1 = ublas::fixed_rank_tensor<value_type, pa, layout_type>; 
        auto a = tensor_type_1(na, value_type{2});                                  
                                                                                
        auto pi = std::array<std::size_t,pa>{};
        auto fac = compute_factorial(pa);                                         
        std::iota(pi.begin(), pi.end(), 1);                                       
                                                                                
        for (auto f = 0ul; f < fac; ++f) {                                        
            auto nb = permute_extents(pi, na); 
            
            using extents_type_2 = typename std::decay<decltype(nb)>::type;  
            using tensor_type_2 = ublas::fixed_rank_tensor<value_type, extents_type_2::_size, layout_type>;                                                     
            auto b = tensor_type_2(nb, value_type{3});                                 
                                                                                    
            boost::mp11::mp_for_each< boost::mp11::mp_iota_c<pa + 1> >([&](auto iter){
                constexpr auto const q = decltype(iter)::value;

                auto phia = std::array<std::size_t,q>{};                              
                auto phib = std::array<std::size_t,q>{};                              
                                                                                    
                std::iota(phia.begin(), phia.end(), 1ul);
                std::transform(phia.begin(), phia.end(), phib.begin(),
                            [&pi](std::size_t i) { return pi.at(i - 1); });
                                                                                    
                auto c = ublas::prod(a, b, phia, phib);                               
                                                                                    
                auto acc = value_type(1);                                             
                for (auto i = 0ul; i < q; ++i)                                        
                acc *= value_type( static_cast< inner_type_t<value_type> >( a.extents().at(phia.at(i) - 1) ) );
                                                                                    
                for (auto i = 0ul; i < c.size(); ++i)                                 
                BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);

            });                                                                        
                                                                                    
            std::next_permutation(pi.begin(), pi.end());                             
        }  
    });

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_inner_prod, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    auto const body = [&](auto const& a, auto const& b){
        auto c = ublas::inner_prod(a, b);
        auto r = std::inner_product(a.begin(),a.end(), b.begin(),value_type(0));

        BOOST_CHECK_EQUAL( c , r );
    };

    for_each_tuple(fixed_rank_extents,[&](auto const&, auto & n){
        using extents_type_1 = typename std::decay<decltype(n)>::type;             
        using extents_type_2 = typename std::decay<decltype(n)>::type;             
        using tensor_type_1 = ublas::fixed_rank_tensor<value_type, extents_type_1::_size, layout_type>;
        using tensor_type_2 = ublas::fixed_rank_tensor<value_type, extents_type_2::_size, layout_type>;
        auto a  = tensor_type_1(n, value_type(2));
        auto b  = tensor_type_2(n, value_type(1));
        body(a,b);

    });

    for_each_tuple(fixed_rank_extents,[&](auto iter, auto & n){
        constexpr auto const I = decltype(iter)::value;
        using extents_type_1 = typename std::decay<decltype(n)>::type;                         
        using tensor_type_1 = ublas::fixed_rank_tensor<value_type, extents_type_1::_size, layout_type>;
        using tensor_type_2 = ublas::dynamic_tensor<value_type, layout_type>;
        auto a  = tensor_type_1(n, value_type(2));
        auto b  = tensor_type_2(extents[I], value_type(1));
        body(a,b);

    });

    for_each_tuple(fixed_rank_extents,[&](auto iter, auto & n){
        constexpr auto const I = decltype(iter)::value;
        using extents_type_2 = typename std::decay<decltype(n)>::type;             
        using tensor_type_1 = ublas::dynamic_tensor<value_type, layout_type>;
        using tensor_type_2 = ublas::fixed_rank_tensor<value_type, extents_type_2::_size, layout_type>;
        auto a  = tensor_type_1(extents[I], value_type(2));
        auto b  = tensor_type_2(n, value_type(1));
        body(a,b);

    });

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_outer_prod, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    for_each_tuple(fixed_rank_extents,[&](auto const&, auto const& n1){
        using extents_type_1 = typename std::decay<decltype(n1)>::type;             
        using tensor_type_1 = ublas::fixed_rank_tensor<value_type, extents_type_1::_size, layout_type>;
        auto a  = tensor_type_1(n1, value_type(2));
        for_each_tuple(fixed_rank_extents,[&](auto const&, auto const& n2){
            using extents_type_2 = typename std::decay<decltype(n2)>::type;             
            using tensor_type_2 = ublas::fixed_rank_tensor<value_type, extents_type_2::_size, layout_type>;
            auto b  = tensor_type_2(n2, value_type(1));
            auto c  = ublas::outer_prod(a, b);

            for(auto const& cc : c)
                BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
            
        });

    });

    for_each_tuple(fixed_rank_extents,[&](auto iter, auto const& n1){ 
        constexpr auto const I = decltype(iter)::value;
        using tensor_type_1 = ublas::dynamic_tensor<value_type, layout_type>;
        auto a  = tensor_type_1(extents[I], value_type(2));
        for_each_tuple(fixed_rank_extents,[&](auto const&, auto const& n2){
            using extents_type_2 = typename std::decay<decltype(n2)>::type;             
            using tensor_type_2 = ublas::fixed_rank_tensor<value_type, extents_type_2::_size, layout_type>;
            auto b  = tensor_type_2(n2, value_type(1));
            auto c  = ublas::outer_prod(a, b);

            for(auto const& cc : c)
                BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
        });

    });

    for_each_tuple(fixed_rank_extents,[&](auto const&, auto const& n1){
        using extents_type_1 = typename std::decay<decltype(n1)>::type;             
        using tensor_type_1 = ublas::fixed_rank_tensor<value_type, extents_type_1::_size, layout_type>;
        auto a  = tensor_type_1(n1, value_type(2));
        for(auto const& n2 : extents){           
            using tensor_type_2 = ublas::dynamic_tensor<value_type, layout_type>;
            auto b  = tensor_type_2(n2, value_type(1));
            auto c  = ublas::outer_prod(a, b);

            for(auto const& cc : c)
                BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
        }

    });

}

BOOST_AUTO_TEST_SUITE_END()
