//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numeric/ublas/tensor/detail/extents_helper.hpp>
#include <boost/test/unit_test.hpp>
#include <map>
#include <numeric>
#include "utility.hpp"
#include <vector>

BOOST_AUTO_TEST_SUITE(test_basic_extents_impl)

template <size_t... E>
using extents = boost::numeric::ublas::detail::basic_extents_impl<0, E...>;

BOOST_AUTO_TEST_CASE(test_basic_extents_impl_ctor) {
  using namespace boost::numeric;

  auto e0 = extents<>{};
  BOOST_CHECK(e0.empty());
  BOOST_CHECK_EQUAL(e0.size(), 0);

  auto e1 = extents<1>{};
  BOOST_CHECK(!e1.empty());
  BOOST_CHECK_EQUAL(e1.size(), 1);

  auto e2 = extents<1,2,3>{};
  BOOST_CHECK(!e2.empty());
  BOOST_CHECK_EQUAL(e2.size(), 3);
}

struct fixture {
  fixture() = default;
  extents<> e0{};                               // 0
  extents<1, 1, 1, 1> e1{};                     // 1
  extents<1, 2, 3> e2{};                        // 2
  extents<4, 2, 3> e3{};                        // 3
  extents<4, 2, 1, 3> e4{};                     // 4
  extents<1, 4, 2, 1, 3, 1> e5{};               // 5
  
  std::tuple<
    extents<>
  > rank_0_extents;

  std::tuple<
    extents<1>,
    extents<2>,
    extents<3>,
    extents<4>,
    extents<5>,
    extents<6>
  > rank_1_extents;

  std::tuple<
    extents<1,1>,
    extents<2,2>,
    extents<3,3>,
    extents<4,4>,
    extents<5,5>,
    extents<6,6>
  > rank_2_extents;

  std::tuple<
    extents<1>,
    extents<1,1>,
    extents<1,1,1>,
    extents<1,1,1,1>,
    extents<1,1,1,1,1>,
    extents<1,1,1,1,1,1>
  > scalars;

  std::tuple<
    extents<1,2>,
    extents<1,3,1>,
    extents<1,4,1,1>,
    extents<5,1,1,1,1>,
    extents<6,1,1,1,1,1>
  > vectors;

  std::tuple<
    extents<2,2>,
    extents<3,2>,
    extents<2,3>,
    extents<3,3,1>,
    extents<2,3,1>,
    extents<3,2,1>,
    extents<4,4,1,1>,
    extents<3,4,1,1>,
    extents<4,3,1,1>,
    extents<5,5,1,1,1>,
    extents<6,6,1,1,1,1>
  > matrices;

  std::tuple<
    extents<3,3,3>,
    extents<2,3,3>,
    extents<3,2,3>,
    extents<3,2,2>,
    extents<4,4,4,1>,
    extents<3,4,4,1>,
    extents<3,4,1,4>,
    extents<4,3,5,1>,
    extents<4,3,3,3>,
    extents<5,5,5,1,1>,
    extents<5,5,1,5,1>,
    extents<6,6,6,1,1,1>,
    extents<6,6,1,6,1,1>,
    extents<6,6,1,1,6,1>,
    extents<6,6,1,1,1,6>
  > tensors;
};

BOOST_FIXTURE_TEST_CASE(test_basic_extents_impl_access, fixture,
                        *boost::unit_test::label("basic_extents_impl") *
                            boost::unit_test::label("access")) {
  using namespace boost::numeric;

  BOOST_CHECK_EQUAL(e0.size(), 0);
  BOOST_CHECK(e0.empty());

  BOOST_REQUIRE_EQUAL(e1.size(), 4);
  BOOST_REQUIRE_EQUAL(e2.size(), 3);
  BOOST_REQUIRE_EQUAL(e3.size(), 3);
  BOOST_REQUIRE_EQUAL(e4.size(), 4);
  BOOST_REQUIRE_EQUAL(e5.size(), 6);

  BOOST_CHECK_EQUAL(e1[0], 1);
  BOOST_CHECK_EQUAL(e1[1], 1);
  BOOST_CHECK_EQUAL(e1[2], 1);
  BOOST_CHECK_EQUAL(e1[3], 1);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,0, 0, 0, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,1, 0, 0, 0), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,0, 1, 0, 0), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,0, 0, 1, 0), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,0, 0, 0, 1), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,1, 1, 1, 1), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,1, 1, 1, -1), false);

  BOOST_CHECK_EQUAL(e2[0], 1);
  BOOST_CHECK_EQUAL(e2[1], 2);
  BOOST_CHECK_EQUAL(e2[2], 3);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,0, 0, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,1, 0, 0), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,0, 1, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,0, 0, 1), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,0, 3, 0), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,1, 1, 1), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,0, 1, 1), true);

  BOOST_CHECK_EQUAL(e3[0], 4);
  BOOST_CHECK_EQUAL(e3[1], 2);
  BOOST_CHECK_EQUAL(e3[2], 3);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,0, 0, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,3, 0, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,0, 1, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,0, 0, 1), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,0, 3, 0), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,1, 1, 1), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,0, 1, 1), true);

  BOOST_CHECK_EQUAL(e4[0], 4);
  BOOST_CHECK_EQUAL(e4[1], 2);
  BOOST_CHECK_EQUAL(e4[2], 1);
  BOOST_CHECK_EQUAL(e4[3], 3);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,0, 0, 0, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,1, 0, 0, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,0, 1, 0, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,0, 0, 1, 0), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,0, 0, 0, 1), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,1, 1, 1, 1), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,1, 1, 1, -1), false);

  BOOST_CHECK_EQUAL(e5[0], 1);
  BOOST_CHECK_EQUAL(e5[1], 4);
  BOOST_CHECK_EQUAL(e5[2], 2);
  BOOST_CHECK_EQUAL(e5[3], 1);
  BOOST_CHECK_EQUAL(e5[4], 3);
  BOOST_CHECK_EQUAL(e5[5], 1);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 0, 0, 0, 0, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,1, 0, 0, 0, 0, 0), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 1, 0, 0, 0, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 0, 1, 0, 0, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 0, 0, 1, 0, 0), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 0, 0, 0, 1, 0), true);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 0, 0, 0, 0, 1), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,1, 1, 1, 1, 1, 1), false);
  // BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,1, 1, 1, -1, 1), false);

  for_each_tuple(rank_0_extents,[](auto const& I, auto const& e){
    for(auto i = 0; i < e.size(); i++){
      BOOST_CHECK_EQUAL(e[i],1);
    }
  });

  for_each_tuple(rank_1_extents,[](auto const& I, auto const& e){
    for(auto i = 0; i < e.size(); i++){
      BOOST_CHECK_EQUAL(e[i],I + 1);
    }
  });

  for_each_tuple(rank_2_extents,[](auto const& I, auto const& e){
    for(auto i = 0; i < e.size(); i++){
      BOOST_CHECK_EQUAL(e[i],I + 1);
    }
  });


  for_each_tuple(scalars,[](auto const& I, auto const& e){
    for(auto i = 0; i < e.size(); i++){
      BOOST_CHECK_EQUAL(e[i],1);
    }
  });

  // for_each_tuple(rank_1_extents,[](auto const& I, auto const& e){
  //   BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,-100),false);
  //   BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,100),false);
  //   BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,0),true);
  // });

  // for_each_tuple(rank_2_extents,[](auto const& I, auto const& e){
  //   BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,-100,0),false);
  //   BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,100,0),false);
  // });


  auto product_lm = [](auto const& e){
    auto p = 1;
    for(auto i = 0; i < e.size();i++){
      p *= e.at(i);
    }
    return p;
  };

  for_each_tuple(rank_1_extents,[&](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(e.product(),product_lm(e));
  });

  for_each_tuple(rank_2_extents,[&](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(e.product(),product_lm(e));
  });

  for_each_tuple(scalars,[&](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(e.product(),product_lm(e));
  });

}

BOOST_FIXTURE_TEST_CASE(test_basic_extents_impl_copy_ctor, fixture,
                        *boost::unit_test::label("basic_extents_impl") *
                            boost::unit_test::label("copy_ctor")) {

  auto e_c_0 = e0;   // {}
  auto e_c_1 = e1;   // {1,1,1,1}
  auto e_c_2 = e2;   // {1,2,3}
  auto e_c_3 = e3;   // {4,2,3}
  auto e_c_4 = e4;   // {4,2,1,3}
  auto e_c_5 = e5;   // {1,4,2,1,3,1}

  BOOST_CHECK_EQUAL(e_c_0.size(), 0);
  BOOST_CHECK(e_c_0.empty());

  BOOST_REQUIRE_EQUAL(e_c_1.size(), 4);
  BOOST_REQUIRE_EQUAL(e_c_2.size(), 3);
  BOOST_REQUIRE_EQUAL(e_c_3.size(), 3);
  BOOST_REQUIRE_EQUAL(e_c_4.size(), 4);
  BOOST_REQUIRE_EQUAL(e_c_5.size(), 6);

  BOOST_CHECK_EQUAL(e_c_1[0], 1);
  BOOST_CHECK_EQUAL(e_c_1[1], 1);
  BOOST_CHECK_EQUAL(e_c_1[2], 1);
  BOOST_CHECK_EQUAL(e_c_1[3], 1);

  BOOST_CHECK_EQUAL(e_c_2[0], 1);
  BOOST_CHECK_EQUAL(e_c_2[1], 2);
  BOOST_CHECK_EQUAL(e_c_2[2], 3);

  BOOST_CHECK_EQUAL(e_c_3[0], 4);
  BOOST_CHECK_EQUAL(e_c_3[1], 2);
  BOOST_CHECK_EQUAL(e_c_3[2], 3);

  BOOST_CHECK_EQUAL(e_c_4[0], 4);
  BOOST_CHECK_EQUAL(e_c_4[1], 2);
  BOOST_CHECK_EQUAL(e_c_4[2], 1);
  BOOST_CHECK_EQUAL(e_c_4[3], 3);

  BOOST_CHECK_EQUAL(e_c_5[0], 1);
  BOOST_CHECK_EQUAL(e_c_5[1], 4);
  BOOST_CHECK_EQUAL(e_c_5[2], 2);
  BOOST_CHECK_EQUAL(e_c_5[3], 1);
  BOOST_CHECK_EQUAL(e_c_5[4], 3);
  BOOST_CHECK_EQUAL(e_c_5[5], 1);
}

BOOST_FIXTURE_TEST_CASE(test_basic_extents_impl_product, fixture,
                        *boost::unit_test::label("basic_extents_impl") *
                            boost::unit_test::label("product")) {

  auto p0 = e0.product();   // {}
  auto p1 = e1.product();   // {0,0,0,0}
  auto p2 = e2.product();   // {1,2,3}
  auto p3 = e3.product();   // {4,2,3}
  auto p4 = e4.product();   // {4,2,1,3}
  auto p5 = e5.product();   // {1,4,2,1,3,1}

  BOOST_CHECK_EQUAL(p0, 1);
  BOOST_CHECK_EQUAL(p1, 1);
  BOOST_CHECK_EQUAL(p2, 6);
  BOOST_CHECK_EQUAL(p3, 24);
  BOOST_CHECK_EQUAL(p4, 24);
  BOOST_CHECK_EQUAL(p5, 24);
}

BOOST_AUTO_TEST_SUITE_END()
