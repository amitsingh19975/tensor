//  Copyright (c) 2019 Mohammad Ashar Khan
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//

#include <boost/numeric/ublas/tensor/expression_operator.hpp>
#include <boost/numeric/ublas/tensor/tensor.hpp>
#include <boost/test/unit_test.hpp>

#include "utility.hpp"

using test_types =
    zip<int, long, float, double>::with_t<boost::numeric::ublas::first_order,
                                          boost::numeric::ublas::last_order>;

struct V {
  virtual void f() {}
};
struct B : virtual V {};
struct C : virtual B {};

using test_types2 = zip<V, B, C>::with_t<boost::numeric::ublas::first_order,
                                         boost::numeric::ublas::last_order>;

struct fixture {
  using extents_type = boost::numeric::ublas::shape;
  fixture()
      : extents{extents_type{1, 1},             // 1
                extents_type{1, 2},             // 2
                extents_type{1, 2, 3},          // 6
                extents_type{4, 2, 3},          // 9
                extents_type{1, 4, 2, 1, 3, 1}} // 12
  {}
  std::vector<extents_type> extents;
};

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_cast, value, test_types,
                                 fixture) {
  using namespace boost::numeric;
  using value_type = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::tensor<value_type, layout_type>;

  for (auto const &e : extents) {

    tensor_type t(e, value_type{1});

    if constexpr (std::is_same_v<value_type, int>) {

      auto a = ublas::static_tensor_cast<long>(t);
      auto b = ublas::static_tensor_cast<float>(t);
      auto c = ublas::static_tensor_cast<double>(t);

      static_assert(std::is_same_v<typename decltype(a)::value_type, long>);
      static_assert(std::is_same_v<typename decltype(b)::value_type, float>);
      static_assert(std::is_same_v<typename decltype(c)::value_type, double>);

      static_assert(
          std::is_same_v<typename decltype(a)::layout_type, layout_type>);
      static_assert(
          std::is_same_v<typename decltype(b)::layout_type, layout_type>);
      static_assert(
          std::is_same_v<typename decltype(c)::layout_type, layout_type>);
    }

    if constexpr (std::is_same_v<value_type, long>) {

      auto a = ublas::static_tensor_cast<int>(t);
      auto b = ublas::static_tensor_cast<float>(t);
      auto c = ublas::static_tensor_cast<double>(t);

      static_assert(std::is_same_v<typename decltype(a)::value_type, int>);
      static_assert(std::is_same_v<typename decltype(b)::value_type, float>);
      static_assert(std::is_same_v<typename decltype(c)::value_type, double>);

      static_assert(
          std::is_same_v<typename decltype(a)::layout_type, layout_type>);
      static_assert(
          std::is_same_v<typename decltype(b)::layout_type, layout_type>);
      static_assert(
          std::is_same_v<typename decltype(c)::layout_type, layout_type>);
    }

    if constexpr (std::is_same_v<value_type, float>) {

      auto a = ublas::static_tensor_cast<long>(t);
      auto b = ublas::static_tensor_cast<int>(t);
      auto c = ublas::static_tensor_cast<double>(t);

      static_assert(std::is_same_v<typename decltype(a)::value_type, long>);
      static_assert(std::is_same_v<typename decltype(b)::value_type, int>);
      static_assert(std::is_same_v<typename decltype(c)::value_type, double>);

      static_assert(
          std::is_same_v<typename decltype(a)::layout_type, layout_type>);
      static_assert(
          std::is_same_v<typename decltype(b)::layout_type, layout_type>);
      static_assert(
          std::is_same_v<typename decltype(c)::layout_type, layout_type>);
    }
    if constexpr (std::is_same_v<value_type, double>) {

      auto a = ublas::static_tensor_cast<long>(t);
      auto b = ublas::static_tensor_cast<float>(t);
      auto c = ublas::static_tensor_cast<int>(t);

      static_assert(std::is_same_v<typename decltype(a)::value_type, long>);
      static_assert(std::is_same_v<typename decltype(b)::value_type, float>);
      static_assert(std::is_same_v<typename decltype(c)::value_type, int>);

      static_assert(
          std::is_same_v<typename decltype(a)::layout_type, layout_type>);
      static_assert(
          std::is_same_v<typename decltype(b)::layout_type, layout_type>);
      static_assert(
          std::is_same_v<typename decltype(c)::layout_type, layout_type>);
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_dynamic_cast, value, test_types2,
                                 fixture) {
  using namespace boost::numeric;

  using value_type = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::tensor<value_type *, layout_type>;

  for (auto const &e : extents) {
    auto val = value_type{};
    auto t = tensor_type(e, &val);

    if constexpr (std::is_same_v<value_type, C>) {
      auto a = ublas::dynamic_tensor_cast<V *>(t);
      auto b = ublas::dynamic_tensor_cast<B *>(t);

      for (auto const &elem : a)
        assert(elem != nullptr); // every thing was casted Okay

      for (auto const &elem : b)
        assert(elem != nullptr);

      static_assert(std::is_same_v<typename decltype(a)::value_type, V *>);
      static_assert(std::is_same_v<typename decltype(b)::value_type, B *>);
    }
  }
}

using test_types3 = zip<int>::with_t<boost::numeric::ublas::first_order,
                                     boost::numeric::ublas::last_order>;

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_reinterpret_cast, value,
                                 test_types3, fixture) {
  using namespace boost::numeric;

  using value_type = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::tensor<value_type, layout_type>;

  for (auto const &e : extents) {
    auto val = value_type{1};
    auto t = tensor_type(e, val);
    auto a = ublas::reinterpret_tensor_cast<char *>(t);

    for (auto const &elem : a)
      assert(elem != nullptr); // every thing was casted Okay

    static_assert(std::is_same_v<typename decltype(a)::value_type, char *>);
  }
}
