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

#include <boost/numeric/ublas/tensor/expression_operator.hpp>
#include <boost/numeric/ublas/tensor/tensor.hpp>
#include <boost/numeric/ublas/tensor/tensor_expression.hpp>
#include <boost/test/unit_test.hpp>

#include "utility.hpp"

#include <functional>

using test_types = zip<int, long, float, double, std::complex<float>>::with_t<
    boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

struct fixture {
  using extents_type = boost::numeric::ublas::shape;
  fixture()
      : extents{//extents_type{}, // 0

                extents_type{1, 1}, // 1
                extents_type{1, 2}, // 2
                extents_type{2, 1}, // 3

                extents_type{2, 3},          // 4
                extents_type{2, 3, 1},       // 5
                extents_type{1, 2, 3},       // 6
                extents_type{1, 1, 2, 3},    // 7
                extents_type{1, 2, 3, 1, 1}, // 8

                extents_type{4, 2, 3},          // 9
                extents_type{4, 2, 1, 3},       // 10
                extents_type{4, 2, 1, 3, 1},    // 11
                extents_type{1, 4, 2, 1, 3, 1}} // 12
  {}
  std::vector<extents_type> extents;
};

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_expression_call_operator, value,
                                 test_types, fixture) {
  using namespace boost::numeric;
  using value_type = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::tensor<value_type, layout_type>;

  for (auto const &e : extents) {

    auto t = tensor_type(e);
    auto v = value_type{0};
    for (auto &tt : t) {
      tt = v;
      v += value_type{1};
    }

    auto expr = t + t;
    for (auto st = 0u; st < t.extents().product(); st++)
      BOOST_CHECK_EQUAL(expr(st), t(st) + t(st));

    auto expr2 = 2.0f * t;
    for (auto st = 0u; st < t.extents().product(); st++)
      BOOST_CHECK_EQUAL(expr2(st), 2.0f * t(st));

    auto expr3 = ((2.0f * t) / (t + 1.0f));
    for (auto st = 0u; st < t.extents().product(); st++)
      BOOST_CHECK_EQUAL(expr3(st), ((2.0f * t(st)) / (t(st) + 1.0f)));
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_expression_bool_operator, value,
                                 test_types, fixture) {
  using namespace boost::numeric;
  using value_type = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::tensor<value_type, layout_type>;

  for (auto const &e : extents) {

    auto t = tensor_type(e);
    auto v = value_type{0};
    for (auto &tt : t) {
      tt = v;
      v += value_type{1};
    }

    auto expr = t == t;
    bool result = expr;

    BOOST_CHECK(result);

    if constexpr (!std::is_same_v<value_type, std::complex<float>>) {
      auto expr3 = t > t - 1.0f;
      bool result3 = expr3;
      BOOST_CHECK(result3);

      auto expr4 = t < t + 1.0f;
      bool result4 = expr4;
      BOOST_CHECK(result4);

      auto expr5 = t + 1.0f == t;
      bool result5 = expr5;
      if (e.product() > 0)
        BOOST_CHECK_EQUAL(result5, false);

      auto expr2 = 2.0f * t == t + t;
      bool result2 = expr2;
      BOOST_CHECK(result2);

      BOOST_CHECK_THROW(static_cast<bool>(t - t), std::runtime_error);
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_expression_optimizer, value,
                                 test_types, fixture) {
  using namespace boost::numeric;
  using value_type = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::tensor<value_type, layout_type>;

  for (auto const &e : extents) {

    auto a = tensor_type(e);
    auto v = value_type{0};
    for (auto &tt : a) {
      tt = v;
      v += value_type{1};
    }

    auto b = a;
    std::reverse(b.begin(), b.end());
    tensor_type c = a + b;

    auto expr1 = a * b + c * a;
    auto expr2 = a * c + c * b;
    auto expr3 = a * b - a * c;

    for (auto i = 0u; i < a.size(); i++) {

      auto i1 =
          boost::yap::transform(expr1, ublas::detail::transforms::at_index{i});
      auto optimized1 = boost::yap::transform(
          i1, ublas::detail::transforms::apply_distributive_law{});

      auto i2 =
          boost::yap::transform(expr2, ublas::detail::transforms::at_index{i});
      auto optimized2 = boost::yap::transform(
          i2, ublas::detail::transforms::apply_distributive_law{});

      auto i3 =
          boost::yap::transform(expr3, ublas::detail::transforms::at_index{i});
      auto optimized3 = boost::yap::transform(
          i3, ublas::detail::transforms::apply_distributive_law{});

      auto optimized_expr1 = boost::yap::make_terminal(a(i) * (b(i) + c(i)));
      auto optimized_expr2 = boost::yap::make_terminal(c(i) * (b(i) + a(i)));
      auto optimized_expr3 = boost::yap::make_terminal(a(i) * (b(i) - c(i)));

      static_assert(
          std::is_same_v<decltype(optimized1), decltype(optimized_expr1)>);
      static_assert(
          std::is_same_v<decltype(optimized2), decltype(optimized_expr2)>);
      static_assert(
          std::is_same_v<decltype(optimized3), decltype(optimized_expr3)>);

      BOOST_CHECK((bool)(boost::yap::evaluate(optimized1) ==
                         boost::yap::evaluate(optimized_expr1)));
      BOOST_CHECK((bool)(optimized2 == optimized_expr2));
      BOOST_CHECK((bool)(optimized3 == optimized_expr3));
    }
  }
}
