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
#include <boost/numeric/ublas/tensor/tensor_expression.hpp>
#include <boost/test/unit_test.hpp>

#include "utility.hpp"

#include <functional>

using test_types = zip<int, long, float, double, std::complex<float>>::with_t<
    boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

struct fixture {
  using extents_type = boost::numeric::ublas::shape;
  fixture()
      : extents{extents_type{}, // 0

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

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_expression_eval, value, test_types,
                                 fixture) {
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
    auto result = expr.template eval<value_type, layout_type>();

    BOOST_CHECK(result.extents() == e);
    for (auto st = 0u; st < t.extents().product(); st++)
      BOOST_CHECK_EQUAL(result(st), t(st) + t(st));

    auto expr2 = 2.0f * t;
    auto result2 = expr2.template eval<value_type, layout_type>();

    BOOST_CHECK(result2.extents() == e);
    for (auto st = 0u; st < t.extents().product(); st++)
      BOOST_CHECK_EQUAL(result2(st), 2.0f * t(st));

    auto expr3 = ((2.0f * t) / (t + 1.0f));
    auto result3 = expr3.template eval<value_type, layout_type>();

    BOOST_CHECK(result3.extents() == e);
    for (auto st = 0u; st < t.extents().product(); st++)
      BOOST_CHECK_EQUAL(
          result3(st), static_cast<value_type>(((2.0f * t(st)) / (t(st) + 1.0f))));
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_expression_eval_to, value,
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
    tensor_type result;
    expr.eval_to(result);

    BOOST_CHECK(result.extents() == e);
    for (auto st = 0u; st < t.extents().product(); st++)
      BOOST_CHECK_EQUAL(result(st), t(st) + t(st));

    auto expr2 = 2.0f * t;
    tensor_type result2;
    expr2.eval_to(result2);

    BOOST_CHECK(result2.extents() == e);
    for (auto st = 0u; st < t.extents().product(); st++)
      BOOST_CHECK_EQUAL(result2(st), 2.0f * t(st));

    auto expr3 = ((2.0f * t) / (t + 1.0f));
    tensor_type result3;
    expr3.eval_to(result3);

    BOOST_CHECK(result3.extents() == e);
    for (int st = 0; st < t.extents().product(); st++)
      BOOST_CHECK_EQUAL(
          result3(st), static_cast<value_type>(((2.0f * t(st)) / (t(st) + 1.0f))));
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

      BOOST_CHECK_THROW(static_cast<bool>(t-t), std::runtime_error);
    }
  }
}
