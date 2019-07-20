//  Copyright (c) 2019 Mohammad Ashar Khan
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  and Google in producing this work
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

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_expression_for_each_tensor, value,
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

    tensor_type t_copy1 = t;
    tensor_type t_copy2 = t;

    BOOST_TEST_CHECKPOINT("Running with " + e.to_string());

    BOOST_TEST_PASSPOINT();
    auto terminal_tensor = boost::yap::make_terminal<ublas::detail::tensor_expression>(t_copy1);
    BOOST_TEST_PASSPOINT();
    auto transformed_expr1 = ublas::for_each2(terminal_tensor, [](auto const& ep){return 5.0f;});
    BOOST_TEST_PASSPOINT();
    auto transformed_expr2 = ublas::for_each2(terminal_tensor, [](auto const& ep){return 5.0f+ep;});
    BOOST_TEST_PASSPOINT();
    auto transformed_expr3 = ublas::for_each2(terminal_tensor, [](auto const& ep){return ep*ep;});
    BOOST_TEST_PASSPOINT();
    auto transformed_expr4 = ublas::for_each2(terminal_tensor, [](value_type const& ep){return sqrt(ep);});

    BOOST_TEST_PASSPOINT();
    auto transformed_expr5 = ublas::for_each2(t_copy2, [](auto const& ep){return 5.0f;});
    BOOST_TEST_PASSPOINT();
    auto transformed_expr6 = ublas::for_each2(t_copy2, [](auto const& ep){return 5.0f+ep;});
    BOOST_TEST_PASSPOINT();
    auto transformed_expr7 = ublas::for_each2(t_copy2, [](auto const& ep){return ep*ep;});
    BOOST_TEST_PASSPOINT();
    auto transformed_expr8 = ublas::for_each2(t_copy2, [](auto const& ep){return sqrt(ep);});

    BOOST_TEST_PASSPOINT();
    static_assert(ublas::is_tensor_expression_v<decltype(terminal_tensor)>);
    static_assert(ublas::is_tensor_expression_v<decltype(transformed_expr1)>);
    static_assert(ublas::is_tensor_expression_v<decltype(transformed_expr2)>);
    static_assert(ublas::is_tensor_expression_v<decltype(transformed_expr3)>);
    static_assert(ublas::is_tensor_expression_v<decltype(transformed_expr4)>);
    static_assert(ublas::is_tensor_expression_v<decltype(transformed_expr5)>);
    static_assert(ublas::is_tensor_expression_v<decltype(transformed_expr6)>);
    static_assert(ublas::is_tensor_expression_v<decltype(transformed_expr7)>);
    static_assert(ublas::is_tensor_expression_v<decltype(transformed_expr8)>);

    BOOST_TEST_PASSPOINT();
    tensor_type x = transformed_expr1;
    BOOST_TEST_PASSPOINT();
    tensor_type x2 = transformed_expr2;
    BOOST_TEST_PASSPOINT();
    tensor_type x3 = transformed_expr3;
    BOOST_TEST_PASSPOINT();
    tensor_type x4 = transformed_expr4;

    tensor_type x5 = transformed_expr5;
    tensor_type x6 = transformed_expr6;
    tensor_type x7 = transformed_expr7;
    tensor_type x8 = transformed_expr8;



//    BOOST_CHECK((bool)(x == 5.0f));
//    BOOST_CHECK((bool)(x2 == t+5.0f));
//    BOOST_CHECK((bool)(x3 == t*t));
//
//    std::for_each(t.begin(), t.end(), [](auto&e){ e = sqrt(e);});
//    BOOST_CHECK((bool)(x4 == t));
//
//    BOOST_CHECK((bool)(x == x5));
//    BOOST_CHECK((bool)(x2 == x6));
//    BOOST_CHECK((bool)(x3 == x7));
//    BOOST_CHECK((bool)(x4 == x8));
  }


}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_expression_for_each_expression,
                                 value, test_types, fixture) {
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

    auto reverse_t = t;
    std::reverse(reverse_t.begin(), reverse_t.end());

    auto expr = reverse_t - t + value_type{5};

    auto m_expr2 = ublas::for_each2(expr, [](auto const &e){return e*e;});
    auto m_expr3 = ublas::for_each2(expr, [](auto const &e){return value_type{2}*e-value_type{8};});
    auto m_expr4 = ublas::for_each2(expr, [](auto const &e){return e == value_type{0} ? value_type{1}:value_type{0};}); // compliment function
    auto m_expr5 = ublas::for_each2(expr, [](auto const &e){return value_type{1};});

    tensor_type x2 = m_expr2;
    tensor_type x3 = m_expr3;
    tensor_type x4 = m_expr4;
    tensor_type x5 = m_expr5;

    tensor_type k = expr;
//    BOOST_CHECK((bool)(x2 == k*k));
//    BOOST_CHECK((bool)(x3 == value_type{2}*k-value_type{8}));
//    BOOST_CHECK((bool)(x5 == value_type{1}));
//
//    std::for_each(k.begin(), k.end(), [](auto &x){ if(x == value_type{0}) x = value_type{1}; else x = value_type{0};});
//    BOOST_CHECK((bool)(x4 == k));

  }




}
