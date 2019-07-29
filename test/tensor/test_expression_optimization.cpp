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

template <boost::yap::expr_kind A, typename B>
using expr_t = typename boost::numeric::ublas::detail::tensor_expression<A, B>;

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_expression_optimizer_distributive, value,
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
        auto expr3 = a * b + a * c;
        auto expr4 = a * c + b * c;


        auto expr5 = a * b - c * a;
        auto expr6 = a * c - c * b;
        auto expr7 = a * b - a * c;
        auto expr8 = a * c - b * c;

        auto optimizer1 = boost::numeric::ublas::detail::transforms::apply_distributive_law{};
        auto optimizer2 = boost::numeric::ublas::detail::transforms::apply_distributive_law{};
        auto optimizer3 = boost::numeric::ublas::detail::transforms::apply_distributive_law{};
        auto optimizer4 = boost::numeric::ublas::detail::transforms::apply_distributive_law{};
        auto optimizer5 = boost::numeric::ublas::detail::transforms::apply_distributive_law{};
        auto optimizer6 = boost::numeric::ublas::detail::transforms::apply_distributive_law{};
        auto optimizer7 = boost::numeric::ublas::detail::transforms::apply_distributive_law{};
        auto optimizer8 = boost::numeric::ublas::detail::transforms::apply_distributive_law{};

        BOOST_CHECK(!optimizer1.usable);
        BOOST_CHECK(!optimizer2.usable);
        BOOST_CHECK(!optimizer3.usable);
        BOOST_CHECK(!optimizer4.usable);
        BOOST_CHECK(!optimizer5.usable);
        BOOST_CHECK(!optimizer6.usable);
        BOOST_CHECK(!optimizer7.usable);
        BOOST_CHECK(!optimizer8.usable);

        auto optimized1 = boost::yap::transform(expr1, optimizer1);
        auto optimized2 = boost::yap::transform(expr2, optimizer2);
        auto optimized3 = boost::yap::transform(expr3, optimizer3);
        auto optimized4 = boost::yap::transform(expr4, optimizer4);
        auto optimized5 = boost::yap::transform(expr5, optimizer5);
        auto optimized6 = boost::yap::transform(expr6, optimizer6);
        auto optimized7 = boost::yap::transform(expr7, optimizer7);
        auto optimized8 = boost::yap::transform(expr8, optimizer8);


        BOOST_CHECK(optimizer1.usable);
        BOOST_CHECK(optimizer2.usable);
        BOOST_CHECK(optimizer3.usable);
        BOOST_CHECK(optimizer4.usable);
        BOOST_CHECK(optimizer5.usable);
        BOOST_CHECK(optimizer6.usable);
        BOOST_CHECK(optimizer7.usable);
        BOOST_CHECK(optimizer8.usable);


        tensor_type t1 = optimized1;
        tensor_type k1 = expr1;


        tensor_type t2 = optimized2;
        tensor_type k2 = expr2;


        tensor_type t3 = optimized3;
        tensor_type k3 = expr3;


        tensor_type t4 = optimized4;
        tensor_type k4 = expr4;


        tensor_type t5 = optimized5;
        tensor_type k5 = expr5;


        tensor_type t6 = optimized6;
        tensor_type k6 = expr6;


        tensor_type t7 = optimized7;
        tensor_type k7 = expr7;


        tensor_type t8 = optimized8;
        tensor_type k8 = expr8;

        BOOST_CHECK((bool)(t1 == k1));
        BOOST_CHECK((bool)(t2 == k2));
        BOOST_CHECK((bool)(t3 == k3));
        BOOST_CHECK((bool)(t4 == k4));
        BOOST_CHECK((bool)(t5 == k5));
        BOOST_CHECK((bool)(t6 == k6));
        BOOST_CHECK((bool)(t7 == k7));
        BOOST_CHECK((bool)(t8 == k8));

        using term_t = expr_t<boost::yap::expr_kind::terminal, boost::hana::tuple<tensor_type &>>;
        using inner_add_op_t = expr_t<boost::yap::expr_kind::plus, boost::hana::tuple<term_t, term_t>>;
        using inner_sub_op_t = expr_t<boost::yap::expr_kind::minus, boost::hana::tuple<term_t, term_t>>;
        using optimized_plus_t = expr_t<boost::yap::expr_kind::multiplies, boost::hana::tuple<term_t, inner_add_op_t>>;
        using optimized_minus_t = expr_t<boost::yap::expr_kind::multiplies, boost::hana::tuple<term_t, inner_sub_op_t>>;

        static_assert(std::is_same_v<std::remove_reference_t<decltype(optimized1)>, optimized_plus_t>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(optimized2)>, optimized_plus_t>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(optimized3)>, optimized_plus_t>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(optimized4)>, optimized_plus_t>);

        static_assert(std::is_same_v<std::remove_reference_t<decltype(optimized5)>, optimized_minus_t>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(optimized6)>, optimized_minus_t>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(optimized7)>, optimized_minus_t>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(optimized8)>, optimized_minus_t>);

    }
}


