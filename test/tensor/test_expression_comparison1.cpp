//  Copyright (c) 2018-2019 Cem Bassoy, Mohammad Ashar Khan
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//


#include <boost/numeric/ublas/tensor/tensor.hpp>
#include <boost/numeric/ublas/tensor/expression_operator.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "utility.hpp"

using double_extended = boost::multiprecision::cpp_bin_float_double_extended;


using test_types = zip<int,long,float,double,double_extended>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

struct fixture {
	using extents_type = boost::numeric::ublas::basic_extents<std::size_t>;
	fixture()
	  : extents{
				extents_type{},    // 3
				extents_type{2,3}, // 4
				extents_type{4,2,3}, // 8
				extents_type{4,2,3,5}} // 9
	{
	}
	std::vector<extents_type> extents;
};

BOOST_AUTO_TEST_SUITE(test_tensor_comparison, * boost::unit_test::depends_on("test_tensor"))


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_comparison, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;


	auto check = [](auto const& e)
	{
		auto t  = tensor_type (e);
		auto t2 = tensor_type (e);
		auto v  = value_type  {};

		std::iota(t.begin(), t.end(), v);
		std::iota(t2.begin(), t2.end(), v+2);

		BOOST_CHECK( (bool)(t == t)  );
		BOOST_CHECK( (bool)(t != t2) );

		if(t.empty())
			return;

		BOOST_CHECK(!(bool)(t < t));
		BOOST_CHECK(!(bool)(t > t));
		BOOST_CHECK( (bool)(t < t2) );
		BOOST_CHECK( (bool)(t2 > t) );
		BOOST_CHECK( (bool)(t <= t) );
		BOOST_CHECK( (bool)(t >= t) );
		BOOST_CHECK( (bool)(t <= t2) );
		BOOST_CHECK( (bool)(t2 >= t) );
		BOOST_CHECK( (bool)(t2 >= t2) );
		BOOST_CHECK( (bool)(t2 >= t) );
	};

	for(auto const& e : extents)
		check(e);

	auto e0 = extents.at(0);
	auto e1 = extents.at(1);
	auto e2 = extents.at(2);


	auto b = false;

	BOOST_CHECK_NO_THROW ( b = (tensor_type(e0) == tensor_type(e0)));
	BOOST_CHECK_NO_THROW ( b = (tensor_type(e1) == tensor_type(e2)));
	BOOST_CHECK_NO_THROW ( b = (tensor_type(e0) == tensor_type(e2)));
	BOOST_CHECK_NO_THROW ( b = (tensor_type(e1) != tensor_type(e2)));

	BOOST_CHECK_THROW    ( b = (tensor_type(e1) >= tensor_type(e2)), std::runtime_error  );
	BOOST_CHECK_THROW    ( b = (tensor_type(e1) <= tensor_type(e2)), std::runtime_error  );
	BOOST_CHECK_THROW    ( b = (tensor_type(e1) <  tensor_type(e2)), std::runtime_error  );
	BOOST_CHECK_THROW    ( b = (tensor_type(e1) >  tensor_type(e2)), std::runtime_error  );

}

BOOST_AUTO_TEST_SUITE_END()
