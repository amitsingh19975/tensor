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

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_comparison_with_scalar, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;


	auto check = [](auto const& e)
	{

		BOOST_CHECK( (bool) (tensor_type(e,value_type{2}) == tensor_type(e,value_type{2}))  );
		BOOST_CHECK( (bool) (tensor_type(e,value_type{2}) != tensor_type(e,value_type{1}))  );

		if(e.empty())
			return;

		BOOST_CHECK( ! (bool)(tensor_type(e,2) <  2) );
		BOOST_CHECK( ! (bool)(tensor_type(e,2) >  2) );
		//BOOST_CHECK(  (bool)(tensor_type(e,2) >= 2) );
		//BOOST_CHECK(  (bool)(tensor_type(e,2) <= 2) );
		BOOST_CHECK(  (bool)(tensor_type(e,2) == 2) );
		BOOST_CHECK(  (bool)(tensor_type(e,2) != 3) );

		//BOOST_CHECK( !(bool)(2 >  tensor_type(e,2)) );
		//BOOST_CHECK( !(bool)(2 <  tensor_type(e,2)) );
		BOOST_CHECK(  (bool)(2 <= tensor_type(e,2)) );
		BOOST_CHECK(  (bool)(2 >= tensor_type(e,2)) );
		//BOOST_CHECK(  (bool)(2 == tensor_type(e,2)) );
		//BOOST_CHECK(  (bool)(3 != tensor_type(e,2)) );

		//BOOST_CHECK( !(bool)( tensor_type(e,2)+3 <  5) );
		//BOOST_CHECK( !(bool)( tensor_type(e,2)+3 >  5) );
		BOOST_CHECK(  (bool)( tensor_type(e,2)+3 >= 5) );
		//BOOST_CHECK(  (bool)( tensor_type(e,2)+3 <= 5) );
		BOOST_CHECK(  (bool)( tensor_type(e,2)+3 == 5) );
		//BOOST_CHECK(  (bool)( tensor_type(e,2)+3 != 6) );

		BOOST_CHECK( !(bool)( 5 >  tensor_type(e,2)+3) );
		BOOST_CHECK( !(bool)( 5 <  tensor_type(e,2)+3) );
		//BOOST_CHECK(  (bool)( 5 >= tensor_type(e,2)+3) );
		BOOST_CHECK(  (bool)( 5 <= tensor_type(e,2)+3) );
		//BOOST_CHECK(  (bool)( 5 == tensor_type(e,2)+3) );
		BOOST_CHECK(  (bool)( 6 != tensor_type(e,2)+3) );


		//BOOST_CHECK( !(bool)( tensor_type(e,2)+tensor_type(e,3) <  5) );
		BOOST_CHECK( !(bool)( tensor_type(e,2)+tensor_type(e,3) >  5) );
		BOOST_CHECK(  (bool)( tensor_type(e,2)+tensor_type(e,3) >= 5) );
		//BOOST_CHECK(  (bool)( tensor_type(e,2)+tensor_type(e,3) <= 5) );
		BOOST_CHECK(  (bool)( tensor_type(e,2)+tensor_type(e,3) == 5) );
		BOOST_CHECK(  (bool)( tensor_type(e,2)+tensor_type(e,3) != 6) );


		BOOST_CHECK( !(bool)( 5 >  tensor_type(e,2)+tensor_type(e,3)) );
		//BOOST_CHECK( !(bool)( 5 <  tensor_type(e,2)+tensor_type(e,3)) );
		//BOOST_CHECK(  (bool)( 5 >= tensor_type(e,2)+tensor_type(e,3)) );
		BOOST_CHECK(  (bool)( 5 <= tensor_type(e,2)+tensor_type(e,3)) );
		//BOOST_CHECK(  (bool)( 5 == tensor_type(e,2)+tensor_type(e,3)) );
		//BOOST_CHECK(  (bool)( 6 != tensor_type(e,2)+tensor_type(e,3)) );
	};

	for(auto const& e : extents)
		check(e);

}
