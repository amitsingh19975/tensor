//
//  Copyright (c) 2018-2019, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef _BOOST_UBLAS_TENSOR_STATIC_FUNCTIONS_HPP_
#define _BOOST_UBLAS_TENSOR_STATIC_FUNCTIONS_HPP_


#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>

#include <boost/numeric/ublas/tensor/multiplication.hpp>
#include <boost/numeric/ublas/tensor/algorithms.hpp>
#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor/expression_evaluation.hpp>
#include <boost/numeric/ublas/tensor/storage_traits.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>

namespace boost::numeric::ublas
{

	namespace detail{

		template<size_t M, size_t I, typename T, size_t E0, size_t... E, size_t O0, size_t... OtherE, size_t... R>
		BOOST_UBLAS_INLINE
		constexpr auto extents_result_type_tensor_times_vector(basic_static_extents<T,E0,E...>, 
			basic_static_extents<T, O0, OtherE...> ones, basic_static_extents<T, R...> res = basic_static_extents<T>{})
		{
			if constexpr(I != M - 1){
				return extents_result_type_tensor_times_vector<M,I + 1>
					( basic_static_extents<T,E...>{}, basic_static_extents<T,OtherE...>{}, basic_static_extents<T, R..., E0>{} );
			}else{
				return extents_result_type_tensor_times_vector<M,I + 1>
					( basic_static_extents<T,E...>{}, basic_static_extents<T,O0,OtherE...>{}, basic_static_extents<T, R...>{} );
			}
		}

		template<size_t M, size_t I, typename T, size_t... E, size_t... R>
		BOOST_UBLAS_INLINE
		constexpr auto extents_result_type_tensor_times_vector(basic_static_extents<T>, 
			basic_static_extents<T, E...> ones, basic_static_extents<T, R...>)
		{
			return basic_static_extents<T, R..., E...>{};
		}

		template<size_t I, typename T, size_t... OtherE>
		BOOST_UBLAS_INLINE
		constexpr auto extents_result_set_to_ones(
			basic_static_extents<T,OtherE...> res = basic_static_extents<T>{})
		{
			if constexpr( I == 0 ){
				return res;
			}else{
				return extents_result_set_to_ones< I - 1 >( basic_static_extents<T,OtherE...,1ul> {} );
			}
		}

		template<size_t M, typename T, size_t E0, size_t... E>
		BOOST_UBLAS_INLINE
		constexpr auto extents_result_type_tensor_times_vector(basic_static_extents<T,E0,E...> const& e)
		{
			using size_type = typename basic_static_extents<T>::size_type;
			auto ones = extents_result_set_to_ones< std::max( size_type( sizeof...(E) ), size_type(2) ), T >();
			return extents_result_type_tensor_times_vector<M,0>(e, ones);
		}

		template<size_t I, size_t NE, typename T, size_t E0, size_t... E, size_t... OtherE>
		BOOST_UBLAS_INLINE
		constexpr auto static_extents_set_at
			( basic_static_extents<T,E0,E...> const& e, basic_static_extents<T,OtherE...> res = basic_static_extents<T>{}){
			static_assert( I < sizeof...(E) + 1, "boost::numeric::ublas::detail::static_extents_set_at(): out of bound");
			if constexpr( sizeof...(E) == 0 ){
				if constexpr( I == 0 ){
					return basic_static_extents<T,OtherE..., NE>{};
				}else{
					return basic_static_extents<T,OtherE...,E0>{};
				}
			}else{
				if constexpr(I == 0){
					return basic_static_extents<T,OtherE..., NE, E...>{};
				}else{
					return static_extents_set_at<I - 1, NE>( basic_static_extents<T,E...>{}, basic_static_extents<T,OtherE..., E0>{} );
				}
			}
		}

		template<typename T, size_t... E, size_t... OtherE>
		BOOST_UBLAS_INLINE
		constexpr auto concat_static_extents( basic_static_extents<T, E...> const&, basic_static_extents<T,OtherE...> const& ){
			return basic_static_extents<T, E..., OtherE...>();
		}

	} // namespace detail
	
	/** @brief Computes the m-mode tensor-times-vector product
	 *
	 * Implements C[i1,...,im-1,im+1,...,ip] = A[i1,i2,...,ip] * b[im]
	 *
	 * @note calls ublas::ttv
	 *
	 * @tparam 	  m contraction dimension with 1 <= m <= p
	 * @param[in] a tensor object A with order p
	 * @param[in] b vector object B
	 *
	 * @returns tensor object C with order p-1, the same storage format and allocator type as A
	*/
	template <size_t M, class V, class E, class F, class A1, class A2, 
		std::enable_if_t<detail::is_static<E>::value,int> = 0
	>
	BOOST_UBLAS_INLINE decltype(auto) prod(tensor<V, E, F, A1> const &a, vector<V, A2> const &b)
	{

		using tensor_type = tensor<V, E, F, A1>;
		using extents_type = typename tensor_type::extents_type;
		using value_type = typename tensor_type::value_type;

		auto const p = std::size_t(a.rank());

		static_assert( M != 0, 
				"error in boost::numeric::ublas::prod(ttv): "
				"contraction mode must be greater than zero.");

		static_assert( extents_type::Rank >= M,
				"error in boost::numeric::ublas::prod(ttv): rank of tensor must be "
				"greater than or equal to the modus.");

		static_assert(extents_type::Rank != 0,
				"error in boost::numeric::ublas::prod(ttv): first "
				"argument tensor should not be empty.");

		if (b.size() == 0)
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttv): second "
				"argument vector should not be empty.");

		auto nc = detail::extents_result_type_tensor_times_vector<M>(a.extents());
		auto nb = std::vector<typename extents_type::value_type>{b.size(), 1};

		auto c = tensor(nc, value_type{});
		auto bb = &(b(0));

		auto a_static_extents = a.extents().base();
		auto c_static_extents = c.extents().base();

		auto a_static_strides = a.strides().base();
		auto c_static_strides = c.strides().base();

		ttv(M, p,
			c.data(), c_static_extents.data(), c_static_strides.data(),
			a.data(), a_static_extents.data(), a_static_strides.data(),
			bb, nb.data(), nb.data());

		return c;
	}

	/** @brief Computes the m-mode tensor-times-matrix product
	 *
	 * Implements C[i1,...,im-1,j,im+1,...,ip] = A[i1,i2,...,ip] * B[j,im]
	 *
	 * @note calls ublas::ttm
	 *
	 * @tparam    m contraction dimension with 1 <= m <= p
	 * @param[in] a tensor object A with order p
	 * @param[in] b vector object B
	 *
	 * @returns tensor object C with order p, the same storage format and allocator type as A
	*/
	template <size_t M, size_t MatricRow, class V, class E, class F, class A1, class A2,
		std::enable_if_t<detail::is_static<E>::value,int> = 0
	>
	BOOST_UBLAS_INLINE decltype(auto) prod(tensor<V, E, F, A1> const &a, matrix<V, F, A2> const &b)
	{
		using tensor_type = tensor<V, E, F, A1>;
		using dynamic_strides_type = strides_t<dynamic_extents<>,F>;
		using value_type = typename tensor_type::value_type;

		auto const p = a.rank();

		static_assert(M != 0,
				"error in boost::numeric::ublas::prod(ttm): "
				"contraction mode must be greater than zero.");

		static_assert( E::Rank >= M ,
				"error in boost::numeric::ublas::prod(ttm): rank "
				"of the tensor must be greater equal the modus.");

		static_assert(E::Rank,
				"error in boost::numeric::ublas::prod(ttm): first "
				"argument tensor should not be empty.");

		if (b.size1() * b.size2() == 0)
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttm): second "
				"argument matrix should not be empty.");

		auto nc = detail::static_extents_set_at< M - 1, MatricRow >( a.extents() );
		auto nb = dynamic_extents<>{b.size1(), b.size2()};

		auto wb = dynamic_strides_type(nb);

		auto c = tensor(nc, value_type{});

		auto bb = &(b(0, 0));

		auto a_static_extents = a.extents().base();
		auto c_static_extents = c.extents().base();

		auto a_static_strides = a.strides().base();
		auto c_static_strides = c.strides().base();
		ttm(M, p,
			c.data(), c_static_extents.data(), c_static_strides.data(),
			a.data(), a_static_extents.data(), a_static_strides.data(),
			bb, nb.data(), wb.data());

		return c;
	}

	/** @brief Computes the outer product of two tensors
	 *
	 * Implements C[i1,...,ip,j1,...,jq] = A[i1,i2,...,ip] * B[j1,j2,...,jq]
	 *
	 * @note calls outer function
	 *
	 * @param[in] a tensor object A
	 * @param[in] b tensor object B
	 *
	 * @returns tensor object C with the same storage format F and allocator type A1
	*/
	template <class V, class E1, class E2, class F, class A1, class A2,
		std::enable_if_t<
			detail::is_static<E1>::value &&
			detail::is_static<E2>::value
			,int> = 0
	>
	BOOST_UBLAS_INLINE decltype(auto) outer_prod(tensor<V, E1, F, A1> const &a, tensor<V, E2, F, A2> const &b)
	{
		if (a.empty() || b.empty())
			throw std::runtime_error(
				"error in boost::numeric::ublas::outer_prod: "
				"tensors should not be empty.");

		auto nc = detail::concat_static_extents(a.extents(), b.extents());

		auto a_extents = a.extents();
		auto b_extents = b.extents();

		
		auto c = tensor(nc, V{});

		auto a_static_extents = a_extents.base();
		auto a_static_strides = a.strides().base();
		
		auto b_static_extents = b_extents.base();
		auto b_static_strides = b.strides().base();
		
		auto c_static_extents = c.extents().base();
		auto c_static_strides = c.strides().base();

		outer(c.data(), c.rank(), c_static_extents.data(), c_static_strides.data(),
			a.data(), a.rank(), a_static_extents.data(), a_static_strides.data(),
			b.data(), b.rank(), b_static_extents.data(), b_static_strides.data());

		return c;
	}

}


#endif
