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

#ifndef BOOST_UBLAS_INCLUDE_BOOST_NUMERIC_UBLAS_TENSOR_LAMBDA_TRAITS_HPP_
#define BOOST_UBLAS_INCLUDE_BOOST_NUMERIC_UBLAS_TENSOR_LAMBDA_TRAITS_HPP_

#include <tuple>
namespace boost::numeric::ublas::detail {

template <class T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const> {
  enum { arity = sizeof...(Args) };

  typedef ReturnType result_type;

  template <size_t i> struct arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
  };
};

} // namespace boost::numeric::ublas::detail

#endif // BOOST_UBLAS_INCLUDE_BOOST_NUMERIC_UBLAS_TENSOR_LAMBDA_TRAITS_HPP_
