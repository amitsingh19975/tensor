//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//

#ifndef _BOOST_NUMERIC_UBLAS_DETAIL_TENSOR_FUNCTIONS_HPP_
#define _BOOST_NUMERIC_UBLAS_DETAIL_TENSOR_FUNCTIONS_HPP_

#include <boost/numeric/ublas/tensor/detail/extents_functions.hpp>

namespace boost::numeric::ublas {

/** @brief Reshapes the tensor
 *
 *
 * (1) @code reshape(A,extents{m,n,o});     @endcode or
 * (2) @code reshape(A,extents{m,n,o},4);   @endcode
 *
 * If the size of this smaller than the specified extents than
 * default constructed (1) or specified (2) value is appended.
 *
 * @note rank of the tensor_core might also change.
 *
 * @param t tensor to be reshaped.
 * @param e extents with which the tensor_core is reshaped.
 * @param v value which is appended if the tensor_core is enlarged.
 */
template <typename T, typename ExtentsType, typename ValueType = typename tensor_core<T>::value_type>
inline
constexpr auto reshape(tensor_core<T>& t, ExtentsType const &e, ValueType v = ValueType{}) {
    using resizable_tag = typename tensor_core<T>::resizable_tag;
    using strides_type = typename tensor_core<T>::strides_type;
    static_assert(is_extents_v<ExtentsType>, "boost::numeric::ublas::reshape(tensor_core<T>&,ExtentsType const&) : invalid type, type should be an extents");
  
    static_assert(is_dynamic_v<ExtentsType>,
        "boost::numeric::ublas::reshape(tensor_core<T>&,ExtentsType const&): static extents cannot be reshaped");
  
    static_assert(is_dynamic_v<strides_type>,
        "boost::numeric::ublas::reshape(tensor_core<T>&,ExtentsType const&): static strides cannot be reshaped");

    auto& te = t.extents();
    auto& base = t.base();

    auto pp = product(te);
    te = e;
    t.strides() = strides_type(te);
    t.invalidate_size();

    auto p = product(te);

    if constexpr(std::is_same_v<resizable_tag,storage_resizable_container_tag>){
        if(p != base.size())
            base.resize(p,v);
    }else{
        if( p > base.size() ){
            throw std::length_error(
                "boost::numeric::ublas::tensor_core<T>::reshape(size_type, value_type): "
                "tensor cannot be reshaped because extents required size exceeds the static storage size"
            );
        }

        if( pp < p ){
            std::fill_n(base.begin() + pp, p - pp, v);
        }
    }
}


} // namespace boost::numeric::ublas

#endif
