#ifndef _BOOST_UBLAS_TENSOR_DETAIL_PARALLEL_QUERY_OR_HPP
#define _BOOST_UBLAS_TENSOR_DETAIL_PARALLEL_QUERY_OR_HPP

#include <type_traits>
#include "../properties.hpp"

namespace boost::numeric::ublas::parallel::detail{
    
    template<typename T, typename Property, typename Default>
    auto query_or(T const& query_me, Property const& p, Default&& def)
    {
        if constexpr( can_query_v<T,Property> ){
            return query(query_me, p);
        }else{
            return def;
        }
    }

} // namespace boost::numeric::ublas::parallel::detail


#endif // _BOOST_UBLAS_TENSOR_DETAIL_PARALLEL_QUERY_OR_HPP