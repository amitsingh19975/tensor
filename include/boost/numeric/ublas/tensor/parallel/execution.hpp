#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_HPP

#include "fwd.hpp"
#include "properties.hpp"

namespace boost::numeric::ublas::parallel::execution{
    
    template<class Executor> using executor_shape_t = typename executor_shape<Executor>::type;
    template<class Executor> using executor_index_t = typename executor_index<Executor>::type;

    template<class InterfaceProperty, class... SupportableProperties>
    using executor = typename InterfaceProperty::template polymorphic_executor_type<InterfaceProperty, SupportableProperties...>;
    
} // namespace boost::numeric::ublas::parallel::execution


#include "execution/is_executor.hpp"
#include "execution/is_oneway_executor.hpp"
#include "execution/is_bulk_oneway_executor.hpp"
#include "execution/context.hpp"
#include "execution/executor_concept.hpp"
#include "execution/oneway.hpp"
#include "execution/bulk_oneway.hpp"
#include "execution/blocking.hpp"
#include "execution/blocking_adaptation.hpp"
#include "execution/relationship.hpp"
#include "execution/outstanding_work.hpp"
#include "execution/bulk_guarantee.hpp"
#include "execution/mapping.hpp"
#include "execution/allocator.hpp"
#include "execution/occupancy.hpp"
#include "execution/executor_shape.hpp"
#include "execution/executor_index.hpp"
#include "execution/oneway_executor.hpp"
#include "execution/bulk_oneway_executor.hpp"

#endif