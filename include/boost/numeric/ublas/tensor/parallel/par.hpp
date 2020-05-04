#pragma once

#include "thread_pool.hpp"
#include <thread>
#include <algorithm>
#include <iostream>
#include <boost/core/demangle.hpp>

namespace boost::numeric::ublas::parallel{


namespace impl
{


  template<class, class> class basic_execution_policy;


  template<class ER, class E>
  basic_execution_policy<ER,E> make_basic_execution_policy(const E& ex);


  template<class ExecutionRequirement, class Executor>
  class basic_execution_policy
  {
    public:
      static constexpr ExecutionRequirement execution_requirement{};

      BOOST_UBLAS_TENSOR_ALWAYS_INLINE Executor executor() const
      {
        return executor_;
      }

    private:
      template<class ER, class E>
      friend basic_execution_policy<ER,E> make_basic_execution_policy(const E& ex);

      basic_execution_policy(const Executor& executor)
        : executor_(executor)
      {}

      Executor executor_;
  };


  template<class ER, class E>
  basic_execution_policy<ER,E> make_basic_execution_policy(const E& ex)
  {
    return basic_execution_policy<ER,E>{ex};
  }


template<class Executor, class ExecutionRequirement>
constexpr bool satisfies_on_requirements_v =
  can_require_concept_v<
    Executor
    , execution::bulk_oneway_t
> &&
  can_require_v<
    Executor
    , ExecutionRequirement
    , execution::blocking_t::always_t
    , execution::mapping_t::thread_t
>;


  } // end impl


  class parallel_policy
  {


    public:
      static constexpr execution::bulk_guarantee_t::parallel_t execution_requirement{};

      template<class Executor
        ,class = std::enable_if_t<
                impl::satisfies_on_requirements_v<Executor, decltype(execution_requirement)>
              >
      >
      BOOST_UBLAS_TENSOR_ALWAYS_INLINE impl::basic_execution_policy<decltype(execution_requirement), Executor> on(const Executor& ex) const
      {
        return impl::make_basic_execution_policy<decltype(execution_requirement)>(ex);
      }
  };




  namespace impl
  {


    execution::chunk_thread_pool<execution::task<void>> system_thread_pool{std::max(1u,std::thread::hardware_concurrency())};


  } // end impl

  constexpr parallel_policy par{};
  template<typename Range, typename Size1, typename Size2, typename Function>
  BOOST_UBLAS_TENSOR_ALWAYS_INLINE void parallel_for(parallel_policy const& policy, Range first, Size1 n, Function f, Size2 stride)
  {
      using size_type = std::common_type_t<Size1,Size2>;
      impl::parallel_for(par.on(impl::system_thread_pool.executor()), first, static_cast<size_type>(n), std::move(f), static_cast<size_type>(stride));
  }

} // namespace boost::numeric::ublas::parallel
