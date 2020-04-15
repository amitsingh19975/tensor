#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BAD_EXECUTOR_EXCEPTION_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_EXECUTION_BAD_EXECUTOR_EXCEPTION_HPP

#include <exception>

namespace boost::numeric::ublas::parallel::execution{
    
    struct bad_executor : std::exception{
        bad_executor() noexcept{}
        ~bad_executor() noexcept {}

        virtual char const* what() const noexcept{
            return "bad executor";
        }
    };

} // namespace boost::numeric::ublas::parallel::execution




#endif