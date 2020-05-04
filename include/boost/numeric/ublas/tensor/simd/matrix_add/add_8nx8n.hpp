#if !defined(BOOST_NUMERIC_UBLAS_SIMD_ADD_8nX8n_HPP)
#define BOOST_NUMERIC_UBLAS_SIMD_ADD_8nX8n_HPP

#include <boost/numeric/ublas/tensor/simd/matrix_sub/sub_8x8.hpp>

namespace boost::numeric::ublas::simd{
    
    template<typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE 
    void add_8nx8n( float* c, SizeType const* nc, SizeType const* wc, 
        float const* a, SizeType const* na, SizeType const* wa,
        float const* b, SizeType const* nb, SizeType const* wb) noexcept
    {
        assert( c != nullptr && a != nullptr && b!= nullptr );
        
        __m256 ymm0, ymm1, ymm2;

        constexpr SizeType block_size = 8;
        
        auto ci = c;
        auto ai = a;
        auto bi = b;
        for( SizeType i = 0; i < nc[0]; i += block_size, ci += wc[0] * block_size, ai += wa[0] * block_size, bi += wb[0] * block_size ){
            auto cj = ci;
            auto aj = ai;
            auto bj = bi;
            for( SizeType j = 0; j < nc[1]; j += block_size, cj += wc[1] * block_size, aj += wa[1] * block_size, bj += wb[1] * block_size ){
                add_8x8(cj, nc, wc, aj, na, wa, bj, nb, wb);
            }
        }

    }

} // namespace boost::numeric::ublas::simd


#endif // BOOST_NUMERIC_UBLAS_SIMD_ADD_8nX8n_HPP
