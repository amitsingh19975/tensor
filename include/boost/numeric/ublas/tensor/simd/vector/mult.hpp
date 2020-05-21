#if !defined(VECTOR_MULT_COL_HPP)
#define VECTOR_MULT_COL_HPP
#include <boost/numeric/ublas/functional.hpp>

namespace boost::numeric::ublas
{
    using first_order = column_major;
    using last_order = row_major;
} // namespace boost::numeric::ublas


namespace boost::numeric::ublas::simd{
    
    template<typename F>
    struct kernel;

} // namespace boost::numeric::ublas::simd


#include <boost/numeric/ublas/tensor/simd/vector/mult_col.hpp>
#include <boost/numeric/ublas/tensor/simd/vector/mult_row.hpp>

namespace boost::numeric::ublas::simd{
    
    template<typename F>
    struct partition;

    template<>
    struct partition<first_order>{
        inline constexpr size_t k(size_t K) const noexcept{
            if( K < 2000 ){
                return 96;
            }else{
                return 8;
            }
        }
        inline constexpr size_t m(size_t M) const noexcept{
            if( M > 2000 ){
                auto nm = M / 4;
                auto rem = nm % 8;
                return nm - rem;
            }else{
                return 88;
            }
        }
        inline constexpr size_t n(size_t) const noexcept{
            return 1;
        }
    };

    template<>
    struct partition<last_order>{
        inline constexpr size_t k(size_t K) const noexcept{
            if( K < 2000 ){
                return 88;
            }else{
                auto nm = K / 4;
                auto rem = nm % 8;
                return nm - rem;
            }
        }
        inline constexpr size_t m(size_t M) const noexcept{
            if( M > 2000 ){
                auto nm = M / 4;
                auto rem = nm % 8;
                return nm - rem;
            }else{
                return 8;
            }
        }
        inline constexpr size_t n(size_t) const noexcept{
            return 1;
        }
    };


    template<typename PartitionType = partition<first_order>, typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void mtv(
        float* __restrict__ c, SizeType const* __restrict__ nc, SizeType const* __restrict__ wc,
        float const* __restrict__ a, SizeType const* __restrict__ na, SizeType const* __restrict__ wa,
        float const* __restrict__ b, SizeType const* __restrict__ nb, SizeType const* __restrict__ wb,
        kernel<first_order> ker = {}, PartitionType par = {}
    ) noexcept{

        auto ai = a;
        auto bi = b;
        auto ci = c;
        
        auto const K = par.k(na[1]);
        auto const M = par.m(na[0]);

        #pragma omp parallel for schedule(dynamic)
        for( auto i = 0ul; i < na[0]; i += M ){
            auto const ib = std::min(na[0] - i, M);

            auto ak = a + i * wa[0];
            auto bk = b;
            auto ck = c + i * wc[0];

            for(auto k = 0ul; k < nb[0]; k += K){
                auto const kb = std::min(nb[0] - k, K);
                auto aii = ak;
                auto bii = bk;
                auto cii = ck;


                SizeType const nta[] = { ib, kb};

                SizeType const ntb[] = { kb, 1};

                SizeType const ntc[] = { ib, 1};

                {
                    ker(
                        ck, ntc, wc,
                        ak, nta, wa,
                        bk, ntb, wb
                    );
                }
                ak += wa[1] * kb;
                bk += kb;
            }
        }

    }

    template<typename PartitionType = partition<last_order>, typename SizeType>
    BOOST_UBLAS_TENSOR_ALWAYS_INLINE
    void mtv(
        float* __restrict__ c, SizeType const* __restrict__ nc, SizeType const* __restrict__ wc,
        float const* __restrict__ a, SizeType const* __restrict__ na, SizeType const* __restrict__ wa,
        float const* __restrict__ b, SizeType const* __restrict__ nb, SizeType const* __restrict__ wb,
        kernel<last_order> ker = {}, PartitionType par = {}
    ) noexcept{

        auto ai = a;
        auto bi = b;
        auto ci = c;
        
        auto const K = par.k(na[1]);
        auto const M = par.m(na[0]);
        
        #pragma omp parallel for schedule(dynamic)
        for( auto i = 0ul; i < na[1]; i += M ){
            auto ib = std::min( na[1] - i, M );
            auto ak = ai + wa[0] * i;
            auto bk = bi;
            auto ck = ci + wc[0] * i;

            for( auto k = 0ul; k < na[0]; k += K){
                auto kb = std::min( na[0] - k, K );
                SizeType const nta[] = { kb, ib };
                SizeType const ntb[] = { 1, kb };
                SizeType const ntc[] = { 1, ib };


                ker(
                    ck, ntc, wc,
                    ak, nta, wa,
                    bk, ntb, wb
                );

                ak += wa[1] * kb;
                bk += kb;
            }

        }

    }

} // namespace boost::numeric::ublas::simd


#endif // VECTOR_MULT_COL_HPP
