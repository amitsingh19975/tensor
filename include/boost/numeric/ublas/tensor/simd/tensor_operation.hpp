#ifndef _BOOST_UBLAS_TENSOR_SIMD_TENSOR_OPERATION_HPP
#define _BOOST_UBLAS_TENSOR_SIMD_TENSOR_OPERATION_HPP

#include "mat_storage.hpp"
#include <cassert>
#include <type_traits>

namespace boost::numeric::ublas::simd::detail::recursive{
    
    /** @brief Computes the tensor-times-tensor product for q contraction modes
     *
     * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir+q] * B[j1,...,js+q]  )
     *
     * nc[x]         = na[phia[x]  ] for 1 <= x <= r
     * nc[r+x]       = nb[phib[x]  ] for 1 <= x <= s
     * na[phia[r+x]] = nb[phib[s+x]] for 1 <= x <= q
     *
     * @note is used in function ttt
     *
     * @param k  zero-based recursion level starting with 0
     * @param r  number of non-contraction indices of A
     * @param s  number of non-contraction indices of B
     * @param q  number of contraction indices with q > 0
     * @param phia pointer to the permutation tuple of length q+r for A
     * @param phib pointer to the permutation tuple of length q+s for B
     * @param c  pointer to the output tensor C with rank(A)=r+s
     * @param nc pointer to the extents of tensor C
     * @param wc pointer to the strides of tensor C
     * @param a  pointer to the first input tensor with rank(A)=r+q
     * @param na pointer to the extents of the first input tensor A
     * @param wa pointer to the strides of the first input tensor A
     * @param b  pointer to the second input tensor B with rank(B)=s+q
     * @param nb pointer to the extents of the second input tensor B
     * @param wb pointer to the strides of the second input tensor B
    */

    template <class PointerOut, class PointerIn1, class PointerIn2, class SizeType>
    void ttt(SizeType const k,
            SizeType const r, SizeType const s, SizeType const q,
            SizeType const*const phia, SizeType const*const phib,
            PointerOut c, SizeType const*const nc, SizeType const*const wc,
            PointerIn1 a, SizeType const*const na, SizeType const*const wa,
            PointerIn2 b, SizeType const*const nb, SizeType const*const wb)
    {
        if(k < r)
        {
            assert(nc[k] == na[phia[k]-1]);
            for(size_t ic = 0u; ic < nc[k]; a += wa[phia[k]-1], c += wc[k], ++ic)
                ttt(k+1, r, s, q,  phia,phib,  c, nc, wc,   a, na, wa,   b, nb, wb);
        }
        else if(k < r+s)
        {
            assert(nc[k] == nb[phib[k-r]-1]);
            for(size_t ic = 0u; ic < nc[k]; b += wb[phib[k-r]-1], c += wc[k], ++ic)
                ttt(k+1, r, s, q,  phia, phib,    c, nc, wc,   a, na, wa,   b, nb, wb);
        }
        else if(k < r+s+q-1)
        {
            assert(na[phia[k-s]-1] == nb[phib[k-r]-1]);
            for(size_t ia = 0u; ia < na[phia[k-s]-1]; a += wa[phia[k-s]-1], b += wb[phib[k-r]-1], ++ia)
                ttt(k+1, r, s, q,  phia, phib,  c, nc, wc,   a, na, wa,   b, nb, wb);
        }
        else
        {
            assert(na[phia[k-s]-1] == nb[phib[k-r]-1]);
            auto sa = vec_storage<256,float>(a, na[phia[k-s]-1], wa[phia[k-s]-1]);
            auto sb = vec_storage<256,float>(b, na[phia[k-s]-1], wb[phia[k-s]-1]);
            for(size_t ia = 0u; ia < sa.size(); ++ia)
                *c += dot_prod(sa.at(ia), sb.at(ia));
        }
    }


    template <class PointerOut, class PointerIn1, class PointerIn2, class SizeType>
    void outer_2x2(SizeType const pa,
                PointerOut c, SizeType const*const   , SizeType const*const wc,
                PointerIn1 a, SizeType const*const na, SizeType const*const wa,
                PointerIn2 b, SizeType const*const nb, SizeType const*const wb)
    {
        //	assert(rc == 3);
        //	assert(ra == 1);
        //	assert(rb == 1);

        for(auto ib1 = 0u; ib1 < nb[1]; b += wb[1], c += wc[pa+1], ++ib1) {
            auto c2 = c;
            auto b0 = b;
            for(auto ib0 = 0u; ib0 < nb[0]; b0 += wb[0], c2 += wc[pa], ++ib0) {
                const auto b = *b0;
                auto c1 = c2;
                auto a1 = a;
                for(auto ia1 = 0u; ia1 < na[1]; a1 += wa[1], c1 += wc[1], ++ia1) {
                    auto a0 = a1;
                    auto c0 = c1;
                    for(SizeType ia0 = 0u; ia0 < na[0]; a0 += wa[0], c0 += wc[0], ++ia0)
                        *c0 = *a0 * b;
                }
            }
        }
    }

    /** @brief Computes the outer product of two tensors
     *
     * Implements C[i1,...,ip,j1,...,jq] = A[i1,i2,...,ip] * B[j1,j2,...,jq]
     *
     * @note called by outer
     *
     *
     * @param[in]  pa number of dimensions (rank) of the first input tensor A with pa > 0
     *
     * @param[in]  rc recursion level for C that starts with pc-1
     * @param[out] c  pointer to the output tensor
     * @param[in]  nc pointer to the extents of output tensor c
     * @param[in]  wc pointer to the strides of output tensor c
     *
     * @param[in]  ra recursion level for A that starts with pa-1
     * @param[in]  a  pointer to the first input tensor
     * @param[in]  na pointer to the extents of the first input tensor a
     * @param[in]  wa pointer to the strides of the first input tensor a
     *
     * @param[in]  rb recursion level for B that starts with pb-1
     * @param[in]  b  pointer to the second input tensor
     * @param[in]  nb pointer to the extents of the second input tensor b
     * @param[in]  wb pointer to the strides of the second input tensor b
    */
    template<class PointerOut, class PointerIn1, class PointerIn2, class SizeType>
    void outer(SizeType const pa,
            SizeType const rc, PointerOut c, SizeType const*const nc, SizeType const*const wc,
            SizeType const ra, PointerIn1 a, SizeType const*const na, SizeType const*const wa,
            SizeType const rb, PointerIn2 b, SizeType const*const nb, SizeType const*const wb)
    {
        if(rb > 1)
            for(auto ib = 0u; ib < nb[rb]; b += wb[rb], c += wc[rc], ++ib)
                outer(pa, rc-1, c, nc, wc,    ra, a, na, wa,    rb-1, b, nb, wb);
        else if(ra > 1)
            for(auto ia = 0u; ia < na[ra]; a += wa[ra], c += wc[ra], ++ia)
                outer(pa, rc-1, c, nc, wc,   ra-1, a, na, wa,   rb, b, nb, wb);
        else
            outer_2x2(pa, c, nc, wc,   a, na, wa,    b, nb, wb); //assert(ra==1 && rb==1 && rc==3);
    }


    /** @brief Computes the outer product with permutation tuples
     *
     * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir] * B[j1,...,js]  )
     *
     * nc[x]         = na[phia[x]] for 1 <= x <= r
     * nc[r+x]       = nb[phib[x]] for 1 <= x <= s
     *
     * @note maybe called by ttt function
     *
     * @param k  zero-based recursion level starting with 0
     * @param r  number of non-contraction indices of A
     * @param s  number of non-contraction indices of B
     * @param phia pointer to the permutation tuple of length r for A
     * @param phib pointer to the permutation tuple of length s for B
     * @param c  pointer to the output tensor C with rank(A)=r+s
     * @param nc pointer to the extents of tensor C
     * @param wc pointer to the strides of tensor C
     * @param a  pointer to the first input tensor with rank(A)=r
     * @param na pointer to the extents of the first input tensor A
     * @param wa pointer to the strides of the first input tensor A
     * @param b  pointer to the second input tensor B with rank(B)=s
     * @param nb pointer to the extents of the second input tensor B
     * @param wb pointer to the strides of the second input tensor B
    */

    template <class PointerOut, class PointerIn1, class PointerIn2, class SizeType>
    void outer(SizeType const k,
            SizeType const r, SizeType const s,
            SizeType const*const phia, SizeType const*const phib,
            PointerOut c, SizeType const*const nc, SizeType const*const wc,
            PointerIn1 a, SizeType const*const na, SizeType const*const wa,
            PointerIn2 b, SizeType const*const nb, SizeType const*const wb)
    {
        if(k < r)
        {
            assert(nc[k] == na[phia[k]-1]);
            for(size_t ic = 0u; ic < nc[k]; a += wa[phia[k]-1], c += wc[k], ++ic)
                outer(k+1, r, s,   phia,phib,  c, nc, wc,   a, na, wa,   b, nb, wb);
        }
        else if(k < r+s-1)
        {
            assert(nc[k] == nb[phib[k-r]-1]);
            for(size_t ic = 0u; ic < nc[k]; b += wb[phib[k-r]-1], c += wc[k], ++ic)
                outer(k+1, r, s, phia, phib,    c, nc, wc,   a, na, wa,   b, nb, wb);
        }
        else
        {
            assert(nc[k] == nb[phib[k-r]-1]);
            for(size_t ic = 0u; ic < nc[k]; b += wb[phib[k-r]-1], c += wc[k], ++ic)
                *c = *a * *b;
        }
    }



} // namespace boost::numeric::ublas::simd::detail::recursive


namespace boost::numeric::ublas::simd{

    /** @brief Computes the tensor-times-tensor product
     *
     * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir+q] * B[j1,...,js+q]  )
     *
     * @note calls detail::recursive::ttt or ttm or ttv or inner or outer
     *
     * nc[x]         = na[phia[x]  ] for 1 <= x <= r
     * nc[r+x]       = nb[phib[x]  ] for 1 <= x <= s
     * na[phia[r+x]] = nb[phib[s+x]] for 1 <= x <= q
     *
     * @param[in]  pa number of dimensions (rank) of the first input tensor a with pa > 0
     * @param[in]  pb number of dimensions (rank) of the second input tensor b with pb > 0
     * @param[in]	 q  number of contraction dimensions with pa >= q and pb >= q and q >= 0
     * @param[in]	 phia pointer to a permutation tuple for the first input tensor a
     * @param[in]	 phib pointer to a permutation tuple for the second input tensor b
     * @param[out] c  pointer to the output tensor with rank p-1
     * @param[in]  nc pointer to the extents of tensor c
     * @param[in]  wc pointer to the strides of tensor c
     * @param[in]  a  pointer to the first input tensor
     * @param[in]  na pointer to the extents of input tensor a
     * @param[in]  wa pointer to the strides of input tensor a
     * @param[in]  b  pointer to the second input tensor
     * @param[in]  nb pointer to the extents of input tensor b
     * @param[in]  wb pointer to the strides of input tensor b
    */

    template <class PointerIn1, class PointerIn2, class PointerOut, class SizeType>
    void ttt(SizeType const pa, SizeType const pb, SizeType const q,
            SizeType const*const phia, SizeType const*const phib,
            PointerOut c, SizeType const*const nc, SizeType const*const wc,
            PointerIn1 a, SizeType const*const na, SizeType const*const wa,
            PointerIn2 b, SizeType const*const nb, SizeType const*const wb)
    {
        static_assert( std::is_pointer<PointerOut>::value & std::is_pointer<PointerIn1>::value & std::is_pointer<PointerIn2>::value,
                    "Static error in boost::numeric::ublas::ttm: Argument types for pointers are not pointer types.");

        if( pa == 0 || pb == 0)
            throw std::length_error("Error in boost::numeric::ublas::ttt: tensor order must be greater zero.");

        if( q > pa && q > pb)
            throw std::length_error("Error in boost::numeric::ublas::ttt: number of contraction must be smaller than or equal to the tensor order.");


        SizeType const r = pa - q;
        SizeType const s = pb - q;

        if(c == nullptr || a == nullptr || b == nullptr)
            throw std::length_error("Error in boost::numeric::ublas::ttm: Pointers shall not be null pointers.");

        for(auto i = 0ul; i < r; ++i)
            if( na[phia[i]-1] != nc[i] )
                throw std::length_error("Error in boost::numeric::ublas::ttt: dimensions of lhs and res tensor not correct.");

        for(auto i = 0ul; i < s; ++i)
            if( nb[phib[i]-1] != nc[r+i] )
                throw std::length_error("Error in boost::numeric::ublas::ttt: dimensions of rhs and res not correct.");

        for(auto i = 0ul; i < q; ++i)
            if( nb[phib[s+i]-1] != na[phia[r+i]-1] )
                throw std::length_error("Error in boost::numeric::ublas::ttt: dimensions of lhs and rhs not correct.");


        if(q == 0ul)
            detail::recursive::outer(SizeType{0},r,s,  phia,phib, c,nc,wc, a,na,wa, b,nb,wb);
        else
            detail::recursive::ttt(SizeType{0},r,s,q,  phia,phib, c,nc,wc, a,na,wa, b,nb,wb);
    }

} // namespace boost::numeric::ublas::simd


#endif