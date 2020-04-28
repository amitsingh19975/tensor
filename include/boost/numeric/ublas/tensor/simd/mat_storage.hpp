#ifndef _BOOST_UBLAS_TENSOR_SIMD_MAT_STORAGE_HPP
#define _BOOST_UBLAS_TENSOR_SIMD_MAT_STORAGE_HPP

#include "vector_storage.hpp"
#include <boost/numeric/ublas/tensor/tensor.hpp>
namespace boost::numeric::ublas::simd{

    namespace op{
        struct transpose{};
        struct no_op{};
    } // namespace op    

} // namespace boost::numeric::ublas::simd

namespace boost::numeric::ublas::simd{
    
    template<typename T, size_t B = 256, typename O = op::no_op>
    struct mat_storage : basic_tensor< mat_storage<T,B,O> >{
        using self_type                 = mat_storage<T,B,O>;
        using super_type                = basic_tensor< self_type >;

        template<class derived_type>
        using tensor_expression_type    = typename super_type::template tensor_expression_type<derived_type>;

        template<class derived_type>
        using matrix_expression_type    = typename super_type::template matrix_expression_type<derived_type>;

        template<class derived_type>
        using vector_expression_type    = typename super_type::template vector_expression_type<derived_type>;

        using array_type                = typename tensor_traits<self_type>::container_type;
        using layout_type               = typename tensor_traits<self_type>::layout_type;

        using size_type                 = typename array_type::size_type;
        using difference_type           = typename array_type::difference_type;
        using value_type                = T;

        using reference                 = typename array_type::reference;
        using const_reference           = typename array_type::const_reference;

        using pointer                   = typename array_type::pointer;
        using const_pointer             = typename array_type::const_pointer;

        using iterator                  = typename array_type::iterator;
        using const_iterator            = typename array_type::const_iterator;

        using reverse_iterator          = typename array_type::reverse_iterator;
        using const_reverse_iterator    = typename array_type::const_reverse_iterator;

        using tensor_temporary_type     = self_type;
        using storage_category          = dense_tag;

        using extents_type              = typename tensor_traits<self_type>::extents_type;
        using strides_type              = strides_t<extents_type,layout_type>;

        using matrix_type               = typename super_type::matrix_type;
        using vector_type               = typename super_type::vector_type;
        using operation_type            = O;
        using simd_type                 = basic_simd_type<B,T>;
        


        constexpr static auto bsz = detail::block_size_v<B,T>;

        mat_storage() = default;

        template<typename U = operation_type, std::enable_if_t< std::is_same_v< U, op::transpose >, int > = 0>
        mat_storage( size_type row, size_type col ) 
            : super_type( extents_type{detail::get_array_size<B,T>(col),row} )
        {}

        template<typename U = O, std::enable_if_t< std::is_same_v< U, op::no_op >, long > = 0>
        mat_storage( size_type row, size_type col ) 
            : super_type( extents_type{row, detail::get_array_size<B,T>(col)} )
        {}

        mat_storage( value_type const* data, size_type row, size_type col, size_type wr, size_type wc)
            : mat_storage(row,col)
        {
            if constexpr( std::is_same_v< operation_type, op::transpose > ){
                set_data_transpose(data,row,col, wr, wc);
            }else{
                set_data(data,row,col, wr, wc);
            }
        }

        mat_storage( value_type const* data, size_type row, size_type col )
            : mat_storage(row,col)
        {
            auto s = strides_type( extents_type{row,col} );
            if constexpr( !std::is_same_v< operation_type, op::transpose > ){
                set_data_transpose(data,row,col, s[0], s[1]);
            }else{
                set_data(data,row,col, s[0], s[1]);
            }
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto row() const noexcept{ 
            return super_type::extents_[0] ;
        }
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto col() const noexcept{ 
            return super_type::extents_[1];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto& element( size_type k ) noexcept{
            auto block_pos = k / bsz;
            auto pos = k % bsz;
            return super_type::data_[block_pos][pos];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto const& element( size_type k ) const noexcept{
            auto block_pos = k / bsz;
            auto pos = k % bsz;
            return super_type::data_[block_pos][pos];
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr size_type size() const noexcept {
            return super_type::data_.size();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr auto elements() const noexcept{
            return super_type::data_.size() * bsz;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE iterator begin() noexcept{
            return super_type::data_.begin();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE iterator end() noexcept{
            return super_type::data_.end();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE const_iterator begin() const noexcept{
            return super_type::data_.begin();
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE const_iterator end() const noexcept{
            return super_type::data_.end();
        }
    private:

        constexpr void set_data(value_type const* data, size_type row, size_type col, size_type wr, size_type wc) noexcept{
            auto curr_data = reinterpret_cast<value_type*>(super_type::data());

            auto rem = col % bsz;
            auto chunk = col / bsz;

            auto pi = data;
            
            if ( rem != 0 ){

                for( auto i = size_type(0); i < row; pi += wr, ++i ){
                    auto pj = pi;
                    for( auto j = size_type(0); j < chunk; pj += wc * bsz, ++j ){
                        *reinterpret_cast<simd_type*>(curr_data) = detail::load<B,T>{}(pj,bsz,wc);
                        curr_data += bsz;
                    }
                    *reinterpret_cast<simd_type*>(curr_data) = detail::load<B,T>{}(pj,rem,wc);
                    curr_data += bsz;
                }

            }else{
                for( auto i = size_type(0); i < row; pi += wr, ++i ){
                    auto pj = pi;
                    for( auto j = size_type(0); j < chunk; pj += wc * bsz, ++j ){
                        *reinterpret_cast<simd_type*>(curr_data) = detail::load<B,T>{}(pj,bsz,wc);
                        curr_data += bsz;
                    }
                }
            }
        }

        constexpr void set_data_transpose(value_type const* data, size_type row, size_type col, size_type wr, size_type wc) noexcept{
            auto curr_data = reinterpret_cast<value_type*>(super_type::data_.data());
            
            auto rem = row % bsz;
            auto chunk = row / bsz;
            
            auto pj = data;

            if ( rem != 0 ){
                
                for( auto j = size_type(0); j < col; pj += wc, ++j ){
                    auto pi = pj;
                    for( auto i = size_type(0); i < chunk; pi += wr, ++i ){
                        *reinterpret_cast<simd_type*>(curr_data) = detail::load<B,T>{}(pi,bsz, wr);
                        curr_data += bsz;
                    }
                    
                    *reinterpret_cast<simd_type*>(curr_data) = detail::load<B,T>{}(pi,bsz - rem, wr);
                    curr_data += rem;
                }

            }else{
                for( auto j = size_type(0); j < col; pj += wc, ++j ){
                    auto pi = pj;
                    for( auto i = size_type(0); i < chunk; pi += wr, ++i ){
                        *reinterpret_cast<simd_type*>(curr_data) = detail::load<B,T>{}(pi,bsz, wr);
                        curr_data += bsz;
                    }
                }
            }
        }
    };


} // namespace boost::numeric::ublas::simd


namespace boost::numeric::ublas{

    template<size_t B, typename T, typename O>
    struct tensor_traits< simd::mat_storage<T,B,O> >{
        using container_type= std::vector< simd::basic_simd_type<B,T> >;
        using extents_type  = dynamic_extents<2>;
        using layout_type   = first_order;
        using container_tag = dynamic_tensor_tag;
    };

}

#endif