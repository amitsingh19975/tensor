#ifndef _BOOST_UBLAS_TENSOR_PARALLEL_PARTITIONER_HPP
#define _BOOST_UBLAS_TENSOR_PARALLEL_PARTITIONER_HPP

#include "jthread.hpp"
#include <array>
#include "detail/range_vector.hpp"

namespace boost::numeric::ublas::parallel{

    struct proportional_split{
        using size_type = size_t;

        proportional_split() = default;
        proportional_split(size_type l, size_type r)
            : m_left(l), m_right(r)
        {}

        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr size_type left() const noexcept { return m_left; }
        [[nodiscard]] BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr size_type right() const noexcept { return m_right; }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr operator split() const noexcept { return split{}; }

    private:
        size_type m_left{1};
        size_type m_right{1};
    };

    template<typename Partition>
    struct partition_base{
        using self_type     = Partition;
        using split_type    = split;

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr self_type& self() noexcept{
            return *static_cast<Partition*>(this);
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr self_type const& self() const noexcept{
            return *static_cast<Partition const*>(this);
        }

        template <typename Range> BOOST_UBLAS_TENSOR_ALWAYS_INLINE  split_type get_split() { return split(); }

        template<typename StartType, typename Range>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE void workbalance( StartType &start, Range& range ){
            start.run(range);
        }

        template<typename StartType, typename Range>
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr void execute(StartType& start, Range& range) const noexcept{
            while(range.is_divisible() && self().is_divisible()){
                typename Partition::split_type split_obj = self().template get_split<Range>();
            }
            self().workbalance(start,range);
        } 
    };

    template<typename Partition>
    struct adaptive_mode : partition_base<Partition> {
        using base_type     = partition_base<Partition>;
        using size_type     = size_t;
        using partition_type= Partition;
        static constexpr unsigned factor = 1u;

        constexpr adaptive_mode()
            : m_divisor(jthread::hardware_concurrency() * partition_type::factor)
        {}

        constexpr adaptive_mode( adaptive_mode& src, split )
            : m_divisor( do_split(src,split{}) )
        {}

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr size_type do_split(adaptive_mode& src, split){
            return m_divisor / unsigned(2);
        }

    protected:
        size_type m_divisor;
    };

    template<typename Range, typename = void > 
    struct proportion_helper{
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr auto get_split(size_t){ return proportional_split{}; }
    };

    template<typename Range> 
    struct proportion_helper<Range, typename std::enable_if_t< Range::is_splittable_in_proportion, void >::type >{
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE static constexpr auto get_split( size_t n ){ 
            auto right = n / 2;
            auto left = n - right;
            return proportional_split{right, left};
        }
    };

    template<typename Partition>
    struct proportional_mode : adaptive_mode<Partition> {
        using partition_type    = Partition;
        using base_type         = adaptive_mode<Partition>;
        using base_type::self;

        constexpr proportional_mode() : base_type() {}
        constexpr proportional_mode( proportional_mode& src, split )
         : base_type( src, split{} )
        {} 
        
        constexpr proportional_mode( proportional_mode& src, proportional_split& split_obj ){
            self().m_divisor = do_split(src,split_obj);
        } 
        
        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr size_t do_split( proportional_mode& src, proportional_split& s ) noexcept{
            size_t proption = s.right() * partition_type::factor;
            proption = (proption + partition_type::factor / 2 ) & (0ul - partition_type::factor);
            src.m_divisor -= proption;
            return proption;
        }

        BOOST_UBLAS_TENSOR_ALWAYS_INLINE constexpr bool is_divisible() const noexcept{
            return self().m_divisor > partition_type::factor;
        }

        template <typename Range>
        proportional_split get_split() {
            return proportion_helper<Range>::get_split( self().m_divisor / partition_type::factor );
        }
    };

    template<typename Mode>
    struct dynamic_grainsize_mode : Mode{
        using base_type = Mode;
        using base_type::self;
        static constexpr auto max_pool_size = TENSOR_MAX_POOL_SIZE;

    private:

    };

} // namespace boost::numeric::ublas::parallel


#endif // _BOOST_UBLAS_TENSOR_PARALLEL_PARTITIONER_HPP