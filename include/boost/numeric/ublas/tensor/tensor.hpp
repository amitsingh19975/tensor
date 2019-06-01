//  Copyright (c) 2018-2019
//  Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//

/// \file tensor.hpp Definition for the tensor template class

#ifndef BOOST_UBLAS_TENSOR_IMPL_HPP
#define BOOST_UBLAS_TENSOR_IMPL_HPP

#include <boost/config.hpp>
#include <boost/yap/yap.hpp>

#include <initializer_list>

#include "algorithms.hpp"
#include "extents.hpp"
#include "index.hpp"
#include "strides.hpp"
#include "tensor_core.hpp"
#include "tensor_expression.hpp"

namespace boost {
namespace numeric {
namespace ublas {

template <class T, class F, class A> class tensor;

template <class T, class F, class A> class matrix;

template <class T, class A> class vector;

template <class A, typename B> auto static_tensor_cast(B &);

using namespace boost::hana::literals;

template <class T, class F = first_order,
          class A = std::vector<T, std::allocator<T>>>
class tensor : public detail::tensor_expression<
                   boost::yap::expr_kind::terminal,
                   boost::hana::tuple<detail::tensor_core<T, F, A>>> {
  static_assert(std::is_same<F, first_order>::value ||
                    std::is_same<F, last_order>::value,
                "boost::numeric::tensor template class only supports first- or "
                "last-order storage formats.");

  using self_type = tensor<T, F, A>;
  using core_type = detail::tensor_core<T, F, A>;
  using expression_type = detail::tensor_expression<
      boost::yap::expr_kind::terminal,
      boost::hana::tuple<detail::tensor_core<T, F, A>>>;

public:
  template <class derived_type>
  using matrix_expression_type = matrix_expression<derived_type>;

  template <class derived_type>
  using vector_expression_type = vector_expression<derived_type>;

  using array_type = A;
  using layout_type = F;

  using size_type = typename array_type::size_type;
  using difference_type = typename array_type::difference_type;
  using value_type = typename array_type::value_type;

  using reference = typename array_type::reference;
  using const_reference = typename array_type::const_reference;

  using pointer = typename array_type::pointer;
  using const_pointer = typename array_type::const_pointer;

  using iterator = typename array_type::iterator;
  using const_iterator = typename array_type::const_iterator;

  using reverse_iterator = typename array_type::reverse_iterator;
  using const_reverse_iterator = typename array_type::const_reverse_iterator;

  using tensor_temporary_type = self_type;
  using storage_category = dense_tag;

  using strides_type = basic_strides<std::size_t, layout_type>;
  using extents_type = shape;

  using matrix_type = matrix<value_type, layout_type, array_type>;
  using vector_type = vector<value_type, array_type>;

  /** @brief Constructs a tensor.
   *
   * @note the tensor is empty.
   * @note the tensor needs to reshaped for further use.
   *
   */
  BOOST_UBLAS_INLINE
  constexpr tensor() : expression_type() {}

  /** @brief Constructs a tensor with an initializer list
   *
   * By default, its elements are initialized to 0.
   *
   * @code tensor<float> A{4,2,3}; @endcode
   *
   * @param l initializer list for setting the dimension extents of the tensor
   */
  explicit BOOST_UBLAS_INLINE tensor(std::initializer_list<size_type> l)
      : expression_type() {
    auto e = extents_type{std::move(l)};
    this->elements[0_c].extents_ = e;
    this->elements[0_c].strides_ = strides_type{e};
    this->elements[0_c].data_.resize(e.product());
  }

  /** @brief Constructs a tensor with a \c shape
   *
   * By default, its elements are initialized to 0.
   *
   * @code tensor<float> A{extents{4,2,3}}; @endcode
   *
   * @param s initial tensor dimension extents
   */
  explicit BOOST_UBLAS_INLINE tensor(extents_type const &s)
      : expression_type() {
    this->elements[0_c].extents_ = s;
    this->elements[0_c].strides_ = strides_type{s};
    this->elements[0_c].data_.resize(s.product());
  }

  /** @brief Constructs a tensor with a \c shape and initiates it with
   * one-dimensional data
   *
   * @code tensor<float> A{extents{4,2,3}, array }; @endcode
   *
   *
   *  @param s initial tensor dimension extents
   *  @param a container of \c array_type that is copied according to the
   * storage layout
   */
  BOOST_UBLAS_INLINE
  tensor(extents_type const &s, const array_type &a) : expression_type() {

    this->elements[0_c].extents_ = s;
    this->elements[0_c].strides_ = strides_type{s};
    this->elements[0_c].data_ = a;

    if (this->elements[0_c].extents_.product() !=
        this->elements[0_c].data_.size())
      throw std::runtime_error(
          "Error in boost::numeric::ublas::tensor: size of provided data and "
          "specified extents do not match.");
  }

  /** @brief Constructs a tensor using a shape tuple and initiates it with a
   * value.
   *
   *  @code tensor<float> A{extents{4,2,3}, 1 }; @endcode
   *
   *  @param e initial tensor dimension extents
   *  @param i initial value of all elements of type \c value_type
   */
  BOOST_UBLAS_INLINE
  tensor(extents_type const &e, const value_type &i) : expression_type() {
    this->elements[0_c].extents_ = e;
    this->elements[0_c].strides_ = strides_type{e};
    this->elements[0_c].data_.resize(e.product(), i);
  }

  /** @brief Constructs a tensor from another tensor
   *
   *  @param v tensor to be copied.
   */
  BOOST_UBLAS_INLINE
  tensor(const tensor &v) : expression_type() { this->elements = v.elements; }

  /** @brief Constructs a tensor from another tensor
   *
   *  @param v tensor to be moved.
   */
  BOOST_UBLAS_INLINE
  tensor(tensor &&v) : expression_type() {
    this->elements = std::move(v.elements);
  }

  /** @brief Constructs a tensor with a matrix
   *
   * \note Initially the tensor will be two-dimensional.
   *
   *  @param v matrix to be copied.
   */
  BOOST_UBLAS_INLINE
  tensor(const matrix_type &v) : expression_type() {

    this->elements[0_c].data_ = v.data();

    if (!this->elements[0_c].data_.empty()) {
      this->elements[0_c].extents_ = extents_type{v.size1(), v.size2()};
      this->elements[0_c].strides_ = strides_type{this->elements[0_c].extents_};
    }
  }

  /** @brief Constructs a tensor with a matrix
   *
   * \note Initially the tensor will be two-dimensional.
   *
   *  @param v matrix to be moved.
   */
  BOOST_UBLAS_INLINE
  tensor(matrix_type &&v) : expression_type() {
    if (v.size1() * v.size2() != 0) {
      this->elements[0_c].extents_ = extents_type{v.size1(), v.size2()};
      this->elements[0_c].strides_ = strides_type(this->elements[0_c].extents_);
      this->elements[0_c].data_ = std::move(v.data());
    }
  }

  /** @brief Constructs a tensor using a \c vector
   *
   * @note It is assumed that vector is column vector
   * @note Initially the tensor will be one-dimensional.
   *
   *  @param v vector to be copied.
   */
  BOOST_UBLAS_INLINE
  tensor(const vector_type &v) : expression_type() {

    this->elements[0_c].data_ = v.data();

    if (!this->elements[0_c].data_.empty()) {
      this->elements[0_c].extents_ =
          extents_type{this->elements[0_c].data_.size(), 1};
      this->elements[0_c].strides_ = strides_type(this->elements[0_c].extents_);
    }
  }

  /** @brief Constructs a tensor using a \c vector
   *
   *  @param v vector to be moved.
   */
  BOOST_UBLAS_INLINE
  tensor(vector_type &&v) : expression_type() {
    if (v.size() != 0) {
      this->elements[0_c].extents_ = extents_type{v.size(), 1};
      this->elements[0_c].strides_ = strides_type(this->elements[0_c].extents_);
      this->elements[0_c].data_ = std::move(v.data());
    }
  }

  /** @brief Constructs a tensor with another tensor with a different layout
   *
   * @param other tensor with a different layout to be copied.
   */
  BOOST_UBLAS_INLINE
  template <class other_layout>
  tensor(const tensor<value_type, other_layout> &other) : expression_type() {

    this->elements[0_c].extents_ = other.elements[0_c].extents_;
    this->elements[0_c].strides_ = strides_type{other.elements[0_c].extents_};
    this->elements[0_c].data_.resize(other.elements[0_c].extents_.product());

    copy(this->rank(), this->elements[0_c].extents_.data(),
         this->elements[0_c].data_.data(), this->elements[0_c].strides().data(),
         other.elements[0_c].data_.data(),
         other.elements[0_c].strides().data());
  }

  /** @brief Constructs a tensor with an tensor expression
   *
   * @code tensor<float> A = B + 3 * C; @endcode
   *
   * @note type must be specified of tensor must be specified.
   * @note dimension extents are extracted from tensors within the expression.
   *
   * @param expr tensor expression
   */
  BOOST_UBLAS_INLINE
  template <boost::yap::expr_kind Kind, typename Tuple>
  tensor(detail::tensor_expression<Kind, Tuple> &expr) : expression_type() {

    auto expr_shape =
        boost::yap::transform(expr, detail::transforms::get_extents{});

    this->elements[0_c].extents_ = expr_shape;
    this->elements[0_c].strides_ = strides_type{expr_shape};

    // Todo : Make this run on a separate device when implementing device policy
    // #pragma omp parallel for

    this->elements[0_c].data_.resize(expr_shape.product());

    for (auto i = 0u; i < expr_shape.product(); i++)
      this->elements[0_c].data_[i] = expr(i);
  }

  BOOST_UBLAS_INLINE
  template <boost::yap::expr_kind Kind, typename Tuple>
  tensor(detail::tensor_expression<Kind, Tuple> &&expr) : expression_type() {

    auto expr_shape =
        boost::yap::transform(expr, detail::transforms::get_extents{});

    this->elements[0_c].extents_ = expr_shape;
    this->elements[0_c].strides_ = strides_type{expr_shape};

    // Todo : Make this run on a separate device when implementing device policy
    // #pragma omp parallel for

    this->elements[0_c].data_.resize(expr_shape.product());

    for (auto i = 0u; i < expr_shape.product(); i++)
      this->elements[0_c].data_[i] = std::move(expr(i));
  }

  /** @brief Constructs a tensor with a matrix expression
  *
  * @code tensor<float> A = B + 3 * C; @endcode
  *
  * @note matrix expression is evaluated and pushed into a temporary matrix
  * before assignment.
  * @note extents are automatically extracted from the temporary matrix
  *
  * @param expr matrix expression
  //    */
  BOOST_UBLAS_INLINE
  template <class derived_type>
  tensor(const matrix_expression_type<derived_type> &expr)
      : tensor(matrix_type(expr)) {}

  /** @brief Constructs a tensor with a vector expression
   *
   * @code tensor<float> A = b + 3 * b; @endcode
   *
   * @note matrix expression is evaluated and pushed into a temporary matrix
   * before assignment.
   * @note extents are automatically extracted from the temporary matrix
   *
   * @param expr vector expression
   */
  BOOST_UBLAS_INLINE
  template <class derived_type>
  tensor(const vector_expression_type<derived_type> &expr)
      : tensor(vector_type(expr)) {}

  /** @brief Evaluates the tensor_expression and assigns the results to the
   * tensor
   *
   * @code A = B + C * 2;  @endcode
   *
   * @note rank and dimension extents of the tensors in the expressions must
   * conform with this tensor.
   *
   * @param expr expression that is evaluated.
   */
  BOOST_UBLAS_INLINE
  template <boost::yap::expr_kind Kind, typename Tuple>
  tensor &operator=(const detail::tensor_expression<Kind, Tuple> &expr) {
    expr.eval_to(*this);
    return *this;
  }

  tensor &operator=(tensor other) {
    swap(this->elements[0_c], other.elements[0_c]);
    return *this;
  }

  tensor &operator=(const_reference v) {
    std::fill(this->begin(), this->end(), v);
    return *this;
  }

  /** @brief Returns true if the tensor is empty (\c size==0) */
  BOOST_UBLAS_INLINE
  bool empty() const { return this->elements[0_c].data_.empty(); }

  /** @brief Returns the size of the tensor */
  BOOST_UBLAS_INLINE
  size_type size() const { return this->elements[0_c].data_.size(); }

  /** @brief Returns the size of the tensor */
  BOOST_UBLAS_INLINE
  size_type size(size_type r) const {
    return this->elements[0_c].extents_.at(r);
  }

  /** @brief Returns the number of dimensions/modes of the tensor */
  BOOST_UBLAS_INLINE
  size_type rank() const { return this->elements[0_c].extents_.size(); }

  /** @brief Returns the number of dimensions/modes of the tensor */
  BOOST_UBLAS_INLINE
  size_type order() const { return this->elements[0_c].extents_.size(); }

  /** @brief Returns the strides of the tensor */
  BOOST_UBLAS_INLINE
  strides_type const &strides() const { return this->elements[0_c].strides_; }

  /** @brief Returns the extents of the tensor */
  BOOST_UBLAS_INLINE
  extents_type const &extents() const { return this->elements[0_c].extents_; }

  /** @brief Returns a \c const reference to the container. */
  BOOST_UBLAS_INLINE
  const_pointer data() const { return this->elements[0_c].data_.data(); }

  /** @brief Returns a \c const reference to the container. */
  BOOST_UBLAS_INLINE
  pointer data() { return this->elements[0_c].data_.data(); }

  /** @brief Element access using a single index.
   *
   *  @code auto a = A[i]; @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size()
   */
  BOOST_UBLAS_INLINE
  const_reference operator[](size_type i) const {
    return this->elements[0_c].data_[i];
  }

  /** @brief Element access using a single index.
   *
   *
   *  @code A[i] = a; @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size()
   */
  BOOST_UBLAS_INLINE
  reference operator[](size_type i) { return this->elements[0_c].data_[i]; }

  /** @brief Element access using a multi-index or single-index.
   *
   *
   *  @code auto a = A.at(i,j,k); @endcode or
   *  @code auto a = A.at(i);     @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) ==
   * 0, else 0<= i < this->size(0)
   *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r
   * < this->rank()
   */
  template <class... size_types>
  BOOST_UBLAS_INLINE const_reference at(size_type i, size_types... is) const {
    if constexpr (sizeof...(is) == 0)
      return this->elements[0_c].data_[i];
    else
      return this->elements[0_c]
          .data_[detail::access<0ul>(size_type(0), this->elements[0_c].strides_,
                                     i, std::forward<size_types>(is)...)];
  }

  /** @brief Element access using a multi-index or single-index.
   *
   *
   *  @code A.at(i,j,k) = a; @endcode or
   *  @code A.at(i) = a;     @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) ==
   * 0, else 0<= i < this->size(0)
   *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r
   * < this->rank()
   */
  BOOST_UBLAS_INLINE
  template <class... size_types> reference at(size_type i, size_types... is) {
    if constexpr (sizeof...(is) == 0)
      return this->elements[0_c].data_[i];
    else
      return this->elements[0_c]
          .data_[detail::access<0ul>(size_type(0), this->elements[0_c].strides_,
                                     i, std::forward<size_types>(is)...)];
  }

  /** @brief Element access using a single index.
   *
   *
   *  @code A(i) = a; @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size()
   */
  BOOST_UBLAS_INLINE
  const_reference operator()(size_type i) const {
    return this->elements[0_c].data_[i];
  }

  /** @brief Element access using a single index.
   *
   *  @code A(i) = a; @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size()
   */
  BOOST_UBLAS_INLINE
  reference operator()(size_type i) { return this->elements[0_c].data_[i]; }

  /** @brief Generates a tensor index for tensor contraction
   *
   *
   *  @code auto Ai = A(_i,_j,k); @endcode
   *
   *  @param i placeholder
   *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r
   * < this->rank()
   */
  BOOST_UBLAS_INLINE
  template <std::size_t I, class... index_types>
  decltype(auto) operator()(index::index_type<I> p, index_types... ps) const {
    constexpr auto N = sizeof...(ps) + 1;
    if (N != this->rank())
      throw std::runtime_error(
          "Error in boost::numeric::ublas::operator(): size of provided "
          "index_types does not match with the rank.");

    return std::make_pair(std::cref(*this),
                          std::make_tuple(p, std::forward<index_types>(ps)...));
  }

  /** @brief Reshapes the tensor
   *
   *
   * (1) @code A.reshape(extents{m,n,o});     @endcode or
   * (2) @code A.reshape(extents{m,n,o},4);   @endcode
   *
   * If the size of this smaller than the specified extents than
   * default constructed (1) or specified (2) value is appended.
   *
   * @note rank of the tensor might also change.
   *
   * @param e extents with which the tensor is reshaped.
   * @param v value which is appended if the tensor is enlarged.
   */
  BOOST_UBLAS_INLINE
  void reshape(extents_type const &e, value_type v = value_type{}) {
    this->elements[0_c].extents_ = e;
    this->elements[0_c].strides_ = strides_type(this->elements[0_c].extents_);

    if (e.product() != this->size())
      this->elements[0_c].data_.resize(this->elements[0_c].extents_.product(),
                                       v);
  }

  friend void swap(tensor &lhs, tensor &rhs) {
    std::swap(lhs.elements[0_c].data_, rhs.elements[0_c].data_);
    std::swap(lhs.elements[0_c].extents_, rhs.elements[0_c].extents_);
    std::swap(lhs.elements[0_c].strides_, rhs.elements[0_c].strides_);
  }

  /// \brief return an iterator on the first element of the tensor
  BOOST_UBLAS_INLINE
  const_iterator begin() const { return this->elements[0_c].data_.begin(); }

  /// \brief return an iterator on the first element of the tensor
  BOOST_UBLAS_INLINE
  const_iterator cbegin() const { return this->elements[0_c].data_.cbegin(); }

  /// \brief return an iterator after the last element of the tensor
  BOOST_UBLAS_INLINE
  const_iterator end() const { return this->elements[0_c].data_.end(); }

  /// \brief return an iterator after the last element of the tensor
  BOOST_UBLAS_INLINE
  const_iterator cend() const { return this->elements[0_c].data_.cend(); }

  /// \brief Return an iterator on the first element of the tensor
  BOOST_UBLAS_INLINE
  iterator begin() { return this->elements[0_c].data_.begin(); }

  /// \brief Return an iterator at the end of the tensor
  BOOST_UBLAS_INLINE
  iterator end() { return this->elements[0_c].data_.end(); }

  /// \brief Return a const reverse iterator before the first element of the
  /// reversed tensor (i.e. end() of normal tensor)
  BOOST_UBLAS_INLINE
  const_reverse_iterator rbegin() const {
    return this->elements[0_c].data_.rbegin();
  }

  /// \brief Return a const reverse iterator before the first element of the
  /// reversed tensor (i.e. end() of normal tensor)
  BOOST_UBLAS_INLINE
  const_reverse_iterator crbegin() const {
    return this->elements[0_c].data_.crbegin();
  }

  /// \brief Return a const reverse iterator on the end of the reverse tensor
  /// (i.e. first element of the normal tensor)
  BOOST_UBLAS_INLINE
  const_reverse_iterator rend() const {
    return this->elements[0_c].data_.rend();
  }

  /// \brief Return a const reverse iterator on the end of the reverse tensor
  /// (i.e. first element of the normal tensor)
  BOOST_UBLAS_INLINE
  const_reverse_iterator crend() const {
    return this->elements[0_c].data_.crend();
  }

  /// \brief Return a const reverse iterator before the first element of the
  /// reversed tensor (i.e. end() of normal tensor)
  BOOST_UBLAS_INLINE
  reverse_iterator rbegin() { return this->elements[0_c].data_.rbegin(); }

  /// \brief Return a const reverse iterator on the end of the reverse tensor
  /// (i.e. first element of the normal tensor)
  BOOST_UBLAS_INLINE
  reverse_iterator rend() { return this->elements[0_c].data_.rend(); }

#if 0
                // -------------
                // Serialization
                // -------------

                /// Serialize a tensor into and archive as defined in Boost
                /// \param ar Archive object. Can be a flat file, an XML file or any other stream
                /// \param file_version Optional file version (not yet used)
                template<class Archive>
                void serialize ( Archive & ar, const unsigned int /* file_version */ )
                {
                    ar & serialization::make_nvp ( "data",data_ );
                }
#endif
};

} // namespace ublas
} // namespace numeric
} // namespace boost

#endif
