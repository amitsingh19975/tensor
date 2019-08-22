//#define BOOST_UBLAS_NO_RECURSIVE_OPTIMIZATION

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/yap/print.hpp>

/**
 *
 * This file is not run as test it is just for my local development quick
 * testings
 *
 */

template <class T>
void preety(T e) {
  std::cout << __PRETTY_FUNCTION__;
}

int main() {
  using namespace boost::numeric::ublas;

  using tensor_type = tensor<int>;
  using value_type = int;
  std::vector<int> sa(50 * 50 * 30);
  std::vector<int> sb(2500);

  std::iota(sa.begin(), sa.end(), 1);
  std::iota(sb.begin(), sb.end(), 1);

  tensor_type a{shape{50, 50, 30}, sa};
  tensor_type b{shape{50, 50}, 5};
  tensor_type c{shape{50, 50}, 1};

  auto expr = a + a;

  auto e = boost::yap::transform(
      expr, detail::transforms::runtime_scalar_optimizer{});
  // This e expression has varaint inside of it

  tensor_type ans{shape{50, 50, 30}};
  for (size_t t = 0; t < 50 * 50 * 30; t++) {
    auto at_t = boost::yap::transform(e, detail::transforms::at_index{t});
    ans(t) = boost::yap::transform(at_t,
                                   detail::transforms::evaluate_with_variant{});
  }
  // We evaluate variant expression with out own evalute called
  // `evaluate_with_variant`

  tensor_type ans2 = expr;

  assert((bool)(ans == ans2));

  // Passed

  return 0;
}
