#pragma once

// Define me for speed.
#define FFT_USE_FMA

#include "fft2.hxx"
// #include "fft3.hxx"   under construction.

namespace mgpu {

// TODO: Add fft_radix3_t support.

template<typename real_t, int n>
struct fft_inner_t {
  typedef fft_radix2_t<typename fft_inner_t<real_t, n / 2>::inner_t> inner_t;
};
template<typename real_t>
struct fft_inner_t<real_t, 1> {
  typedef fft1_t<real_t> inner_t;
};

template<typename real_t, int n>
struct fft_t : fft_inner_t<real_t, n>::inner_t { };

template<typename real_t, int n>
struct fft_real_t : 
  real_radix2_t<typename fft_inner_t<real_t, n / 2>::inner_t> { };

template<typename real_t>
struct fft_real_t<real_t, 1> {
  array_t<complex_t<real_t>, 1> operator()(array_t<real_t, 1> x) const {
    array_t<complex_t<real_t>, 1> y;
    y[0] = complex_t<real_t>(x[0], 0);
    return y;
  }
};

} 
