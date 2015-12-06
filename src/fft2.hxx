 #pragma once

#include "fft_common.hxx"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// Radix-2 DFT on a complex-valued input.

template<typename inner_t> 
struct fft_radix2_t {

  enum { n = 2 * inner_t::n };
  typedef typename inner_t::real_t real_t;
  typedef mgpu::complex_t<real_t> complex_t;
  typedef array_t<complex_t, n> full_array_t;
  typedef array_t<complex_t, n / 2> radix_array_t;

  HOST_DEVICE full_array_t operator()(full_array_t x) const {

    radix_array_t x0 = inner(split<2, 0>(x));
    radix_array_t x1 = inner(split<2, 1>(x));

    full_array_t y;

    PRAGMA_UNROLL
    for(int k = 0; k < n / 2; ++k) {
#ifdef FFT_USE_FMA
      complex_t t = fma(-w[k], x1[k], x0[k]);
      y[k        ] = fma(x0[k], (real_t)2, -t);
      y[k + n / 2] = t;
#else
      complex_t even = x0[k];
      complex_t odd = x1[k] * w[k];
      y[k        ] = even + odd;
      y[k + n / 2] = even - odd;
#endif
    }
    
    // Compute the angles of the unit circle that have trivial twiddle factors.
    
    // 0 and 180 degrees.
    y[0    ] = x0[0] + x1[0];
    y[n / 2] = x0[0] - x1[0];

    // 90 and 270 degrees.
    if(0 == n % 4) {
      y[1 * n / 4] = x0[n / 4] + r270(x1[n / 4]);
      y[3 * n / 4] = x0[n / 4] - r270(x1[n / 4]);
    }

    return y;
  }

  fft_radix2_t() {
    // Computing twiddle factors is expensive. This should only run on
    // the host.
    for(int k = 0; k < n / 2; ++k)
      w[k] = W<real_t>(k, n);
  }

  // The twiddle factors are computed on the host and accessed through 
  // GPU constant memory.
  complex_t w[n / 2];

  // Recursively call the inner implementation. This may be of another radix
  // for composite transforms.
  inner_t inner;
};

////////////////////////////////////////////////////////////////////////////////
// Radix-2 DFT on a real-valued input.

template<typename inner_t>
struct real_radix2_t {
  enum { n = 2 * inner_t::n };
  typedef typename inner_t::real_t real_t;
  typedef mgpu::complex_t<real_t> complex_t;
  typedef array_t<complex_t, n> full_array_t;
  typedef array_t<complex_t, n / 2> half_array_t;
  typedef array_t<real_t, n> real_array_t;

  HOST_DEVICE full_array_t operator()(real_array_t x_real) const {
    // We want the FFT of the even part into x0 and the FFT of the
    // odd part into x1. Howewer, because both inputs are real-valued,
    // we can interleave them and compute them with a single FFT.
    array_t<complex_t, n / 2> x_interleaved = 
      pack_real(split<2, 0>(x_real), split<2, 1>(x_real));

    // Transform x0 and x1 simultaneously.
    array_t<complex_t, n / 2> x01 = inner(x_interleaved);

    return transformed_inner(x01);
  }

  HOST_DEVICE full_array_t transformed_inner(half_array_t x01) const {

    // Extract the real and imaginary components, but don't normalize them
    // here. We want to fold normalization into the twiddle factors.
    array_t<complex_t, n / 2> x0 = unpack_real_a(x01, false);
    array_t<complex_t, n / 2> x1 = unpack_real_b(x01, false);

    full_array_t y;

    // Note that we only iterate over n / 4 elements of x0 and x1. The
    // second half of the elements are just complex conjugates of the first
    // half.
    PRAGMA_UNROLL
    for(int k = 1; k < n / 4; ++k) {
      complex_t even = x0[k];
      complex_t odd = x1[k] * w[k];
      complex_t t0 = .5f * even + odd;
      complex_t t1 = .5f * even - odd;

      // Now exploit the Hermitian symmetry of FFTs of real numbers.
      y[k] = t0;
      y[n - k] = conj(t0);
      y[k + n / 2] = t1;
      y[n - (k + n / 2)] = conj(t1);
    }

    // Compute the angles of the unit circle with trivial twiddle factors.
    x1[0] *= .5f;
    y[0    ] = (.5f * x0[0] + x1[0]);
    y[n / 2] = (.5f * x0[0] - x1[0]);
    
    // 90 and 270 degrees.
    if(0 == n % 4) {
      x1[n / 4] *= .5f;
      y[1 * n / 4] = (.5f * x0[n / 4] + r270(x1[n / 4]));
      y[3 * n / 4] = (.5f * x0[n / 4] - r270(x1[n / 4]));
    }

    return y;    
  }

  HOST_DEVICE half_array_t half_complex(real_array_t x_real) const {
    full_array_t full = (*this)(x_real);
    half_array_t half;

    half[0] = complex_t(full[0].real, full[n / 2].real);
    PRAGMA_UNROLL
    for(int i = 1; i < n / 2; ++i)
      half[i] = full[i];

    return half;
  }

  real_radix2_t() {
    // Computing twiddle factors is expensive. This should only run on
    // the host.
    for(int k = 0; k < n / 4; ++k)
      w[k] = .5f * W<real_t>(k, n);
  }

  complex_t w[n / 4];

  // The inner transform works on complex-valued inputs.
  inner_t inner;
};

} // namespace mgpu
