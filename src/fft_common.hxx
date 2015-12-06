#pragma once

#include "util.hxx"

namespace mgpu {

// Compute the twiddle factor exp(-2 pi k / n).  
template<typename real_t>
HOST_DEVICE complex_t<real_t> W(int k, int n) {
  real_t arg = (-2 * (real_t)M_PI / n) * k;
  return complex_t<real_t>(cos(arg), sin(arg));
}


// Real<->complex conversions.

// Pack two real arrays into complex array.
template<typename real_t, int size>
HOST_DEVICE array_t<complex_t<real_t>, size> 
pack_real(array_t<real_t, size> a, array_t<real_t, size> b) {
  array_t<complex_t<real_t>, size> c;
  PRAGMA_UNROLL
  for(int i = 0; i < size; ++i)
    c[i] = complex_t<real_t>(a[i], b[i]);
  return c;
}

// Unpack complex number.
template<typename real_t>
HOST_DEVICE complex_t<real_t> 
unpack_real_a(complex_t<real_t> a, complex_t<real_t> b, bool normalize = true) {
  complex_t<real_t> c = (a + conj(b));
  if(normalize) c *= .5f;
  return c;
}
template<typename real_t>
HOST_DEVICE complex_t<real_t> 
unpack_real_b(complex_t<real_t> a, complex_t<real_t> b, bool normalize = true) {
  complex_t<real_t> c = r270(a - conj(b));
  if(normalize) c *= .5f;
  return c;
}

// Unpack complex array.
template<typename real_t, int size>
HOST_DEVICE array_t<complex_t<real_t>, size> 
unpack_real_a(array_t<complex_t<real_t>, size> c, bool normalize = true) {
  array_t<complex_t<real_t>, size> a;
  PRAGMA_UNROLL
  for(int i = 0; i < size; ++i)
    a[i] = unpack_real_a(c[i], c[(size - i) % size], normalize);
  return a;
}

template<typename real_t, int size>
HOST_DEVICE array_t<complex_t<real_t>, size> 
unpack_real_b(array_t<complex_t<real_t>, size> c, bool normalize = true) {
  array_t<complex_t<real_t>, size> b;
  PRAGMA_UNROLL
  for(int i = 0; i < size; ++i)
    b[i] = unpack_real_b(c[i], c[(size - i) % size], normalize);
  return b;
}


// Apply the circular shift theorem. If applied on input, it moves the output
// m elements. If applied on output, it moves the input m elements.
template<typename real_t, int n>
HOST_DEVICE array_t<complex_t<real_t>, n>
circular_shift(array_t<real_t, n> x, int m) {
  array_t<complex_t<real_t>, n> y;

  if(m != 0) {
    complex_t<real_t> rot = W<real_t>(m, n);
    complex_t<real_t> next(1, 0);
    PRAGMA_UNROLL
    for(int i = 0; i < n / 4; ++i) {
      y[i + 0 * n / 4] *= next;
      y[i + 1 * n / 4] *= r270(next);
      y[i + 2 * n / 4] *= r180(next);
      y[i + 3 * n / 4] *= r90(next);
      next *= rot;
    }
  } else {
    PRAGMA_UNROLL
    for(int i = 0; i < n; ++i)
      y[i] = complex_t<real_t>(x[i], 0);
  }
  return y;
}

template<typename real_t, int n>
HOST_DEVICE array_t<complex_t<real_t>, n>
circular_shift(array_t<complex_t<real_t>, n> x, int m) {
  array_t<complex_t<real_t>, n> y = x;

  if(m != 0) {
    complex_t<real_t> rot = W<real_t>(m, n);
    complex_t<real_t> next(1, 0);
    PRAGMA_UNROLL
    for(int i = 0; i < n / 4; ++i) {
      y[i + 0 * n / 4] *= next;
      y[i + 1 * n / 4] *= r270(next);
      y[i + 2 * n / 4] *= r180(next);
      y[i + 3 * n / 4] *= r90(next);
      next *= rot;
    }
  }
  return y;
}

template<typename real_t, int n>
HOST_DEVICE array_t<complex_t<real_t>, n> shift_array(int k, int m) {
  array_t<complex_t<real_t>, n> shift;
  complex_t<real_t> w = W<real_t>(k, m);
  complex_t<real_t> w2 = w * w;
  complex_t<real_t> w3 = W<real_t>(3 * k, m);

  // Regenarate exact shifts every 4th element. This is to mitigate
  // round-off error that would be caused by accumulating from w1.
  shift[0] = complex_t<real_t>(1, 0);
  if(n > 1) shift[1] = w;
  if(n > 2) shift[2] = w2;
  if(n > 3) shift[3] = w3;
  PRAGMA_UNROLL
  for(int i = 4; i < n; ++i) {
    switch(i % 4) {
      case 0: shift[i] = W<real_t>(k * i, m);    break;
      case 1: shift[i] = shift[i - 1] * w;       break;
      case 2: shift[i] = shift[i - 2] * w2;      break;
      case 3: shift[i] = shift[i - 3] * w3;      break;
    }
  }
  return shift;
}


// Perform an n-point FFT over the warp.
template<int n, typename real_t>
DEVICE complex_t<real_t> warp_fft(complex_t<real_t> x, int lane) {
  typedef complex_t<real_t> complex_t;
  const int log_n = s_log2<n>::value;
  complex_t c0 = __shfl(x, __brev(lane)>> (32 - log_n), n);

  PRAGMA_UNROLL
  for(int level = 0; level < log_n; ++level) {
    complex_t c1 = __shfl_xor(c0, 1<< level);
    if(0 == level) {
      c0 = (1 & lane) ? c1 - c0 : c0 + c1;
    } else if(1 == level) {
      complex_t w = (1 & lane) ? complex_t(0, -1) : complex_t(1, 0);
      if(2 & lane) {
        swap(c0, c1);
        w *= -1;
      }
      c0 = fma(w, c1, c0);
    } else {
      if((1<< level) & lane)
        swap(c0, c1);
      
      int arg = ((2<< level) - 1) & lane;
      complex_t w = W<real_t>(arg, 2<< level);
      c0 = fma(w, c1, c0); 
    }
  }
  return c0;
}


// The concrete type that sits at the center of the FFT composition.

template<typename real_t_>
struct fft1_t {
  enum { n = 1 };
  typedef real_t_ real_t;
  typedef array_t<complex_t<real_t>, n> full_array_t;

  HOST_DEVICE full_array_t operator()(full_array_t x) const {
    return x;
  }
};


} // namespace mgpu
