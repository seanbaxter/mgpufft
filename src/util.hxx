#pragma once

#include <cuda.h>

#define DEVICE __device__ __forceinline__

#if __CUDA_ARCH__ >= 100
#define PRAGMA_UNROLL #pragma PRAGMA_UNROLL
#define HOST_DEVICE __host__ __device__ __forceinline__
#else
#define PRAGMA_UNROLL
#define HOST_DEVICE
#endif

namespace mgpu {


template<int n, bool recurse = (n > 1)>
struct s_log2 {
  enum { value = 1 + s_log2<n / 2>::value };
};
template<int n>
struct s_log2<n, false> {
  enum { value = 0 };
};

template<typename type_t>
HOST_DEVICE type_t sq(type_t x) {
  return x * x;
} 

////////////////////////////////////////////////////////////////////////////////
// complex_t

template<typename real_t>
 struct  __align__(2 * sizeof(real_t)) complex_t {
  real_t real, imag;

  complex_t() = default;
  HOST_DEVICE complex_t(const complex_t&) = default;
  HOST_DEVICE complex_t(real_t r, real_t i) : real(r), imag(i) { }
  complex_t& operator=(const complex_t&) = default;
  complex_t& operator=(complex_t&&) = default;
};

template<typename scalar_t, typename real_t>
HOST_DEVICE complex_t<real_t> 
operator*(scalar_t a, complex_t<real_t> b) {
  return complex_t<real_t>(a * b.real, a * b.imag);
}
template<typename real_t, typename scalar_t>
HOST_DEVICE complex_t<real_t> 
operator*(complex_t<real_t> a, scalar_t b) {
  return b * a;
}

template<typename real_t>
HOST_DEVICE complex_t<real_t> 
operator*(complex_t<real_t> a, complex_t<real_t> b) {
  return complex_t<real_t>(
    a.real * b.real - a.imag * b.imag, 
    a.real * b.imag + a.imag * b.real
  );
}

template<typename real_t>
HOST_DEVICE complex_t<real_t>&
operator*=(complex_t<real_t>& a, complex_t<real_t> b) {
  a = a * b;
  return a;
}
template<typename real_t, typename scalar_t>
HOST_DEVICE complex_t<real_t>&
operator*=(complex_t<real_t>& a, scalar_t b) {
  a = a * b;
  return a;
}

template<typename real_t>
HOST_DEVICE complex_t<real_t> operator-(complex_t<real_t> a) {
  return complex_t<real_t>(-a.real, -a.imag);
}
template<typename real_t>
HOST_DEVICE complex_t<real_t> 
operator+(complex_t<real_t> a, complex_t<real_t> b) {
  return complex_t<real_t>(a.real + b.real, a.imag + b.imag);
}
template<typename real_t>
HOST_DEVICE complex_t<real_t> 
operator-(complex_t<real_t> a, complex_t<real_t> b) {
  return complex_t<real_t>(a.real - b.real, a.imag - b.imag);
}

template<typename real_t>
HOST_DEVICE complex_t<real_t> r90(complex_t<real_t> a) {
  return complex_t<real_t>(-a.imag, a.real);
}
template<typename real_t>
HOST_DEVICE complex_t<real_t> r180(complex_t<real_t> a) {
  return complex_t<real_t>(-a.real, -a.imag);
}
template<typename real_t>
HOST_DEVICE complex_t<real_t> r270(complex_t<real_t> a) {
  return complex_t<real_t>(a.imag, -a.real);
}

template<typename real_t>
HOST_DEVICE complex_t<real_t> conj(complex_t<real_t> a) {
  return complex_t<real_t>(a.real, -a.imag);
}

template<typename real_t>
HOST_DEVICE real_t abs(complex_t<real_t> c) {
  return sqrt(sq(c.real) + sq(c.imag));
}

// return a * b + c.
template<typename real_t>
HOST_DEVICE complex_t<real_t>
fma(complex_t<real_t> a, complex_t<real_t> b, complex_t<real_t> c) {
  return complex_t<real_t>(
  //  (a.real * b.real + c.real) - a.imag * b.imag,
  //  (a.real * b.imag + c.real) + a.imag * b.real
    fmaf(-a.imag, b.imag, fmaf(a.real, b.real, c.real)),
    fmaf(a.imag, b.real, fmaf(a.real, b.imag, c.imag))
  );
}
template<typename real_t>
HOST_DEVICE complex_t<real_t>
fma(complex_t<real_t> a, real_t b, complex_t<real_t> c) {
  return complex_t<real_t>(
    fmaf(a.real, b, c.real), // a * b.real + c.real,
    fmaf(a.imag, b, c.imag)  //  a * b.imag + c.imag
  );
}


template<typename type_t, int size>
struct array_t {
  type_t data[size];

  HOST_DEVICE type_t operator[](int i) const { return data[i]; }
  HOST_DEVICE type_t& operator[](int i) { return data[i]; }
};

template<int s, int o, int n2, typename type_t>
HOST_DEVICE array_t<type_t, n2 / s>
split(array_t<type_t, n2> x) {
  array_t<type_t, n2 / s> y;
  PRAGMA_UNROLL
  for(int i = 0; i < n2 / s; ++i)
    y[i] = x[i * s + o];
  return y;
}

// Offset must be between 0 and size - 1.
template<typename type_t, int size>
array_t<type_t, size> rotate(array_t<type_t, size> x, int offset) {
  array_t<type_t, size> y;
  for(int i = 0; i < size; ++i)
    y[i] = x[(i + offset) % size];
  return y;
}

template<typename type_t, int size>
HOST_DEVICE array_t<type_t, size> reverse(array_t<type_t, size> x) {
  array_t<type_t, size> y;
  PRAGMA_UNROLL
  for(int i = 0; i < size; ++i)
  	y[i] = x[size - 1 - i];
  return y;
}


template<typename real_t>
DEVICE complex_t<real_t> 
__shfl(complex_t<real_t> c, int lane, int warp_size = 32) {
  return complex_t<real_t>(
    ::__shfl(c.real, lane, warp_size),
    ::__shfl(c.imag, lane, warp_size)
  );
}
template<typename real_t>
DEVICE complex_t<real_t> 
__shfl_xor(complex_t<real_t> c, int mask, int warp_size = 32) {
  return complex_t<real_t>(
    ::__shfl_xor(c.real, mask, warp_size),
    ::__shfl_xor(c.imag, mask, warp_size)
  );
}

template<typename real_t>
HOST_DEVICE void swap(complex_t<real_t>& a, complex_t<real_t>& b) {
  complex_t<real_t> c = a; a = b; b = c;
}


} // namespace mgpu
