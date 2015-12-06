#if 0

template<typename inner_t>
struct dit_radix3_t {
  enum { n = 3 * inner_t::n };
  typedef typename inner_t::real_t real_t;
  typedef ::complex_t<real_t> complex_t;
  typedef array_t<complex_t, n> full_array_t;
  typedef array_t<complex_t, n / 3> radix_array_t;

  HOST_DEVICE full_array_t operator()(full_array_t x) const {
    radix_array_t x0 = inner(split<3, 0>(x));
    radix_array_t x1 = inner(split<3, 1>(x));
    radix_array_t x2 = inner(split<3, 2>(x));

    full_array_t y;
    PRAGMA_UNROLL
    for(int k = 0; k < n / 3; ++k) {
#ifdef FFT_USE_FMA
      complex_t w = W<real_t>(k, n);
      complex_t w2 = w * w;
      complex_t c1 = complex_t(-.5, 0);
      complex_t c2 = complex_t(0, -0.86602540378);

      complex_t z1 = w * x1[k];
      complex_t s1 = z1 - w2 * x2[k];
      complex_t s2 = 2 * z1 - s1;
      complex_t s3 = s2 + x0[k];
      complex_t s4 = x0[k] + c1 * s2;
      complex_t s5 = s4 - c2 * s1;
      complex_t s6 = 2 * s4 - s5;

      y[k + 0 * n / 3] = s3;
      y[k + 1 * n / 3] = s6;
      y[k + 2 * n / 3] = s5;
#else
      complex_t w = W<real_t>(k, n);
      complex_t w2 = w * w;
      y[k + 0 * n / 3] = 

      y[k + 0 * n / 3] = y0[k] + y1[k] * W(1 * k0, n) + y2[k] * W(2 * k0, n);
      y[k + 1 * n / 3] = y0[k] + y1[k] * W(1 * k1, n) + y2[k] * W(2 * k1, n);
      y[k + 2 * n / 3] = y0[k] + y1[k] * W(1 * k2, n) + y2[k] * W(2 * k2, n);
#endif      
    }

    return y;
  }

  inner_t inner;
};


#endif
