
#include <type_traits>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>
#include <cuda.h>
#include <cufft.h>

#include "gpu.hxx"
#include "fft_kernels.hxx"


using namespace mgpu;

template<typename real_t>
std::vector<complex_t<real_t> > 
fft_simple(const std::vector<complex_t<real_t> >& x, 
  int offset = -1, int stride = -1) {

  if(-1 == offset) {
    offset = 0;
    stride = 1;
  }
  int n = (int)x.size();
  std::vector<complex_t<real_t> > y(n);
  if(2 == n) {
    y[0] = x[0] + x[1];
    y[1] = x[0] - x[1];
  } else {
    std::vector<complex_t<real_t> > x0(n / 2), x1(n / 2);
    for(int k = 0; k < n / 2; ++k)
      x0[k] = x[2 * k + 0], x1[k] = x[2 * k + 1];

    std::vector<complex_t<real_t> > y0 = 
      fft_simple(x0, offset, 2 * stride);
    std::vector<complex_t<real_t> > y1 = 
      fft_simple(x1, offset + stride, 2 * stride);

    for(int k = 0; k < n / 2; ++k) {
      y[k + 0    ] = y0[k] + y1[k] * W<real_t>(k, n);
      y[k + n / 2] = y0[k] - y1[k] * W<real_t>(k, n);
    }
  }

  return y;
}

template<typename real_t>
std::vector<complex_t<real_t> > 
fft_real(const std::vector<real_t>& x) {
  std::vector<complex_t<real_t> > x_complex(x.size());
  for(size_t i = 0; i < x.size(); ++i)
    x_complex[i] = complex_t<real_t>(x[i], 0);
  return fft_simple(x_complex);
}


struct benchmark_t {
  int n;
  int size;
  int batch;
  int num_iterations;
  double elapsed[2];    // [0] is MGPU. [1] is CUFFT.
  double throughput[2]; // In billion points/sec.
  double bandwidth[2];  // In GB/sec.
};
benchmark_t benchmark_test(int n, int batch, int num_iterations = 10, 
  bool print_transform = false) {

  typedef float real_t;
  benchmark_t benchmark;
  benchmark.n = n;
  benchmark.batch = batch;
  benchmark.size = n * batch;
  benchmark.num_iterations = num_iterations;

  std::vector<real_t> x_real(n * batch);

  for(int sys = 0; sys < batch; ++sys)
    for(int i = 0; i < n; ++i)
      x_real[sys * n + i] = sin(i + sys + 6);

  // Allocate space for both 
  real_t* input_global = host_to_device(x_real);
  complex_t<real_t>* output_mgpu, *output_cufft;

  // MGPU outputs n / 2 complex elements, by packing y[n/2].real into y[0].imag.
  // CUFFT outputs n / 2 + 1 elements. We want to allow CUFFT to write complete
  // cache lines, so round n / 2 + 1 up to the next cache-line size.
  int odist = (n >= 32) ? (~15 & (n / 2 + 1 + 15)) : n / 2 + 1;

  cudaMalloc((void**)&output_mgpu, sizeof(complex_t<real_t>) * (n / 2) * batch);
  cudaMemset(output_mgpu, 0, sizeof(complex_t<real_t>) * (n / 2) * batch);

  cudaMalloc((void**)&output_cufft, sizeof(complex_t<real_t>) * odist * batch);
  cudaMemset(output_cufft, 0, sizeof(complex_t<real_t>) * odist * batch);

  mgpu::timer_t timer;

  // MGPU
  {
    timer.start();
    for(int it = 0; it < num_iterations; ++it)
      fft_kernel(n, input_global, output_mgpu, batch);
    benchmark.elapsed[0] = timer.stop();
  }

  // CUFFT
  {  
    cufftHandle plan;
    int dim[] = { n, 0, 0 };
    int inembed[] = { 0 };
    int onembed[] = { 0 };
    cufftResult result = cufftPlanMany(&plan, 1, dim, inembed, 1, n, 
      onembed, 1, odist, CUFFT_R2C, batch);

    timer.start();
    for(int it = 0; it < num_iterations; ++it)
      result = cufftExecR2C(plan, input_global, (cufftComplex*)output_cufft);
    benchmark.elapsed[1] = timer.stop();

    cufftDestroy(plan);
  }

  // Compute throughputs and bandwidths from elapsed time.
  for(int i = 0; i < 2; ++i) {
    benchmark.throughput[i] = (double)benchmark.size * num_iterations / 
      benchmark.elapsed[i] / 1.0e9;
    benchmark.bandwidth[i] = (double)benchmark.size * num_iterations * 
      2 * sizeof(real_t) / benchmark.elapsed[i] / 1.0e9;
  }

  std::vector<complex_t<real_t> > output_host = 
    device_to_host(output_mgpu, (n / 2) * batch);

  // Print the transform and compare against the CPU-generated reference.
  // Nice for debugging.
  if(print_transform) {
    for(int sys = 0; sys < batch; ++sys) {
      std::vector<real_t> x_real2(
        x_real.data() + sys * n,
        x_real.data() + (sys + 1) * n);
      std::vector<complex_t<real_t> > y = fft_real(x_real2);

      for(int i = 0; i < n / 2; ++i) {
        int index = sys * n / 2 + i;
        complex_t<real_t> ref = i ? y[i] :
          complex_t<real_t>(y[0].real, y[n / 2].real);
        complex_t<real_t> test = output_host[index];
        bool error = abs(ref - test) / abs(ref) > 1.0e-4;

        printf("%5d | %5d: % 10.5f + i% 10.5f      % 10.5f + i% 10.5f %c\n", 
          sys, i, ref.real, ref.imag, test.real, test.imag, 
          error ? '*' : ' ');
      }
    }
  }

  cudaFree(input_global);
  cudaFree(output_mgpu);
  cudaFree(output_cufft);

  return benchmark;
}


int main(int argc, char** argv) {
  size_t freeMem, totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  printf("%s\n", prop.name);
  printf("%d MB device memory.\n", (int)(totalMem / (1<< 20)));

  double bandwidth = 1000.0 * prop.memoryClockRate * prop.memoryBusWidth / 
    8 * 2 / 1.0e9;
  printf("Memory bandwidth: %d GB/s\n", (int)bandwidth);

  printf("\nThroughputs reported by\n"
         "1. billions of points/s.\n"
         "2. GB/s of memory utilization (8 bytes/point).\n\n");
  printf("\n   n:            MGPU                        CUFFT           ratio\n");

  for(int n = 4; n <= 1024; n *= 2) {
    benchmark_t bench = benchmark_test(n, (64<< 20) / n, 10);
    printf("%4d: %6.3f B/s (%6.2f GB/s)    %6.3f B/s (%6.2f GB/s)  %5.2fx\n", 
      bench.n,
      bench.throughput[0], bench.bandwidth[0],
      bench.throughput[1], bench.bandwidth[1],
      bench.throughput[0] / bench.throughput[1]);
  }

  return 0;
}


