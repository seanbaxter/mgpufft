#pragma once

#include "fft.hxx"
#include "transform.hxx"

namespace mgpu {

// Trivial kernel that calls into func_t mfa. All the FFT kernels use this
// entry point.

template<typename func_t>
__global__ void mfa_kernel(func_t mfa) {
  mfa.block(threadIdx.x, blockIdx.x);
}

////////////////////////////////////////////////////////////////////////////////
// FFT kernel for n = 4 to n = 16.

template<typename real_t, int n>
struct mfa_small_k {
  typedef complex_t<real_t> complex_t;

  enum { 
    num_rows = 16,
    sys_per_thread = num_rows / n,
    num_systems = 32 * sys_per_thread
  };

  fft_real_t<real_t, n> fft;
  const real_t* input_global;
  complex_t* output_global;
  int count;

  // Use LDG to load entire systems into each thread. These are not coalesced
  // loads, but we should basically fit within the L1 cache.
  DEVICE void load_column(int lane, int warp_sys, array_t<float, n>& x) const {
    int sys_count = count - warp_sys;
    const float4* input = (const float4*)(input_global + (warp_sys + lane) * n);
    if(lane < sys_count) {
      PRAGMA_UNROLL
      for(int i = 0; i < n / 4; ++i) {
#if __CUDA_ARCH__ >= 350        
        float4 packed = __ldg(input + i);
#else
        float4 packed = *(input + i);
#endif        
        x[4 * i + 0] = packed.x;
        x[4 * i + 1] = packed.y;
        x[4 * i + 2] = packed.z;
        x[4 * i + 3] = packed.w;
      }
    }
  }  

  DEVICE void load_column(int lane, int warp_sys, array_t<double, n>& x) const {
    int sys_count = count - warp_sys;
    const double2* input = (const double2*)
      (input_global + (warp_sys + lane) * n);
    if(lane < sys_count) {
      PRAGMA_UNROLL
      for(int i = 0; i < n / 2; ++i) {
#if __CUDA_ARCH__ >= 350        
        double2 packed = __ldg(input + i);
#else
        double2 packed = *(input + i);
#endif        
        x[2 * i + 0] = packed.x;
        x[2 * i + 1] = packed.y;
      }
    }
  }

  DEVICE void load_columns(int lane, int warp_sys, 
    array_t<real_t, n> (&x)[sys_per_thread]) const {

    PRAGMA_UNROLL
    for(int sys = 0; sys < sys_per_thread; ++sys)
      load_column(lane, warp_sys + 32 * sys, x[sys]);
  }  


  DEVICE void transpose(int lane, 
    const array_t<complex_t, n / 2> (&y_col)[sys_per_thread],
    complex_t (&y_output)[num_rows / 2], float* warp_shared) const {

    // Store the transformed data in memory order with a stride.
    PRAGMA_UNROLL
    for(int sys = 0; sys < sys_per_thread; ++sys) {
      int start = (32 * sys + lane) * (n / 2);
      start += start / 32; // offset to avoid bank conflicts.

      PRAGMA_UNROLL
      for(int row = 0; row < n / 2; ++row)
        warp_shared[start + row] = y_col[sys][row].real;
      __syncthreads();
    }

    // Cooperatively load the transformed data.
    PRAGMA_UNROLL
    for(int i = 0; i < num_rows / 2; ++i)
      y_output[i].real = warp_shared[33 * i + lane];
    __syncthreads();

    // Do the same for the imaginary components.
    PRAGMA_UNROLL
    for(int sys = 0; sys < sys_per_thread; ++sys) {
      int start = (32 * sys + lane) * (n / 2);
      start += start / 32; // offset to avoid bank conflicts.

      PRAGMA_UNROLL
      for(int row = 0; row < n / 2; ++row)
        warp_shared[start + row] = y_col[sys][row].imag;
      __syncthreads();
    }

    PRAGMA_UNROLL
    for(int i = 0; i < num_rows / 2; ++i)
      y_output[i].imag = warp_shared[33 * i + lane];
    __syncthreads();
  }

  DEVICE void block(int tid, int block) {

    enum {
    #if __CUDA_ARCH__ >= 520
      nt = 128,
    #else
      nt = 64,
    #endif
      num_warps = nt / 32 
    };

    struct shared_t {
      real_t transpose[33 * (num_rows / 2)];
    };
    __shared__ shared_t shared[num_warps];

    int lane = num_warps > 1 ? tid % 32 : tid;
    int warp = num_warps > 1 ? tid / 32 : 0;
    int warp_sys = (num_warps * block + warp) * num_systems;
    
    // Load the systems as columns for each thread.
    array_t<real_t, n> x_col[sys_per_thread];
    load_columns(lane, warp_sys, x_col);

    // DFT the systems.
    array_t<complex_t, n / 2> y_col[sys_per_thread];
    PRAGMA_UNROLL
    for(int sys = 0; sys < sys_per_thread; ++sys) {
      y_col[sys] = fft.half_complex(x_col[sys]);
    }

    // Transpose for storage.
    complex_t y_output[num_rows / 2];
    transpose(lane, y_col, y_output, shared[warp].transpose);

    // Cooperatively store to global memory.
    complex_t* output = output_global + (n / 2) * warp_sys;
    int sys_count = count - warp_sys;

    PRAGMA_UNROLL
    for(int i = 0; i < num_rows / 2; ++i) {
      int index = 32 * i + lane;
      int sys = index / (n / 2);
      if(sys < sys_count)
        output[index] = y_output[i];
    }
  }
};


////////////////////////////////////////////////////////////////////////////////
// FFT kernel for n = 32 to n = 512.

template<typename real_t, int n>
struct mfa_large_k {
  enum { 
    // Parameters for the column-wise pass.
    num_rows = n <= 32 ? 16 : 32, 
    rows_per_system = n / 32,
    num_cols = n / num_rows, 
    center_row = num_rows / 2,

    // Parameters for the transposition.
    num_slots = num_rows / 2,
    slot_mask = num_slots - 1,
    slot_stride = num_slots + 1,

    // Parameters for the row-wise pass.
    num_groups = 32 / (num_rows / 2),
    group_size = 32 / num_groups,

    num_systems = 32 * num_rows / n,
    rows_per_thread = (num_systems * num_rows / 2) / 32,
  };

  typedef mgpu::complex_t<real_t> complex_t;

  struct params_t {
    int col1;       // Column for stage one.
    
    int col2;       // Starting column for stage two.
    int row2;       // Row for stage two.
    int sys2;       // Starting system for stage two.

    int warp_sys;   // The first system for this warp.
  };

  fft_real_t<real_t, num_rows> fft_col;
  fft_t<real_t, num_cols> fft_row;

  const real_t* input_global;
  complex_t* output_global;
  int count;


  DEVICE array_t<real_t, num_rows> load_column(int lane, params_t params,
    real_t* warp_shared) const {

    int sys_count = count - params.warp_sys;

    array_t<real_t, num_rows> x_col;

    int sys = lane / num_cols;
    int col = lane % num_cols;

    ////////////////////////////////////////////////////////////////////////////
    // Cooperatively load all the data for the warp into register.

    const real_t* input = input_global + params.warp_sys * n + lane;
    real_t x[num_systems][rows_per_system];

    PRAGMA_UNROLL
    for(int sys = 0; sys < num_systems; ++sys) {
      // Load either the first or second half of each system from device
      // memory.
      if(sys < sys_count) {
        PRAGMA_UNROLL
        for(int row = 0; row < rows_per_system; ++row) {
#if __CUDA_ARCH__ >= 350            
          x[sys][row] = __ldg(input + sys * n + 32 * row);
#else
          x[sys][row] = input[sys * n + 32 * row];
#endif
        }
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Transpose the data so that each thread has a full column of it in
    // row-major order. That is, tid 0 may have elements 0, 4, 8, etc, where
    // num_cols = 4.

    int start = sys * (n / 2 + num_cols) + col;
    if(rows_per_system >= 2) {
      // Store the first half of each system. Because n >= 64, we can transpose
      // full cache lines at a time, so each thread gets a store on each
      // iteration.

      PRAGMA_UNROLL
      for(int half = 0; half < 2; ++half) {
        // Store the half system in shared memory.
        PRAGMA_UNROLL
        for(int sys = 0; sys < num_systems; ++sys) {
          PRAGMA_UNROLL
          for(int row = 0; row < rows_per_system / 2; ++row)
            warp_shared[sys * (n / 2 + num_cols) + 32 * row + lane] = 
              x[sys][half * rows_per_system / 2 + row];
        }
        __syncthreads();

        // Load the half system into register. Spacing each system by num_cols
        // provides bank-conflict avoidance for all configurations.
        PRAGMA_UNROLL
        for(int row = 0; row < num_rows / 2; ++row)
          x_col[row + half * num_rows / 2] = 
            warp_shared[start + num_cols * row];
        __syncthreads();
      }
    } else {
      // Each system is only a single cache line. To conserve shared memory,
      // transpose the first half, then the second half. We select on the
      // lane for this so there is divergence in the stores, but not the
      // loads.
      PRAGMA_UNROLL
      for(int half = 0; half < 2; ++half) {
        if(!half && lane < 16 || half && lane >= 16) {
          int col = half ? lane - 16 : lane;
          PRAGMA_UNROLL
          for(int sys = 0; sys < num_systems; ++sys)
            warp_shared[sys * (n / 2 + num_cols) + col] = x[sys][0];
        }
        __syncthreads();
        PRAGMA_UNROLL
        for(int row = 0; row < num_rows / 2; ++row)
          x_col[row + half * num_rows / 2] = 
            warp_shared[start + num_cols * row];
        __syncthreads();
      }
    }

    return x_col;
  }

  DEVICE void transpose(int lane, params_t params, real_t* warp_shared, 
    array_t<complex_t, num_rows> y_col, 
    array_t<complex_t, num_cols> (&x_row)[rows_per_thread]) const {

    int start = (params.col2 + 33 * params.row2);

    PRAGMA_UNROLL
    for(int row = 0; row < num_rows / 2; ++row)
      warp_shared[lane + 33 * row] = y_col[row].real;
    __syncthreads();

    PRAGMA_UNROLL
    for(int i = 0; i < num_cols * rows_per_thread; ++i)
      x_row[i / num_cols][i % num_cols].real = warp_shared[start + i];
    __syncthreads();

    PRAGMA_UNROLL
    for(int row = 0; row < num_rows / 2; ++row)
      warp_shared[lane + 33 * row] = y_col[row].imag;
    __syncthreads();

    PRAGMA_UNROLL
    for(int i = 0; i < num_cols * rows_per_thread; ++i)
      x_row[i / num_cols][i % num_cols].imag = warp_shared[start + i];
    __syncthreads();
  }

  DEVICE void block(int tid, int block) {

    enum {
    #if __CUDA_ARCH__ >= 520
      nt = 32 == n ? 256 : 128,
    #else
      nt = 32 == n ? 128 : 64,
    #endif
      num_warps = nt / 32 
    };

    int lane = num_warps > 1 ? tid % 32 : tid;
    int warp = num_warps > 1 ? tid / 32 : 0;
    int col = lane % num_cols;

    params_t params;
    params.col1 = lane % num_cols;
    params.col2 = num_cols * rows_per_thread * (lane / (num_rows / 2));
    params.row2 = lane % (num_rows / 2);
    params.sys2 = params.col2 / num_cols;
    params.warp_sys = (num_warps * block + warp) * num_systems;

    union shared_t {
      real_t load_columns[num_systems * (32 * rows_per_system / 2 + num_cols)];
      real_t transpose[33 * (num_rows / 2)];
    };

    __shared__ shared_t shared[num_warps];

    ////////////////////////////////////////////////////////////////////////////
    // Stage 1: 
    // DFT on columns.

    // Load 32-point columns per thread.
    array_t<real_t, num_rows> x_col = load_column(lane, params, 
      shared[warp].load_columns);

    // Perform real-valued DFT on the columns.
    array_t<complex_t, num_rows> y_col = fft_col(x_col);

    // Twiddle and transform the center row for each column.
    complex_t center_w = W<real_t>(center_row * col, n);
    complex_t x_center = center_w * complex_t(y_col[center_row].real, 0);
    complex_t y_center = warp_fft<num_cols>(x_center, col);

    // Compute the twiddle for the Matrix Fourier Algorithm.
    array_t<complex_t, num_slots> shift = 
      shift_array<real_t, num_slots>(col, n);

    PRAGMA_UNROLL
    for(int row = 0; row < num_rows / 2; ++row)
      y_col[row] *= shift[row];

    ////////////////////////////////////////////////////////////////////////////
    // Stage 2:
    // Transpose the columns in each thread to rows. In stage 1 there was 
    // one column assigned to each thread. In stage 3, there are one or more
    // rows assigned to each thread. Because we exploit Hermitian symmetry,
    // stage 3 deals with a problem that is 1/2 the size of stage 1's problem.

    // Each thread loads different columns from the same row.
    array_t<complex_t, num_cols> x_row[rows_per_thread];
    transpose(lane, params, shared[warp].transpose, y_col, x_row);

    ////////////////////////////////////////////////////////////////////////////
    // Stage 3:
    // Compute the FFT on complex number rows to finish the FFT.
    // Store half-complex arrays to global memory.

    array_t<complex_t, num_cols> y_row[rows_per_thread];
    PRAGMA_UNROLL
    for(int sys = 0; sys < rows_per_thread; ++sys)
      y_row[sys] = fft_row(x_row[sys]);

    // Store each of the systems back to global memory.
    complex_t* output = output_global + 
      (params.warp_sys + params.sys2) * (n / 2);
    int sys_count = count - params.warp_sys - params.sys2;

    PRAGMA_UNROLL
    for(int sys = 0; sys < rows_per_thread; ++sys) {
      // Store the transposed matrix out in column-major order.
      // We have each row in register, so this is a simple coalesced store.

      // Pack the real-valued center output into the imaginary part of the
      // leading output.
      if(!params.row2)
        y_row[sys][0].imag = y_row[sys][num_cols / 2].real;

      PRAGMA_UNROLL
      for(int col = 0; col < num_cols; ++col) {
        int index = 2 * (num_rows / 2) * col + params.row2;

        if(col >= num_cols / 2) {
          // Use shuffle to load the DFT of the center real value.
          // This is only needed for threads on row 0 (tid 0 and 16).
          complex_t center = __shfl(y_center, 
            params.col2 + sys * num_cols + col);
          if(!params.row2) y_row[sys][col] = center;
          y_row[sys][col] = conj(y_row[sys][col]);

          int offset = params.row2 ? n : n - num_rows / 2;
          index = offset - index;
        }

        // Store to global memory if the system is in-range.
        if(sys < sys_count) 
            output[index] = y_row[sys][col];
      }

      output += n / 2;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// 1024-point FFT. Process 32 elements per thread and 32 per column.

template<typename real_t>
struct mfa_1024_k {

  typedef mgpu::complex_t<real_t> complex_t;

  fft_real_t<real_t, 32> fft_col;
  fft_t<real_t, 32> fft_row;

  const real_t* input_global;
  complex_t* output_global;
  int count;

  DEVICE array_t<real_t, 32> load_column(int lane, int sys) const {
    const real_t* input = input_global + 1024 * sys + lane;
    array_t<real_t, 32> x_col;

    PRAGMA_UNROLL
    for(int i = 0; i < 32; ++i)
      x_col[i] = __ldg(input + 32 * i);

    return x_col;
  }

  template<int nt>
  DEVICE array_t<complex_t, 32> transpose(int tid, real_t* shared,
    array_t<complex_t, 32> y_col) const {

    int row = 15 & tid;
    int sys = tid / 16;
    int start = row * (nt + 1) + 32 * sys;

    PRAGMA_UNROLL
    for(int row = 0; row < 16; ++row)
      shared[tid + row * (nt + 1)] = y_col[row].real;
    __syncthreads();

    array_t<complex_t, 32> x_row;
    if(tid < nt / 2) {
      PRAGMA_UNROLL
      for(int col = 0; col < 32; ++col)
        x_row[col].real = shared[start + col];
    }
    __syncthreads();

    PRAGMA_UNROLL
    for(int row = 0; row < 16; ++row)
      shared[tid + row * (nt + 1)] = y_col[row].imag;
    __syncthreads();

    if(tid < nt / 2) {
      PRAGMA_UNROLL
      for(int col = 0; col < 32; ++col)
        x_row[col].imag = shared[start + col];
    }
    __syncthreads();

    return x_row;
  }  

  DEVICE void block(int tid, int block) {

    enum {
    #if __CUDA_ARCH__ >= 520
      nt = 128,
    #else
      nt = 64,
    #endif
      num_warps = nt / 32 
    };

    int lane = num_warps > 1 ? tid % 32 : tid;
    int warp = num_warps > 1 ? tid / 32 : 0;
    int sys = num_warps * block + warp;

    struct shared_t {
      real_t transpose[16 * (nt + 1)];
      complex_t center_row[nt];
    };

    __shared__ shared_t shared;

    // DFT on columns.
    // Load 32-point columns per thread.
    array_t<real_t, 32> x_col = load_column(lane, sys);

    // Perform real-valued DFT on the columns.
    array_t<complex_t, 32> y_col = fft_col(x_col);

    // Twiddle and transform the center row for each column.
    complex_t center_w = W<real_t>(16 * lane, 1024);
    complex_t x_center = center_w * complex_t(y_col[16].real, 0);
    complex_t y_center = warp_fft<32>(x_center, lane);

    // Compute the twiddle for the Matrix Fourier Algorithm.
    array_t<complex_t, 16> shift = shift_array<real_t, 16>(lane, 1024);

    // Store the transformed center row.
    shared.center_row[tid] = y_center;

    PRAGMA_UNROLL
    for(int row = 0; row < 16; ++row)
      y_col[row] *= shift[row];

    // Transpose.
    array_t<complex_t, 32> x_row = transpose<nt>(tid, shared.transpose, y_col);

    // Compute the FFT on complex number rows to finish the FFT.
    array_t<complex_t, 32> y_row = fft_row(x_row);

    // Store each of the systems back to global memory.
    int sys0 = num_warps * block;
    sys = 2 * warp + (lane >= 16);

    // Only store half-complex outputs so space by 1024/2 complex_ts.
    complex_t* output = output_global + (1024 / 2) * (sys0 + sys);

    if(sys0 + sys < count && tid < nt / 2) {
      int row = 15 & lane;

      // Pack the real-valued center output into the imaginary part of the
      // leading output.
      if(!row) y_row[0].imag = y_row[16].real;

      PRAGMA_UNROLL
      for(int col = 0; col < 32; ++col) {
        int index = 32 * col + row;
        if(col >= 16) {
          complex_t center = shared.center_row[32 * sys + col];
          if(!row) y_row[col] = center;
          y_row[col] = conj(y_row[col]);
          int offset = row ? 1024 : 1024 - 16;
          index = offset - index;
        }
        output[index] = y_row[col];
      }
    }
  }
};

template<int n, typename real_t>
void small_fft_kernel(const real_t* input_global, 
  complex_t<real_t>* output_global, int count) {

  // Use matrix fourier algorithm.
  typedef mfa_small_k<real_t, n> mfa_t;
  mfa_t mfa;
  mfa.input_global = input_global;
  mfa.output_global = output_global;
  mfa.count = count;

  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, &mfa_kernel<mfa_t>);
  int nt = attr.binaryVersion >= 52 ? 128 : 64;
  int num_warps = nt / 32;
  int systems_per_cta = num_warps * mfa_t::num_systems;
  int num_cta = (count + systems_per_cta - 1) / systems_per_cta;

  mfa_kernel<<<num_cta, nt>>>(mfa);
}

template<int n, typename real_t>
void large_fft_kernel(const real_t* input_global, 
  complex_t<real_t>* output_global, int count) {

  // Use matrix fourier algorithm.
  typedef mfa_large_k<real_t, n> mfa_t;
  mfa_t mfa;
  mfa.input_global = input_global;
  mfa.output_global = output_global;
  mfa.count = count;

  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, &mfa_kernel<mfa_t>);
  int nt = attr.binaryVersion >= 52 ? 
    (32 == n ? 256 : 128) : 
    (32 == n ? 128 : 64);
  int num_warps = nt / 32;
  int systems_per_cta = num_warps * mfa_t::num_systems;
  int num_cta = (count + systems_per_cta - 1) / systems_per_cta;

  mfa_kernel<<<num_cta, nt>>>(mfa);
}

template<typename real_t>
void fft_kernel_1024(const real_t* input_global,
  complex_t<real_t>* output_global, int count) {

  typedef mfa_1024_k<real_t> mfa_t;
  mfa_t mfa;
  mfa.input_global = input_global;
  mfa.output_global = output_global;
  mfa.count = count;

  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, &mfa_kernel<mfa_t>);
  int nt = attr.binaryVersion >= 52 ? 128 : 64;
  int num_warps = nt / 32;
  int systems_per_cta = num_warps;
  int num_cta = (count + systems_per_cta - 1) / systems_per_cta;

  mfa_kernel<<<num_cta, nt>>>(mfa);
}

template<typename real_t>
void fft_kernel(int n, const real_t* input_global, 
  complex_t<real_t>* output_global, int count) {

  switch(n) {
    case 4: small_fft_kernel<4>(input_global, output_global, count); break;
    case 8: small_fft_kernel<8>(input_global, output_global, count); break;
    case 16: small_fft_kernel<16>(input_global, output_global, count); break;
    case 32: large_fft_kernel<32>(input_global, output_global, count); break;
    case 64: large_fft_kernel<64>(input_global, output_global, count); break;
    case 128: large_fft_kernel<128>(input_global, output_global, count); break;
    case 256: large_fft_kernel<256>(input_global, output_global, count); break;
    case 512: large_fft_kernel<512>(input_global, output_global, count); break;
    case 1024: fft_kernel_1024(input_global, output_global, count); break;
    default:
      assert(0);
      
  }
}

} // namespace mgpu
