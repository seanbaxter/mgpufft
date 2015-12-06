#pragma once
#include <cuda.h>

namespace mgpu {

template<typename func_t>
__global__ void transform_kernel(func_t func, int num_threads) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < num_threads)
    func(gid);
}

template<typename func_t>
void gpu_transform(func_t f, int num_threads, cudaStream_t stream = 0,
  int nt = 128) {

  int num_blocks = (num_threads + nt - 1) / nt;
  transform_kernel<<<num_blocks, nt, 0, stream>>>(f, num_threads);
}


template<int a1, typename func_t>
__global__ void transform_block_a1_k(func_t func) {
  func.template block<a1>(threadIdx.x, blockIdx.x);
}

template<int a1, typename func_t>
void gpu_transform_block_a1(func_t f, int num_blocks, int block_size) {
  transform_block_a1_k<a1><<<num_blocks, block_size>>>(f);
}

}
