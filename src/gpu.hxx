#pragma once

#include <cuda.h>
#include <vector>

namespace mgpu {

template<typename type_t>
type_t* host_to_device(const std::vector<type_t>& host) {
  type_t* device;
  cudaMalloc((void**)&device, sizeof(type_t) * host.size());
  cudaMemcpy(device, host.data(), sizeof(type_t) * host.size(), 
    cudaMemcpyHostToDevice);
  return device;
}

template<typename type_t>
std::vector<type_t> device_to_host(const type_t* device, int size) {
  std::vector<type_t> host(size);
  cudaMemcpy(host.data(), device, sizeof(type_t) * size, 
    cudaMemcpyDeviceToHost);
  return host;
}

struct timer_t {
	cudaEvent_t event_start, event_stop;

  timer_t() {
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
  }
  ~timer_t() {
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);
  }

  void start() {
    cudaEventRecord(event_start, 0);
  }
  double stop() {
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, event_start, event_stop);
    return elapsed / 1000;
  }
};

}
