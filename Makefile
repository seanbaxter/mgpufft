# To build with FBFFT, amend this line with -I <facebook_path> -D __BENCHMARK_FBFFT__
all:
	nvcc -g -std=c++11 -lineinfo -arch=sm_35 --use_fast_math -lcufft -Xptxas="-v" -o fft_benchmark -I src/ test/fft_benchmark.cu
