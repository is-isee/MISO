#pragma once
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cassert>

template <typename T>
class Array3DDevice {
public:
    T* array = nullptr;
    int i_total, j_total, k_total;

    Array3DDevice() = default;

    Array3DDevice(int i_total_, int j_total_, int k_total_) :
        i_total(i_total_), j_total(j_total_), k_total(k_total_) {
        cudaMalloc(&array, sizeof(T) * i_total_ * j_total_ * k_total_);
    }

    ~Array3DDevice() {}

    void free() {
        if (array) cudaFree(array); array = nullptr;
    }

    __device__ inline T& operator[](int idx) {
        return array[idx];
    }

    __device__ inline const T& operator[](int idx) const {
        return array[idx];
    }

    void copy_from_host(const Array3D<T>& array_host) {
        cudaMemcpy(array, array_host.data(), sizeof(T) * i_total * j_total * k_total, cudaMemcpyHostToDevice);
    }

    void copy_to_host(Array3D<T>& array_host) const {
        cudaMemcpy(array_host.data(), array, sizeof(T) * i_total * j_total * k_total, cudaMemcpyDeviceToHost);
    }

    void copy_from_device(const Array3DDevice<T>& array_device) {
        cudaMemcpy(array, array_device.data(), sizeof(T) * i_total * j_total * k_total, cudaMemcpyDeviceToDevice);
    }

    __host__ __device__ inline T* data() { return array; }
    __host__ __device__ inline const T* data() const { return array; }

    // コピー操作は禁止（誤用防止）
    Array3DDevice(const Array3DDevice&) = default;
    Array3DDevice& operator=(const Array3DDevice&) = default;
};

#endif // USE_CUDA