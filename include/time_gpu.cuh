#pragma once

template <typename Real>
struct TimeDevice {
    Real* dt_mins_d = nullptr;
    Real* dt_mins_h = nullptr;
    size_t shared_mem_size = 0;
    int n_blocks;

    TimeDevice(CudaManager<Real>& cuda) :
        n_blocks(cuda.grid_dim.x * cuda.grid_dim.y * cuda.grid_dim.z)
    {
        dt_mins_h = new Real[n_blocks];
        shared_mem_size = sizeof(Real) * cuda.block_dim.x * cuda.block_dim.y * cuda.block_dim.z;
        CUDA_CHECK(cudaMalloc(&dt_mins_d, sizeof(Real) * n_blocks) );
    }

    void copy_to_host()
    {
        CUDA_CHECK(cudaMemcpy(dt_mins_h, dt_mins_d, sizeof(Real) * n_blocks, cudaMemcpyDeviceToHost) );
    }

    void copy_to_device()
    {
        CUDA_CHECK(cudaMemcpy(dt_mins_d, dt_mins_h, sizeof(Real) * n_blocks, cudaMemcpyHostToDevice) );
    }
};