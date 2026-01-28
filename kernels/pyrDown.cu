#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>
#include <iostream>
#include "nvtx3.hpp"

const float h_gaussian_window[5][5] = {{1, 4, 7, 4, 1},
                                    {4, 16, 26, 16, 4},
                                    {7, 26, 41, 26, 7},
                                    {4, 16, 26, 16, 4},
                                    {1, 4, 7, 4, 1}
};
__constant__ float gaussian_window[5][5];

template<typename T> __device__ T mirror(T current, T length){
    if (current < 0) {
        current = -current;
    }
    else if (current >= length) {
        current = 2 * length - 2 - current;
    }
    return current;
}

__global__ void gaussian_blur(unsigned char* input, unsigned char* output, int width, int height){
    int pixel_col = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_row = blockIdx.y * blockDim.y + threadIdx.y;
    int window_step = 2;
    int window_row = threadIdx.y + window_step;
    int window_col = threadIdx.x + window_step;

    __shared__ unsigned char shared_tile_img[20][20][3];

    int thread_num = threadIdx.y * blockDim.x + threadIdx.x;
    int block_start_row = blockIdx.y * blockDim.y;
    int block_start_col = blockIdx.x * blockDim.x;

    for (int i = thread_num; i< (blockDim.x + 2*window_step)*(blockDim.y + 2*window_step); i += blockDim.x * blockDim.y) {
        int tile_row = i/20;
        int tile_col = i%20;
        int global_row = block_start_row + tile_row - window_step;
        int global_col = block_start_col + tile_col - window_step;

        global_row = mirror<int>(global_row, height);
        global_col = mirror<int>(global_col, width);

        if (global_row < height && global_col < width) {
            int pixel_idx = (global_row * width + global_col) * 3;
            shared_tile_img[tile_row][tile_col][0] = input[pixel_idx];
            shared_tile_img[tile_row][tile_col][1] = input[pixel_idx + 1];
            shared_tile_img[tile_row][tile_col][2] = input[pixel_idx + 2];
        }
    }

    __syncthreads();


    if (pixel_col < width && pixel_row < height) {
        float pixel_sum_b = 0.0f;
        float pixel_sum_g = 0.0f;
        float pixel_sum_r = 0.0f;

        for (int blur_row = - window_step; blur_row <= window_step; blur_row++) {
            for (int blur_col = - window_step; blur_col <=  window_step; blur_col++) {
                pixel_sum_b += shared_tile_img[window_row + blur_row][window_col + blur_col][0]* gaussian_window[blur_col + window_step][blur_row + window_step];
                pixel_sum_g += shared_tile_img[window_row + blur_row][window_col + blur_col][1]* gaussian_window[blur_col + window_step][blur_row + window_step];
                pixel_sum_r += shared_tile_img[window_row + blur_row][window_col + blur_col][2]* gaussian_window[blur_col + window_step][blur_row + window_step];
            }
        }
        int b_pixel = (pixel_row * width + (pixel_col)) * 3;
        int g_pixel = b_pixel + 1;
        int r_pixel = b_pixel + 2;
        float output_b_pixel = (1.0/273.0) * pixel_sum_b;
        float output_g_pixel = (1.0/273.0) * pixel_sum_g;
        float output_r_pixel = (1.0/273.0) * pixel_sum_r;
        output[b_pixel] = (unsigned char)output_b_pixel;
        output[g_pixel] = (unsigned char)output_g_pixel;
        output[r_pixel] = (unsigned char)output_r_pixel;
    }

}

void resize(unsigned char* d_input_in, unsigned char* r_output_in, int height, int width, int h, int w, cudaStream_t stream){
    uchar3* d_input = (uchar3*)d_input_in;
    uchar3* r_output = (uchar3*)r_output_in;

    cub::DeviceTransform::Transform(thrust::make_counting_iterator(0), r_output, h*w,
        [=] __device__ (int idx) {
           int col = idx % w;
           int row = idx / w;

            int in_col = col*2 + 1;
            int in_row = row*2 + 1;

           return d_input[in_row*width + in_col];
    }, stream);
}

torch::Tensor pyrDown(torch::Tensor img) {
    nvtxRangePushA("pyrDown");
    int image_width = img.size(1);
    int image_height = img.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cudaMemcpyToSymbol(gaussian_window, h_gaussian_window, 25 * sizeof(float));

    auto gauss_output = torch::empty({image_height, image_width, 3},
                               torch::TensorOptions()
                                   .dtype(torch::kByte)
                                   .device(img.device()));
    unsigned char *d_output = gauss_output.data_ptr<unsigned char>();
    unsigned char *d_input = img.data_ptr<unsigned char>();
    int image_size = image_width*image_height * 3 * sizeof(unsigned char);

    dim3 threads(16, 16);
    dim3 blocks((image_width + threads.x - 1) / threads.x,
                (image_height + threads.y - 1) / threads.y);

    nvtxRangePushA("gaussian_blur");
    gaussian_blur<<<blocks, threads, 0, stream>>>(d_input, d_output, image_width, image_height);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nvtxRangePop();

    int new_height = image_height/2;
    int new_width = image_width/2;

    auto result = torch::empty({new_height, new_width, 3},
                               torch::TensorOptions()
                                   .dtype(torch::kByte)
                                   .device(img.device()));

    nvtxRangePushA("resize_cub");
    resize(d_output, result.data_ptr<unsigned char>(), image_height, image_width, new_height, new_width, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nvtxRangePop();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return img;
    }
    nvtxRangePop();
    return result;
}