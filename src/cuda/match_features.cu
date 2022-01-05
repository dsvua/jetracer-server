#include "match_features.cuh"
#include "../cuda_common.h"

namespace Jetracer
{
    __inline__ __device__ int hamming_distance(int descriptor_left, int descriptor_right)
    {
        return __popc(descriptor_left ^ descriptor_right);
    }

    __global__ void match_keypoints_kernel(int2 *matching_pairs,
                                           uint32_t *descriptors_left,
                                           float *score_left,
                                           float2 *pos_left,
                                           uint32_t *descriptors_right,
                                           float2 *pos_right,
                                           int max_descriptor_distance,
                                           float min_score,
                                           float max_points_distance,
                                           int keypoints_num_left,
                                           int keypoints_num_right,
                                           int *keypoints_num_matched)
    {
        __shared__ int warp_counter;
        __shared__ int global_idx;

        int min_descriptor_distance = max_descriptor_distance;
        int descriptor_distance;
        float points_distance;
        int2 current_pair;
        bool is_matched = false;
        int local_idx = 0;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        __syncthreads();

        if (threadIdx.x == 0)
        {
            warp_counter = 0;
        };

        __syncthreads();

        if (idx < keypoints_num_left)
        {
            if (score_left[idx] > min_score)
            {
                for (int idx_right = 0; idx_right < keypoints_num_right; idx_right++)
                {
                    descriptor_distance = hamming_distance(descriptors_left[idx], descriptors_right[idx_right]);
                    float delta_x = pos_right[idx_right].x - pos_left[idx].x;
                    float delta_y = pos_right[idx_right].y - pos_left[idx].y;
                    points_distance = delta_x * delta_x + delta_y * delta_y;

                    if (descriptor_distance < min_descriptor_distance &&
                        points_distance < max_points_distance &&
                        pos_left[idx].x < pos_right[idx_right].x + 1)
                    {
                        current_pair.x = idx;
                        current_pair.y = idx_right;
                        min_descriptor_distance = descriptor_distance;
                        is_matched = true;
                    };
                };
            };
        };

        // __syncthreads();

        if (is_matched)
        {
            local_idx = atomicAdd(&warp_counter, 1);
        };

        __syncthreads();

        if (threadIdx.x == 0 && warp_counter > 0)
        {
            global_idx = atomicAdd(keypoints_num_matched, warp_counter);
        };

        __syncthreads();

        if (is_matched)
        {
            matching_pairs[global_idx + local_idx] = current_pair;
        };
    }

    void match_keypoints(int2 *matching_pairs,
                         uint32_t *descriptors_left,
                         float *score_left,
                         float2 *pos_left,
                         uint32_t *descriptors_right,
                         float2 *pos_right,
                         int max_descriptor_distance,
                         float min_score,
                         float max_points_distance,
                         int keypoints_num_left,
                         int keypoints_num_right,
                         int *d_keypoints_num_matched,
                         int *h_keypoints_num_matched,
                         cudaStream_t stream)
    {
        
        checkCudaErrors(cudaMemcpyAsync((void *)d_keypoints_num_matched,
                                        h_keypoints_num_matched,
                                        sizeof(int),
                                        cudaMemcpyHostToDevice,
                                        stream));


        dim3 threads(CUDA_WARP_SIZE);
        dim3 blocks(calc_block_size(keypoints_num_left, threads.x));

        match_keypoints_kernel<<<blocks, threads, 0, stream>>>(matching_pairs,
                                                               descriptors_left,
                                                               score_left,
                                                               pos_left,
                                                               descriptors_right,
                                                               pos_right,
                                                               max_descriptor_distance,
                                                               min_score,
                                                               max_points_distance,
                                                               keypoints_num_left,
                                                               keypoints_num_right,
                                                               d_keypoints_num_matched);

        checkCudaErrors(cudaMemcpyAsync((void *)h_keypoints_num_matched,
                                                d_keypoints_num_matched,
                                                sizeof(int),
                                                cudaMemcpyDeviceToHost,
                                                stream));
    }
}
