#ifndef JETRACER_CUDA_NMS_KEYPOINTS_H
#define JETRACER_CUDA_NMS_KEYPOINTS_H

#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace Jetracer
{
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
                         cudaStream_t stream);
}

#endif // JETRACER_CUDA_NMS_KEYPOINTS_H
