#ifndef JETRACER_SLAM_TYPES_THREAD_H
#define JETRACER_SLAM_TYPES_THREAD_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <eigen3/Eigen/Eigen>
#include "../RealSense/RealSenseD400.h"

namespace Jetracer
{

#pragma once

    typedef struct slam_frame_callback
    {
        std::shared_ptr<rgbd_frame_t> rgbd_frame;
        bool image_ready_for_process;
        bool exit_gpu_pipeline;
        std::thread gpu_thread;
        std::mutex thread_mutex;
        std::condition_variable thread_cv;

    } slam_frame_callback_t;

    typedef struct slam_frame
    {
        unsigned char *image;
        size_t image_length;
        // uint16_t *keypoints_x;
        // uint16_t *keypoints_y;
        std::shared_ptr<double[]> h_points;
        std::shared_ptr<uint32_t[]> h_descriptors;

        // double *d_points_left;
        // double *d_points_right;
        uint32_t *d_descriptors_left;
        uint32_t *d_descriptors_right;
        float2 *d_pos_left;
        float2 *d_pos_right;
        int2 *d_matching_pairs;

        int keypoints_count;
        int h_valid_keypoints_num;
        int h_matched_keypoints_num;
        float3 theta;
        std::shared_ptr<rgbd_frame_t> rgbd_frame;

        Eigen::Matrix4d T_c2w;
        Eigen::Matrix4d T_w2c;

        ~slam_frame()
        {
            if (image)
                free(image);
            // if (keypoints_x)
            //     free(keypoints_x);
            // if (keypoints_y)
            //     free(keypoints_y);
            // if (d_points_left)
            //     checkCudaErrors(cudaFree(d_points_left));
            // if (d_points_right)
            //     checkCudaErrors(cudaFree(d_points_right));
            if (d_descriptors_left)
                checkCudaErrors(cudaFree(d_descriptors_left));
            if (d_descriptors_right)
                checkCudaErrors(cudaFree(d_descriptors_right));
            if (d_pos_left)
                checkCudaErrors(cudaFree(d_pos_left));
            if (d_pos_right)
                checkCudaErrors(cudaFree(d_pos_right));
            if (d_matching_pairs)
                checkCudaErrors(cudaFree(d_matching_pairs));
            // T_c2w.~MatrixBase();
            // T_w2c.~MatrixBase();
        }

    } slam_frame_t;
}

#endif // JETRACER_SLAM_TYPES_THREAD_H
