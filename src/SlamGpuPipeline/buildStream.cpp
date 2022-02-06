#include "SlamGpuPipeline.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../cuda/cuda_RGB_to_Grayscale.cuh"
#include "../cuda/orb.cuh"
#include "../cuda/match_features.cuh"
#include "../cuda/cuda-align.cuh"
#include "../cuda/fast.cuh"
#include "../cuda/post_processing.cuh"
#include "../cuda_common.h"
#include "defines.h"

#include <npp.h>

#include <unistd.h> // for sleep function
#include <chrono>
#include <nvjpeg.h>
#include <cmath>
#include <numeric>

#include <cstdio> // for printf

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std::chrono;
using namespace Eigen;
using namespace std;

namespace Jetracer
{

    void SlamGpuPipeline::buildStream(int slam_frames_id)
    {
        // spread threads between cores
        // cpu_set_t cpuset;
        // CPU_ZERO(&cpuset);
        // CPU_SET(slam_frames_id + 1, &cpuset);

        // int rc = pthread_setaffinity_np(slam_frames[slam_frames_id]->gpu_thread.native_handle(),
        //                                 sizeof(cpu_set_t), &cpuset);
        // if (rc != 0)
        // {
        //     std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
        // }

        std::cout
            << "GPU thread " << slam_frames_id << " is started on CPU: "
            << sched_getcpu() << std::endl;

        // cv::Ptr<cv::FeatureDetector> detector_ = cv::ORB::create(3000);

        checkCudaErrors(cudaStreamCreateWithFlags(&stream_left_, cudaStreamNonBlocking));
        checkCudaErrors(cudaStreamCreateWithFlags(&stream_right_, cudaStreamNonBlocking));
        checkCudaErrors(cudaStreamCreateWithFlags(&nvjpeg_stream_, cudaStreamNonBlocking));

        int grid_cols = (_ctx->cam_w + CELL_SIZE_WIDTH - 1) / CELL_SIZE_WIDTH;
        int grid_rows = (_ctx->cam_h + CELL_SIZE_HEIGHT - 1) / CELL_SIZE_HEIGHT;
        std::size_t keypoints_num = grid_cols * grid_rows;

        int data_bytes = _ctx->cam_w * _ctx->cam_h * 3;
        size_t length;
        std::size_t width_char = sizeof(char) * _ctx->cam_w;
        // std::size_t width_float = sizeof(float) * _ctx->cam_w;
        std::size_t height = _ctx->cam_h;

        // std::size_t canny_pitch;
        // std::size_t gray_response_pitch;
        checkCudaErrors(cudaMallocPitch((void **)&d_rgb_image_, &rgb_pitch_, width_char * 3, height));
        checkCudaErrors(cudaMallocPitch((void **)&d_gray_image_, &gray_pitch_, width_char, height));
        checkCudaErrors(cudaMallocPitch((void **)&d_nvjpeg_rgb_image_, &nvjpeg_rgb_pitch_, width_char, height * 3));
        checkCudaErrors(cudaMalloc((void **)&d_keypoints_angle_left_, keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_keypoints_angle_right_, keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_descriptors_tmp_left_, keypoints_num * 32 * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc((void **)&d_valid_keypoints_num_left, sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_descriptors_tmp_right_, keypoints_num * 32 * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc((void **)&d_matched_keypoints_num, sizeof(int)));

        unsigned char *d_corner_lut;
        checkCudaErrors(cudaMalloc((void **)&d_corner_lut, 64 * 1024));

        // nvJPEG to encode RGB image to jpeg
        nvjpegHandle_t nv_handle;
        nvjpegEncoderState_t nv_enc_state;
        nvjpegEncoderParams_t nv_enc_params;
        int resize_quality = 90;

        CHECK_NVJPEG(nvjpegCreateSimple(&nv_handle));
        CHECK_NVJPEG(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, nvjpeg_stream_));
        CHECK_NVJPEG(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, nvjpeg_stream_));
        CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(nv_enc_params, resize_quality, nvjpeg_stream_));
        CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_420, nvjpeg_stream_));
        // CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 0, NULL));
        nvjpegImage_t nv_image;

        /*
         * Preallocate FeaturePoint struct of arrays
         * Note to future self:
         * we use SoA, because of the efficient bearing vector calculation
         * float x_
         * float y_:                                      | 2x float
         * float score_:                                  | 1x float
         * int level_:                                    | 1x int
         * float3 point                                   | 3x float
         */
        const std::size_t bytes_per_featurepoint = sizeof(float) * 4;
        std::size_t feature_grid_bytes = keypoints_num * bytes_per_featurepoint;

        checkCudaErrors(cudaMalloc((void **)&d_score_left_, keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_score_right_, keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_level_left_, keypoints_num * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_level_right_, keypoints_num * sizeof(int)));

        // Preparing image pyramid
        int prev_width;
        int prev_height;

        // Preparing NPP
        // NppiSize oSrcSize = {(int)_ctx->cam_w, (int)_ctx->cam_h};
        // NppiPoint oSrcOffset = {0, 0};
        // NppiSize oSizeROI = {(int)_ctx->cam_w, (int)_ctx->cam_h};
        // int nBufferSize = 0;
        // Npp8u *pScratchBufferNPP = 0;
        // nppiFilterCannyBorderGetBufferSize(oSizeROI, &nBufferSize);
        // cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

        // Npp16s nLowThreshold = 20;
        // Npp16s nHighThreshold = 256;

        for (int i = 0; i < PYRAMID_LEVELS; i++)
        {
            pyramid_t level_left;
            pyramid_t level_right;
            if (i != 0)
            {
                level_left.image_width = prev_width / 2;
                level_left.image_height = prev_height / 2;
                level_right.image_width = prev_width / 2;
                level_right.image_height = prev_height / 2;
            }
            else
            {
                level_left.image_width = _ctx->cam_w;
                level_left.image_height = _ctx->cam_h;
                level_right.image_width = _ctx->cam_w;
                level_right.image_height = _ctx->cam_h;
            }
            checkCudaErrors(cudaMallocPitch((void **)&level_left.image,
                                            &level_left.image_pitch,
                                            level_left.image_width * sizeof(char),
                                            level_left.image_height));
            checkCudaErrors(cudaMallocPitch((void **)&level_left.response,
                                            &level_left.response_pitch,
                                            level_left.image_width * sizeof(float),
                                            level_left.image_height));
            pyramid_left_.push_back(level_left);

            checkCudaErrors(cudaMallocPitch((void **)&level_right.image,
                                            &level_right.image_pitch,
                                            level_right.image_width * sizeof(char),
                                            level_right.image_height));
            checkCudaErrors(cudaMallocPitch((void **)&level_right.response,
                                            &level_right.response_pitch,
                                            level_right.image_width * sizeof(float),
                                            level_right.image_height));
            pyramid_right_.push_back(level_right);

            prev_width = level_left.image_width;
            prev_height = level_left.image_height;
        }

        fast_gpu_calculate_lut(d_corner_lut, FAST_MIN_ARC_LENGTH);

        // Create cuda events for streams syncronization
        checkCudaErrors(cudaEventCreate(&event_ir_right_detected_));

        while (!slam_frames[slam_frames_id]->exit_gpu_pipeline)
        {
            std::unique_lock<std::mutex> lk(slam_frames[slam_frames_id]->thread_mutex);
            while (!slam_frames[slam_frames_id]->image_ready_for_process && !slam_frames[slam_frames_id]->exit_gpu_pipeline)
                slam_frames[slam_frames_id]->thread_cv.wait(lk);

            if (!slam_frames[slam_frames_id]->image_ready_for_process)
                continue;

            std::shared_ptr<Jetracer::slam_frame_t> slam_frame = std::make_shared<slam_frame_t>();
            slam_frame->theta = slam_frames[slam_frames_id]->rgbd_frame->theta;

            // std::shared_ptr<float[]> h_feature_grid(new float[keypoints_num * 4]);
            // std::shared_ptr<int2[]> h_matching_pairs(new int2[keypoints_num]);
            // checkCudaErrors(cudaMalloc((void **)&slam_frame->d_points_left, keypoints_num * 3 * sizeof(double)));
            // checkCudaErrors(cudaMalloc((void **)&slam_frame->d_points_right, keypoints_num * 3 * sizeof(double)));
            checkCudaErrors(cudaMalloc((void **)&slam_frame->d_descriptors_left, keypoints_num * sizeof(uint32_t)));
            checkCudaErrors(cudaMalloc((void **)&slam_frame->d_descriptors_right, keypoints_num * sizeof(uint32_t)));
            checkCudaErrors(cudaMalloc((void **)&slam_frame->d_pos_left, keypoints_num * sizeof(float2)));
            checkCudaErrors(cudaMalloc((void **)&slam_frame->d_pos_right, keypoints_num * sizeof(float2)));
            checkCudaErrors(cudaMalloc((void **)&slam_frame->d_matching_pairs, keypoints_num * sizeof(int2)));

            std::chrono::_V2::system_clock::time_point stop;
            std::chrono::_V2::system_clock::time_point start = high_resolution_clock::now();

            checkCudaErrors(cudaMemcpy2DAsync((void *)d_gray_image_,
                                              gray_pitch_,
                                              slam_frames[slam_frames_id]->rgbd_frame->ir_image_left,
                                              _ctx->cam_w,
                                              _ctx->cam_w,
                                              _ctx->cam_h,
                                              cudaMemcpyHostToDevice,
                                              stream_left_));

            checkCudaErrors(cudaMemcpy2DAsync((void *)pyramid_left_[0].image,
                                              pyramid_left_[0].image_pitch,
                                              slam_frames[slam_frames_id]->rgbd_frame->ir_image_left,
                                              _ctx->cam_w,
                                              _ctx->cam_w,
                                              _ctx->cam_h,
                                              cudaMemcpyHostToDevice,
                                              stream_left_));

            checkCudaErrors(cudaMemcpy2DAsync((void *)pyramid_right_[0].image,
                                              pyramid_right_[0].image_pitch,
                                              slam_frames[slam_frames_id]->rgbd_frame->ir_image_right,
                                              _ctx->cam_w,
                                              _ctx->cam_w,
                                              _ctx->cam_h,
                                              cudaMemcpyHostToDevice,
                                              stream_right_));

            pyramid_create_levels(pyramid_left_, stream_left_);
            pyramid_create_levels(pyramid_right_, stream_right_);

            detect(pyramid_left_,
                   d_corner_lut,
                   FAST_EPSILON,
                   slam_frame->d_pos_left,
                   d_score_left_,
                   d_level_left_,
                   stream_left_);

            detect(pyramid_right_,
                   d_corner_lut,
                   FAST_EPSILON,
                   slam_frame->d_pos_right,
                   d_score_right_,
                   d_level_right_,
                   stream_right_);

            // printf("compute_fast_angle left\n");
            compute_fast_angle(d_keypoints_angle_left_,
                               slam_frame->d_pos_left,
                               pyramid_left_[0].image,
                               pyramid_left_[0].image_pitch,
                               _ctx->cam_w,
                               _ctx->cam_h,
                               keypoints_num,
                               stream_left_);

            // printf("compute_fast_angle right\n");
            compute_fast_angle(d_keypoints_angle_right_,
                               slam_frame->d_pos_right,
                               pyramid_right_[0].image,
                               pyramid_right_[0].image_pitch,
                               _ctx->cam_w,
                               _ctx->cam_h,
                               keypoints_num,
                               stream_right_);

            // printf("calc_orb left\n");
            calc_orb(d_keypoints_angle_left_,
                     slam_frame->d_pos_left,
                     d_descriptors_tmp_left_,
                     slam_frame->d_descriptors_left,
                     pyramid_left_[0].image,
                     pyramid_left_[0].image_pitch,
                     _ctx->cam_w,
                     _ctx->cam_h,
                     keypoints_num,
                     stream_left_);

            // printf("calc_orb right\n");
            calc_orb(d_keypoints_angle_right_,
                     slam_frame->d_pos_right,
                     d_descriptors_tmp_right_,
                     slam_frame->d_descriptors_right,
                     pyramid_right_[0].image,
                     pyramid_right_[0].image_pitch,
                     _ctx->cam_w,
                     _ctx->cam_h,
                     keypoints_num,
                     stream_right_);

            checkCudaErrors(cudaEventRecord(event_ir_right_detected_, stream_right_));

            // checkCudaErrors(cudaMemcpyAsync((void *)h_feature_grid.get(),
            //                                 d_feature_grid_left,
            //                                 feature_grid_bytes,
            //                                 cudaMemcpyDeviceToHost,
            //                                 stream_left_));

            // wait for finishing processing right image and then match keypoints on left and right images
            checkCudaErrors(cudaStreamWaitEvent(stream_left_, event_ir_right_detected_));

            h_matched_keypoints_num = 0;
            match_keypoints(slam_frame->d_matching_pairs,
                            slam_frame->d_descriptors_left,
                            d_score_left_,
                            slam_frame->d_pos_left,
                            slam_frame->d_descriptors_right,
                            slam_frame->d_pos_right,
                            _ctx->max_descriptor_distance,
                            _ctx->min_score,
                            _ctx->max_points_distance,
                            keypoints_num,
                            keypoints_num,
                            d_matched_keypoints_num,
                            &h_matched_keypoints_num,
                            stream_left_);

            // checkCudaErrors(cudaMemcpyAsync((void *)h_matching_pairs.get(),
            //                                 slam_frame->d_matching_pairs,
            //                                 h_matched_keypoints_num * sizeof(int2),
            //                                 cudaMemcpyDeviceToHost,
            //                                 stream_left_));

            // checkCudaErrors(cudaStreamSynchronize(stream_left_));
            //------------- compressing image for sending over network ------------
            // Fill nv_image with image data, let's say 848x480 image in RGB format
            for (int i = 0; i < 3; i++)
            {
                unsigned char *d_channel_begin = d_nvjpeg_rgb_image_ + nvjpeg_rgb_pitch_ * _ctx->cam_h * i;
                checkCudaErrors(cudaMemcpy2DAsync((void *)d_channel_begin,
                                                  nvjpeg_rgb_pitch_,
                                                  d_gray_image_,
                                                  gray_pitch_,
                                                  _ctx->cam_w,
                                                  _ctx->cam_h,
                                                  cudaMemcpyDeviceToDevice,
                                                  nvjpeg_stream_));

                nv_image.channel[i] = d_channel_begin;
                nv_image.pitch[i] = nvjpeg_rgb_pitch_;
            };

            // overlay keypoints on grayscale or color image
            overlay_keypoints(nv_image.channel[1],
                              nv_image.pitch[1],
                              slam_frame->d_pos_left,
                              d_score_left_,
                              _ctx->min_score,
                              _d_ir_intristics_left, // ir intrinsics
                              keypoints_num,
                              nvjpeg_stream_);

            // Compress image
            CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
                                           &nv_image, NVJPEG_INPUT_RGB, _ctx->cam_w, _ctx->cam_h, nvjpeg_stream_));

            // getting rotation angles
            // Eigen::Matrix3d R = T_c2w_prev_curr.block<3, 3>(0, 0);
            // Vector3d eu_angles = R.eulerAngles(0, 1, 2);
            // Vector3d angles(floor(eu_angles(0) * 180 / CUDART_PI_D),
            //                 floor(eu_angles(1) * 180 / CUDART_PI_D),
            //                 floor(eu_angles(2) * 180 / CUDART_PI_D));

            // -------------------------------------------------
            // TODO: convert cudaStreamSynchronize to use events
            // -------------------------------------------------

            // get compressed stream size
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, nvjpeg_stream_));
            checkCudaErrors(cudaStreamSynchronize(nvjpeg_stream_));
            // get stream itself
            slam_frame->image = (unsigned char *)malloc(length * sizeof(char));
            slam_frame->image_length = length;
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, (slam_frame->image), &length, nvjpeg_stream_));
            checkCudaErrors(cudaStreamSynchronize(nvjpeg_stream_));
            checkCudaErrors(cudaStreamSynchronize(stream_left_));
            checkCudaErrors(cudaStreamSynchronize(stream_right_));

            stop = high_resolution_clock::now();
            auto rotation_timer = high_resolution_clock::now();

            auto duration = duration_cast<microseconds>(stop - start);
            auto duration_rotation = duration_cast<microseconds>(rotation_timer - stop);

            // slam_frame->image = image;
            slam_frame->keypoints_count = keypoints_num;
            slam_frame->h_matched_keypoints_num = h_matched_keypoints_num;
            slam_frame->h_valid_keypoints_num = keypoints_num;

            pEvent newEvent = std::make_shared<BaseEvent>();
            newEvent->event_type = EventType::event_gpu_slam_frame;
            newEvent->message = slam_frame;
            _ctx->sendEvent(newEvent);

            slam_frames[slam_frames_id]->image_ready_for_process = false;

            std::lock_guard<std::mutex> lock(m_gpu_mutex);
            deleted_slam_frames.push_back(slam_frames_id);

            // std::cout << "Finished work GPU thread " << slam_frames_id
            //           << " duration " << duration.count()
            //           << " duration_rotation " << duration_rotation.count()
            //           << " keypoints_num " << keypoints_num
            //           << " h_matched_keypoints_num " << h_matched_keypoints_num
            //           //   << " theta.x " << theta.x
            //           //   << " theta.y " << theta.y
            //           //   << " theta.z " << theta.z
            //           //   << " keypoints_num " << keypoints_num
            //           << std::endl;
        }

        // checkCudaErrors(cudaFree(d_rgb_image_));
        checkCudaErrors(cudaFree(d_gray_image_));
        // checkCudaErrors(cudaFree(d_aligned_out));
        // checkCudaErrors(cudaFree(d_depth_in));
        // checkCudaErrors(cudaFree(d_keypoints_exist));
        // checkCudaErrors(cudaFree(d_keypoints_response));
        // printf("---mark---\n");
        checkCudaErrors(cudaFree(d_keypoints_angle_left_));
        checkCudaErrors(cudaFree(d_descriptors_tmp_left_));

        checkCudaErrors(cudaFree(d_keypoints_angle_right_));
        checkCudaErrors(cudaFree(d_descriptors_tmp_right_));

        std::cout << "Stopped GPU thread " << slam_frames_id << std::endl;
    }

}