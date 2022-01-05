#include "SlamGpuPipeline.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../cuda/cuda_RGB_to_Grayscale.cuh"
#include "../cuda/orb.cuh"
#include "../cuda/match_features.cuh"
#include "../cuda/cuda-align.cuh"
#include "../cuda/pyramid.cuh"
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

        cudaStream_t stream_left;
        cudaStream_t stream_right;
        cudaStream_t nvjpeg_stream;
        checkCudaErrors(cudaStreamCreateWithFlags(&stream_left, cudaStreamNonBlocking));
        checkCudaErrors(cudaStreamCreateWithFlags(&stream_right, cudaStreamNonBlocking));
        checkCudaErrors(cudaStreamCreateWithFlags(&nvjpeg_stream, cudaStreamNonBlocking));

        // allocate memory for image processing
        unsigned char *d_rgb_image;
        unsigned char *d_gray_image;
        unsigned char *d_nvjpeg_rgb_image;
        // unsigned int *d_aligned_out;
        // unsigned char *d_canny;
        // uint16_t *d_depth_in;
        // int2 *d_pixel_map;
        // float *d_gray_keypoint_response;
        // bool *d_keypoints_exist;
        // float *d_keypoints_response;
        float *d_keypoints_angle_left;
        float *d_keypoints_angle_right;
        // float2 *d_keypoints_pos;
        int2 *d_matching_pairs;
        unsigned char *d_descriptors_tmp_left;
        unsigned char *d_descriptors_tmp_right;
        uint32_t *d_descriptors_left;  // convert 32xchars into 1 x uint32_t
        uint32_t *d_descriptors_right; // convert 32xchars into 1 x uint32_t
        int *d_valid_keypoints_num_left;
        int *d_matched_keypoints_num;
        int h_matched_keypoints_num = 0;

        int grid_cols = (_ctx->cam_w + CELL_SIZE_WIDTH - 1) / CELL_SIZE_WIDTH;
        int grid_rows = (_ctx->cam_h + CELL_SIZE_HEIGHT - 1) / CELL_SIZE_HEIGHT;
        std::size_t keypoints_num = grid_cols * grid_rows;
        // std::size_t keypoints_num = 500;

        int data_bytes = _ctx->cam_w * _ctx->cam_h * 3;
        size_t length;
        std::size_t width_char = sizeof(char) * _ctx->cam_w;
        // std::size_t width_float = sizeof(float) * _ctx->cam_w;
        std::size_t height = _ctx->cam_h;

        std::size_t rgb_pitch;
        std::size_t gray_pitch;
        std::size_t nvjpeg_rgb_pitch;
        // std::size_t canny_pitch;
        // std::size_t gray_response_pitch;
        checkCudaErrors(cudaMallocPitch((void **)&d_rgb_image, &rgb_pitch, width_char * 3, height));
        checkCudaErrors(cudaMallocPitch((void **)&d_gray_image, &gray_pitch, width_char, height));
        checkCudaErrors(cudaMallocPitch((void **)&d_nvjpeg_rgb_image, &nvjpeg_rgb_pitch, width_char, height * 3));
        // checkCudaErrors(cudaMallocPitch((void **)&d_canny, &canny_pitch, width_char, height));
        // checkCudaErrors(cudaMalloc((void **)&d_aligned_out, _ctx->cam_w * sizeof(unsigned int) * _ctx->cam_h));
        // checkCudaErrors(cudaMalloc((void **)&d_depth_in, _ctx->cam_w * sizeof(uint16_t) * _ctx->cam_h));
        // checkCudaErrors(cudaMalloc((void **)&d_pixel_map, _ctx->cam_w * sizeof(int2) * _ctx->cam_h * 2)); // it needs x2 size
        // checkCudaErrors(cudaMalloc((void **)&d_keypoints_exist, keypoints_num * sizeof(bool)));
        // checkCudaErrors(cudaMalloc((void **)&d_keypoints_response, keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_keypoints_angle_left, keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_keypoints_angle_right, keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_matching_pairs, keypoints_num * sizeof(int2)));
        // checkCudaErrors(cudaMalloc((void **)&d_keypoints_pos, keypoints_num * sizeof(float2)));
        checkCudaErrors(cudaMalloc((void **)&d_descriptors_tmp_left, keypoints_num * 32 * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc((void **)&d_descriptors_left, keypoints_num * sizeof(uint32_t)));
        checkCudaErrors(cudaMalloc((void **)&d_valid_keypoints_num_left, sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_descriptors_tmp_right, keypoints_num * 32 * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc((void **)&d_descriptors_right, keypoints_num * sizeof(uint32_t)));
        checkCudaErrors(cudaMalloc((void **)&d_matched_keypoints_num, sizeof(int)));

        unsigned char *d_corner_lut;
        checkCudaErrors(cudaMalloc((void **)&d_corner_lut, 64 * 1024));

        //nvJPEG to encode RGB image to jpeg
        nvjpegHandle_t nv_handle;
        nvjpegEncoderState_t nv_enc_state;
        nvjpegEncoderParams_t nv_enc_params;
        int resize_quality = 90;

        CHECK_NVJPEG(nvjpegCreateSimple(&nv_handle));
        CHECK_NVJPEG(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, nvjpeg_stream));
        CHECK_NVJPEG(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, nvjpeg_stream));
        CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(nv_enc_params, resize_quality, nvjpeg_stream));
        CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_420, nvjpeg_stream));
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

        float *d_feature_grid_left;
        float *d_feature_grid_right;
        checkCudaErrors(cudaMalloc((void **)&d_feature_grid_left, feature_grid_bytes));
        checkCudaErrors(cudaMalloc((void **)&d_feature_grid_right, feature_grid_bytes));
        float2 *d_pos_left = (float2 *)(d_feature_grid_left);
        float2 *d_pos_right = (float2 *)(d_feature_grid_right);
        float *d_score_left = (d_feature_grid_left + keypoints_num * 2);
        float *d_score_right = (d_feature_grid_right + keypoints_num * 2);
        int *d_level_left = (int *)(d_feature_grid_left + keypoints_num * 3);
        int *d_level_right = (int *)(d_feature_grid_right + keypoints_num * 3);
        // double *d_points;
        // double *d_points_prev;
        double *d_matched_points;
        double *d_matched_points_prev;
        int *d_matched_points_num; // should make sure there 3 or more points matched
        // checkCudaErrors(cudaMalloc((void **)&d_points, keypoints_num * 3 * sizeof(double)));
        // checkCudaErrors(cudaMalloc((void **)&d_points_prev, keypoints_num * 3 * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_matched_points, keypoints_num * 3 * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_matched_points_prev, keypoints_num * 3 * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_matched_points_num, sizeof(int)));

        // Preparing image pyramid
        std::vector<pyramid_t> pyramid_left;
        std::vector<pyramid_t> pyramid_right;
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
            pyramid_left.push_back(level_left);
            prev_width = level_left.image_width;
            prev_height = level_left.image_height;

            checkCudaErrors(cudaMallocPitch((void **)&level_right.image,
                                            &level_right.image_pitch,
                                            level_right.image_width * sizeof(char),
                                            level_right.image_height));
            checkCudaErrors(cudaMallocPitch((void **)&level_right.response,
                                            &level_right.response_pitch,
                                            level_right.image_width * sizeof(float),
                                            level_right.image_height));
            pyramid_right.push_back(level_right);
        }

        fast_gpu_calculate_lut(d_corner_lut, FAST_MIN_ARC_LENGTH);

        int h_valid_keypoints_num_previous = 500;
        Eigen::Matrix4d T_w2c;
        std::shared_ptr<Jetracer::slam_frame_t> previous_frame1 = nullptr;
        std::shared_ptr<Jetracer::slam_frame_t> previous_frame2 = nullptr;

        // Create cuda events for streams syncronization
        cudaEvent_t event_ir_right_detected;
        checkCudaErrors(cudaEventCreate(&event_ir_right_detected));

        while (!slam_frames[slam_frames_id]->exit_gpu_pipeline)
        {
            std::unique_lock<std::mutex> lk(slam_frames[slam_frames_id]->thread_mutex);
            while (!slam_frames[slam_frames_id]->image_ready_for_process && !slam_frames[slam_frames_id]->exit_gpu_pipeline)
                slam_frames[slam_frames_id]->thread_cv.wait(lk);

            if (!slam_frames[slam_frames_id]->image_ready_for_process)
                continue;

            std::shared_ptr<Jetracer::slam_frame_t> slam_frame = std::make_shared<slam_frame_t>();
            slam_frame->theta = slam_frames[slam_frames_id]->rgbd_frame->theta;
            // slam_frame->image = (unsigned char *)malloc(_ctx->cam_h * _ctx->cam_w * sizeof(char));

            int h_keypoints_num_matched = 0;

            std::shared_ptr<float[]> h_feature_grid(new float[keypoints_num * 4]);
            float2 *h_pos = (float2 *)(h_feature_grid.get());
            float *h_score = reinterpret_cast<float *>(h_feature_grid.get() + keypoints_num * 2);
            int *h_level = (int *)(h_feature_grid.get() + keypoints_num * 3);
            std::shared_ptr<double[]> h_points(new double[keypoints_num * 3]);
            std::shared_ptr<uint32_t[]> h_descriptors(new uint32_t[keypoints_num]);
            std::shared_ptr<double3[]> h_current_matched_points(new double3[keypoints_num]);
            std::shared_ptr<double3[]> h_previous_matched_points(new double3[keypoints_num]);
            std::shared_ptr<int2[]> h_matching_pairs(new int2[keypoints_num]);
            checkCudaErrors(cudaMalloc((void **)&slam_frame->d_points_left, keypoints_num * 3 * sizeof(double)));
            checkCudaErrors(cudaMalloc((void **)&slam_frame->d_descriptors_left, keypoints_num * sizeof(uint32_t)));
            checkCudaErrors(cudaMalloc((void **)&slam_frame->d_pos_left, keypoints_num * sizeof(float2)));

            std::chrono::_V2::system_clock::time_point stop;
            std::chrono::_V2::system_clock::time_point start = high_resolution_clock::now();

            checkCudaErrors(cudaMemcpy2DAsync((void *)d_gray_image,
                                              gray_pitch,
                                              slam_frames[slam_frames_id]->rgbd_frame->ir_image_left,
                                              _ctx->cam_w,
                                              _ctx->cam_w,
                                              _ctx->cam_h,
                                              cudaMemcpyHostToDevice,
                                              stream_left));

            checkCudaErrors(cudaMemcpy2DAsync((void *)pyramid_left[0].image,
                                              pyramid_left[0].image_pitch,
                                              slam_frames[slam_frames_id]->rgbd_frame->ir_image_left,
                                              _ctx->cam_w,
                                              _ctx->cam_w,
                                              _ctx->cam_h,
                                              cudaMemcpyHostToDevice,
                                              stream_left));

            checkCudaErrors(cudaMemcpy2DAsync((void *)pyramid_right[0].image,
                                              pyramid_right[0].image_pitch,
                                              slam_frames[slam_frames_id]->rgbd_frame->ir_image_right,
                                              _ctx->cam_w,
                                              _ctx->cam_w,
                                              _ctx->cam_h,
                                              cudaMemcpyHostToDevice,
                                              stream_right));

            pyramid_create_levels(pyramid_left, stream_left);
            pyramid_create_levels(pyramid_right, stream_right);

            detect(pyramid_left,
                   d_corner_lut,
                   FAST_EPSILON,
                   d_pos_left,
                   d_score_left,
                   d_level_left,
                   stream_left);

            detect(pyramid_right,
                   d_corner_lut,
                   FAST_EPSILON,
                   d_pos_right,
                   d_score_right,
                   d_level_right,
                   stream_right);

            compute_fast_angle(d_keypoints_angle_left,
                               d_pos_left,
                               pyramid_left[0].image,
                               pyramid_left[0].image_pitch,
                               _ctx->cam_w,
                               _ctx->cam_h,
                               keypoints_num,
                               stream_left);

            compute_fast_angle(d_keypoints_angle_right,
                               d_pos_right,
                               pyramid_right[0].image,
                               pyramid_right[0].image_pitch,
                               _ctx->cam_w,
                               _ctx->cam_h,
                               keypoints_num,
                               stream_right);

            calc_orb(d_keypoints_angle_left,
                     d_pos_left,
                     d_descriptors_tmp_left,
                     d_descriptors_left,
                     pyramid_left[0].image,
                     pyramid_left[0].image_pitch,
                     _ctx->cam_w,
                     _ctx->cam_h,
                     keypoints_num,
                     stream_left);

            calc_orb(d_keypoints_angle_right,
                     d_pos_right,
                     d_descriptors_tmp_right,
                     d_descriptors_right,
                     pyramid_right[0].image,
                     pyramid_right[0].image_pitch,
                     _ctx->cam_w,
                     _ctx->cam_h,
                     keypoints_num,
                     stream_right);

            checkCudaErrors(cudaEventRecord(event_ir_right_detected, stream_right));

            checkCudaErrors(cudaMemcpyAsync((void *)h_feature_grid.get(),
                                            d_feature_grid_left,
                                            feature_grid_bytes,
                                            cudaMemcpyDeviceToHost,
                                            stream_left));

            // wait for finishing processing right image and then match keypoints on left and right images
            checkCudaErrors(cudaStreamWaitEvent(stream_left, event_ir_right_detected));

            h_matched_keypoints_num = 0;
            match_keypoints(d_matching_pairs,
                            d_descriptors_left,
                            d_score_left,
                            d_pos_left,
                            d_descriptors_right,
                            d_pos_right,
                            _ctx->max_descriptor_distance,
                            _ctx->min_score,
                            _ctx->max_points_distance,
                            keypoints_num,
                            keypoints_num,
                            d_matched_keypoints_num,
                            &h_matched_keypoints_num,
                            stream_left);

            checkCudaErrors(cudaMemcpyAsync((void *)h_matching_pairs.get(),
                                            d_matching_pairs,
                                            h_matched_keypoints_num * sizeof(int2),
                                            cudaMemcpyDeviceToHost,
                                            stream_left));

            //------------- compressing image for sending over network ------------
            // Fill nv_image with image data, let's say 848x480 image in RGB format
            for (int i = 0; i < 3; i++)
            {
                unsigned char *d_channel_begin = d_nvjpeg_rgb_image + nvjpeg_rgb_pitch * _ctx->cam_h * i;
                checkCudaErrors(cudaMemcpy2DAsync((void *)d_channel_begin,
                                                  nvjpeg_rgb_pitch,
                                                  d_gray_image,
                                                  gray_pitch,
                                                  _ctx->cam_w,
                                                  _ctx->cam_h,
                                                  cudaMemcpyDeviceToDevice,
                                                  nvjpeg_stream));

                nv_image.channel[i] = d_channel_begin;
                nv_image.pitch[i] = nvjpeg_rgb_pitch;
            };

            // overlay keypoints on grayscale or color image
            overlay_keypoints(nv_image.channel[1],
                              nv_image.pitch[1],
                              d_pos_left,
                              d_score_left,
                              _ctx->min_score,
                              _d_ir_intristics_left, // ir intrinsics
                              keypoints_num,
                              nvjpeg_stream);

            // Compress image
            CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
                                           &nv_image, NVJPEG_INPUT_RGB, _ctx->cam_w, _ctx->cam_h, nvjpeg_stream));

            if (previous_frame1)
            {
                Eigen::Matrix4d T_w2c_prev_curr = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d T_c2w_prev_curr = Eigen::Matrix4d::Identity();

                if (previous_frame2)
                {

                    // std::cout << "previous_frame2->T_w2c" << std::endl
                    //           << previous_frame2->T_w2c << std::endl;
                    // std::cout << "previous_frame1->T_w2c" << std::endl
                    //           << previous_frame1->T_w2c << std::endl;

                    T_w2c_prev_curr = previous_frame1->T_w2c * previous_frame2->T_w2c.inverse() * previous_frame1->T_w2c;
                    // Eigen::Quaterniond q1(previous_frame1->T_w2c.rotation());
                }

                T_w2c_prev_curr.setIdentity();

                // std::cout << "T_w2c_prev_curr" << std::endl
                //           << T_w2c_prev_curr << std::endl;

                // match_keypoints(slam_frame,
                //                 previous_frame1,
                //                 2,
                //                 4,
                //                 T_w2c_prev_curr,
                //                 d_valid_keypoints_num,
                //                 d_keypoints_num_matched,
                //                 &h_keypoints_num_matched,
                //                 h_current_matched_points.get(),
                //                 h_previous_matched_points.get(),
                //                 _d_rgb_intrinsics,
                //                 stream_left);

                if (h_keypoints_num_matched > 2)
                {
                    Eigen::Map<Eigen::MatrixXd> current_points((double *)h_current_matched_points.get(), 3, h_keypoints_num_matched);
                    Eigen::Map<Eigen::MatrixXd> previous_points((double *)h_previous_matched_points.get(), 3, h_keypoints_num_matched);

                    // Eigen::MatrixXd A = previous_points.transpose();
                    // Eigen::MatrixXd B = current_points.transpose();

                    // std::cout << "Matching points" << std::endl;
                    // for (int i = 0; i < h_keypoints_num_matched; i++)
                    // {
                    //     std::cout << A.block<1, 3>(i, 0) << "\t\t" << B.block<1, 3>(i, 0) << std::endl;
                    // }

                    // T_c2w_prev_curr = best_fit_transform(previous_points.transpose(), current_points.transpose());
                }
                else
                {
                    T_w2c_prev_curr.setIdentity();
                    // slam_frame->T_c2w = Eigen::Matrix4d::Identity();
                    // slam_frame->T_w2c = Eigen::Matrix4d::Identity();
                }

                // slam_frame->T_c2w = T_c2w_prev_curr * previous_frame1->T_c2w;
                // slam_frame->T_w2c = slam_frame->T_c2w.inverse();
                slam_frame->T_c2w = Eigen::Matrix4d::Identity();
                slam_frame->T_w2c = Eigen::Matrix4d::Identity();

                // getting rotation angles
                Eigen::Matrix3d R = T_c2w_prev_curr.block<3, 3>(0, 0);
                Vector3d eu_angles = R.eulerAngles(0, 1, 2);
                Vector3d angles(floor(eu_angles(0) * 180 / CUDART_PI_D),
                                floor(eu_angles(1) * 180 / CUDART_PI_D),
                                floor(eu_angles(2) * 180 / CUDART_PI_D));

                // std::cout << std::endl
                //           << "eu_angles" << std::endl
                //           << eu_angles << std::endl
                //           << "angles" << std::endl
                //           << angles << std::endl
                //           << "T_c2w_prev_curr" << std::endl
                //           << T_c2w_prev_curr << std::endl
                //           << "slam_frame->T_c2w" << std::endl
                //           << slam_frame->T_c2w << std::endl;
            }
            else
            {
                slam_frame->T_c2w = Eigen::Matrix4d::Identity();
                slam_frame->T_w2c = Eigen::Matrix4d::Identity();
                h_keypoints_num_matched = keypoints_num;
            }
            // -------------------------------------------------
            // TODO: convert cudaStreamSynchronize to use events
            // -------------------------------------------------

            // get compressed stream size
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, nvjpeg_stream));
            checkCudaErrors(cudaStreamSynchronize(nvjpeg_stream));
            // get stream itself
            slam_frame->image = (unsigned char *)malloc(length * sizeof(char));
            slam_frame->image_length = length;
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, (slam_frame->image), &length, nvjpeg_stream));
            checkCudaErrors(cudaStreamSynchronize(nvjpeg_stream));
            checkCudaErrors(cudaStreamSynchronize(stream_left));
            checkCudaErrors(cudaStreamSynchronize(stream_right));

            stop = high_resolution_clock::now();
            auto rotation_timer = high_resolution_clock::now();

            // sleep(2);
            // std::cout << "cudaStreamSynchronize" << std::endl;
            // checkCudaErrors(cudaStreamSynchronize(stream));
            // std::cout << "cudaStreamSynchronize passed" << std::endl;

            auto duration = duration_cast<microseconds>(stop - start);
            auto duration_rotation = duration_cast<microseconds>(rotation_timer - stop);

            // slam_frame->image = image;
            slam_frame->keypoints_count = keypoints_num;
            slam_frame->h_matched_keypoints_num = h_matched_keypoints_num;
            slam_frame->h_points = h_points;
            slam_frame->h_valid_keypoints_num = keypoints_num;

            pEvent newEvent = std::make_shared<BaseEvent>();
            newEvent->event_type = EventType::event_gpu_slam_frame;
            newEvent->message = slam_frame;
            if (previous_frame2)
                _ctx->sendEvent(newEvent);

            slam_frames[slam_frames_id]->image_ready_for_process = false;

            std::lock_guard<std::mutex> lock(m_gpu_mutex);
            deleted_slam_frames.push_back(slam_frames_id);

            if (previous_frame1)
                previous_frame2 = previous_frame1;
            previous_frame1 = slam_frame;
            h_valid_keypoints_num_previous = keypoints_num;

            std::cout << "Finished work GPU thread " << slam_frames_id
                      << " duration " << duration.count()
                      << " duration_rotation " << duration_rotation.count()
                      << " keypoints_num " << keypoints_num
                      << " h_matched_keypoints_num " << h_matched_keypoints_num
                      //   << " theta.x " << theta.x
                      //   << " theta.y " << theta.y
                      //   << " theta.z " << theta.z
                      //   << " keypoints_num " << keypoints_num
                      << std::endl;
        }

        // checkCudaErrors(cudaFree(d_rgb_image));
        checkCudaErrors(cudaFree(d_gray_image));
        // checkCudaErrors(cudaFree(d_aligned_out));
        // checkCudaErrors(cudaFree(d_depth_in));
        // checkCudaErrors(cudaFree(d_keypoints_exist));
        // checkCudaErrors(cudaFree(d_keypoints_response));
        // printf("---mark---\n");
        checkCudaErrors(cudaFree(d_keypoints_angle_left));
        checkCudaErrors(cudaFree(d_descriptors_tmp_left));
        checkCudaErrors(cudaFree(d_descriptors_left));

        checkCudaErrors(cudaFree(d_keypoints_angle_right));
        checkCudaErrors(cudaFree(d_descriptors_tmp_right));
        checkCudaErrors(cudaFree(d_descriptors_right));

        std::cout << "Stopped GPU thread " << slam_frames_id << std::endl;
    }

}