#ifndef JETRACER_SLAM_GPU_PIPELINE_THREAD_H
#define JETRACER_SLAM_GPU_PIPELINE_THREAD_H

#include <iostream>

#include "../EventsThread.h"
#include "../Context.h"
#include "../Events/BaseEvent.h"
#include "../Events/EventTypes.h"
#include "types.h"
#include <mutex>
#include <atomic>
#include <thread>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <eigen3/Eigen/Eigen>
#include "../cuda/pyramid.cuh"

#include "../RealSense/RealSenseD400.h"

#define CUDA_WARP_SIZE 32
#define RS2_CUDA_THREADS_PER_BLOCK 32

#define USE_PRECALCULATED_INDICES 1
#define FAST_GPU_USE_LOOKUP_TABLE 1
#define FAST_GPU_USE_LOOKUP_TABLE_BITBASED 1

namespace Jetracer
{

    class SlamGpuPipeline : public EventsThread
    {
    public:
        SlamGpuPipeline(const std::string threadName, context_t *ctx);
        // ~SlamGpuPipeline();
        // void pushOverride(pEvent event); // callback events needs to be added no matter what

    private:
        void handleEvent(pEvent event);
        void buildStream(int slam_frames_id);
        void detect_keypoints();
        void process_gyro(rs2_vector gyro_data, double ts);
        void process_accel(rs2_vector accel_data);

        void upload_intrinsics(std::shared_ptr<Jetracer::rgbd_frame_t> rgbd_frame);

        context_t *_ctx;
        std::mutex m_mutex_subscribers;
        int streams_count = 0;
        std::shared_ptr<slam_frame_callback_t> *slam_frames; //shared pointer to auto free memory
        std::vector<int> deleted_slam_frames;
        bool include_anms = false;
        int fastThresh = 20;
        slam_frame_t keyframe;
        unsigned long long rgb_curr_frame_id = 0;
        unsigned long long depth_curr_frame_id = 0;
        unsigned long long rgb_prev_frame_id = 0;
        unsigned long long depth_prev_frame_id = 0;
        unsigned long long prev_ir_left_frame_id = 0;

        bool intristics_are_known = false;
        std::mutex m_gpu_mutex;

        // float depth_scale;
        // rs2_intrinsics *_d_rgb_intrinsics;
        // rs2_intrinsics *_d_depth_intrinsics;
        // rs2_extrinsics *_d_depth_rgb_extrinsics;

        rs2_intrinsics *_d_ir_intristics_left;
        rs2_intrinsics *_d_ir_intristics_right;

        int frame_counter = 0;

        // theta is the angle of camera rotation in x, y and z components
        float3 theta;
        /* alpha indicates the part that gyro and accelerometer take in computation of theta; higher alpha gives more weight to gyro, but too high
        values cause drift; lower alpha gives more weight to accelerometer, which is more sensitive to disturbances */
        float alpha = 0.98;
        bool firstGyro = true;
        bool firstAccel = true;
        // Keeps the arrival time of previous gyro frame
        double last_ts_gyro = 0;

        // allocate memory for image processing
        unsigned char *d_rgb_image_;
        unsigned char *d_gray_image_;
        unsigned char *d_nvjpeg_rgb_image_;

        std::size_t rgb_pitch_;
        std::size_t gray_pitch_;
        std::size_t nvjpeg_rgb_pitch_;

        cudaStream_t stream_left_;
        cudaStream_t stream_right_;
        cudaStream_t nvjpeg_stream_;

        std::vector<pyramid_t> pyramid_left_;
        std::vector<pyramid_t> pyramid_right_;

        float *d_keypoints_angle_left_;
        float *d_keypoints_angle_right_;

        unsigned char *d_descriptors_tmp_left_;
        unsigned char *d_descriptors_tmp_right_;
        int *d_valid_keypoints_num_left;
        int *d_matched_keypoints_num;
        int h_matched_keypoints_num = 0;

        float *d_score_left_;
        float *d_score_right_;
        int *d_level_left_;
        int *d_level_right_;

        cudaEvent_t event_ir_right_detected_;
    };
} // namespace Jetracer

#endif // JETRACER_SLAM_GPU_PIPELINE_THREAD_H
