#pragma once
#ifndef IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H
#define IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H

#include <opencv2/opencv.hpp>

namespace DarkChannel {
    // Error checking macro
#define __Assert__(x,msg)\
    do\
    {\
        if(!(x)){throw std::runtime_error((msg));}\
    }while(false)

    // Structure to store timing information
    struct TimingInfo {
        double darkChannelTime = 0.0;      // Time to compute dark channel
        double atmosphericLightTime = 0.0; // Time to estimate atmospheric light
        double transmissionTime = 0.0;     // Time to estimate transmission
        double refinementTime = 0.0;       // Time for guided filter refinement
        double reconstructionTime = 0.0;   // Time for final image reconstruction
        double totalTime = 0.0;            // Total processing time
    };

    // Serial implementation
    cv::Mat dehaze(const cv::Mat& img);

    // CUDA implementation
    cv::Mat dehaze_cuda(const cv::Mat& img);

    // Function to get the last timing information
    TimingInfo getLastTimingInfo();
}

#endif //IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H