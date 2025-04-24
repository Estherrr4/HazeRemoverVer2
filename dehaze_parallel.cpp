#include "dehaze_parallel.h"
#include "fastguidedfilter.h"
#include "dehaze.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <omp.h>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace DarkChannel {
    extern TimingInfo lastTimingInfo;

    cv::Mat dehazeParallel(const cv::Mat& img, int numThreads) {
        try {
            // Reset timing info
            TimingInfo timingInfo;

            // Start total time measurement
            auto totalStartTime = std::chrono::high_resolution_clock::now();

            // Set number of threads
            omp_set_num_threads(numThreads);

            std::cout << "\n===== OpenMP Performance Timing (" << numThreads << " threads) =====" << std::endl;

            // Check if image is valid
            if (img.empty()) {
                std::cerr << "Error: Input image is empty" << std::endl;
                return img;
            }

            // Check for correct image type
            if (img.type() != CV_8UC3) {
                std::cerr << "Error: Only 3-channel images are supported" << std::endl;
                return img;
            }

            // Convert to double precision for calculations
            cv::Mat img_double;
            img.convertTo(img_double, CV_64FC3);
            img_double /= 255;

            // Start timing for dark channel calculation
            auto darkChannelStartTime = std::chrono::high_resolution_clock::now();

            // Get dark channel
            int patch_radius = 7;
            cv::Mat darkchannel = cv::Mat::zeros(img_double.rows, img_double.cols, CV_64F);

            // Use OpenMP vectorization without thread parallelism
#pragma omp parallel for collapse(2)
            for (int i = 0; i < darkchannel.rows; i++) {
                for (int j = 0; j < darkchannel.cols; j++) {
                    int r_start = std::max(0, i - patch_radius);
                    int r_end = std::min(darkchannel.rows, i + patch_radius + 1);
                    int c_start = std::max(0, j - patch_radius);
                    int c_end = std::min(darkchannel.cols, j + patch_radius + 1);
                    double dark = 255;

                    for (int r = r_start; r < r_end; r++) {
                        for (int c = c_start; c < c_end; c++) {
                            const cv::Vec3d& val = img_double.at<cv::Vec3d>(r, c);
                            dark = std::min(dark, val[0]);
                            dark = std::min(dark, val[1]);
                            dark = std::min(dark, val[2]);
                        }
                    }

                    darkchannel.at<double>(i, j) = dark;
                }
            }

            // End timing for dark channel calculation
            auto darkChannelEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.darkChannelTime = std::chrono::duration<double, std::milli>(darkChannelEndTime - darkChannelStartTime).count();

            // Start timing for atmospheric light estimation
            auto atmosphericStartTime = std::chrono::high_resolution_clock::now();

            // Get atmospheric light
            int pixels = img_double.rows * img_double.cols;
            // Ensure at least 1 pixel is used
            int num = std::max(1, pixels / 1000); // 0.1%
            std::vector<std::pair<double, int>> V;
            V.reserve(pixels);
            int k = 0;

            // Use vectorization
            for (int i = 0; i < darkchannel.rows; i++) {
                for (int j = 0; j < darkchannel.cols; j++) {
                    V.emplace_back(darkchannel.at<double>(i, j), k);
                    k++;
                }
            }

            std::sort(V.begin(), V.end(), [](const std::pair<double, int>& p1, const std::pair<double, int>& p2) {
                return p1.first > p2.first;
                });

            double atmospheric_light[] = { 0, 0, 0 };

            // Vectorizable loop
            #pragma omp parallel for 
            for (k = 0; k < num; k++) {
                int r = V[k].second / darkchannel.cols;
                int c = V[k].second % darkchannel.cols;
                const cv::Vec3d& val = img_double.at<cv::Vec3d>(r, c);
                #pragma omp critical
                {
                atmospheric_light[0] += val[0];
                atmospheric_light[1] += val[1];
                atmospheric_light[2] += val[2];
                }
                
            }

            atmospheric_light[0] /= num;
            atmospheric_light[1] /= num;
            atmospheric_light[2] /= num;

            // Avoid division by zero
            for (int i = 0; i < 3; i++) {
                if (atmospheric_light[i] < 0.00001) {
                    atmospheric_light[i] = 0.00001;
                }
            }

            // End timing for atmospheric light estimation
            auto atmosphericEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.atmosphericLightTime = std::chrono::duration<double, std::milli>(atmosphericEndTime - atmosphericStartTime).count();

            // Start timing for transmission estimation
            auto transmissionStartTime = std::chrono::high_resolution_clock::now();

            double omega = 0.95;
            cv::Mat channels[3];
            cv::split(img_double, channels);

            // Vectorizable operations
#pragma omp parallel for
            for (k = 0; k < 3; k++) {
                channels[k] /= atmospheric_light[k];
            }

            cv::Mat temp;
            cv::merge(channels, 3, temp);

            // Get dark channel of normalized image
            cv::Mat temp_dark = cv::Mat::zeros(temp.rows, temp.cols, CV_64F);

            // Use vectorization
#pragma omp parallel for collapse(2)
            for (int i = 0; i < temp.rows; i++) {
                for (int j = 0; j < temp.cols; j++) {
                    int r_start = std::max(0, i - patch_radius);
                    int r_end = std::min(temp.rows, i + patch_radius + 1);
                    int c_start = std::max(0, j - patch_radius);
                    int c_end = std::min(temp.cols, j + patch_radius + 1);
                    double dark = 255;

                    for (int r = r_start; r < r_end; r++) {
                        for (int c = c_start; c < c_end; c++) {
                            const cv::Vec3d& val = temp.at<cv::Vec3d>(r, c);
                            dark = std::min(dark, val[0]);
                            dark = std::min(dark, val[1]);
                            dark = std::min(dark, val[2]);
                        }
                    }

                    temp_dark.at<double>(i, j) = dark;
                }
            }

            // Get transmission
            cv::Mat transmission = 1 - omega * temp_dark;

            // End timing for transmission estimation
            auto transmissionEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.transmissionTime = std::chrono::duration<double, std::milli>(transmissionEndTime - transmissionStartTime).count();

            // Start timing for refinement
            auto refinementStartTime = std::chrono::high_resolution_clock::now();

            // Apply guided filter for refinement
            transmission = fastGuidedFilter(img_double, transmission, 40, 0.1, 5);

            // End timing for refinement
            auto refinementEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.refinementTime = std::chrono::duration<double, std::milli>(refinementEndTime - refinementStartTime).count();

            // Start timing for scene reconstruction
            auto reconstructionStartTime = std::chrono::high_resolution_clock::now();

            // Ensure minimum transmission value to preserve details in dark regions
            double t0 = 0.1;
            cv::split(img_double, channels);
            cv::Mat trans_;
            cv::max(transmission, t0, trans_);

            cv::Mat temp_;

            // Vectorizable operations
            #pragma omp parallel for
            for (k = 0; k < 3; k++) {
                cv::divide((channels[k] - atmospheric_light[k]), trans_, temp_);
                channels[k] = atmospheric_light[k] + temp_;
            }

            // Merge channels and convert back to 8-bit
            cv::Mat res;
            cv::merge(channels, 3, res);

            // Normalize result
            double maxv;
            cv::minMaxLoc(res, nullptr, &maxv, nullptr, nullptr);
            if (maxv > 0) {
                res /= maxv;
            }
            res *= 255;
            res.convertTo(res, CV_8UC3);

            // End timing for scene reconstruction
            auto reconstructionEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.reconstructionTime = std::chrono::duration<double, std::milli>(reconstructionEndTime - reconstructionStartTime).count();

            // End total time measurement
            auto totalEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.totalTime = std::chrono::duration<double, std::milli>(totalEndTime - totalStartTime).count();

            // Output timing information
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Dark Channel Calculation: " << timingInfo.darkChannelTime << " ms" << std::endl;
            std::cout << "Atmospheric Light Estimation: " << timingInfo.atmosphericLightTime << " ms" << std::endl;
            std::cout << "Transmission Estimation: " << timingInfo.transmissionTime << " ms" << std::endl;
            std::cout << "Transmission Refinement: " << timingInfo.refinementTime << " ms" << std::endl;
            std::cout << "Scene Reconstruction: " << timingInfo.reconstructionTime << " ms" << std::endl;
            std::cout << "Total Execution Time: " << timingInfo.totalTime << " ms" << std::endl;
            std::cout << "========================================" << std::endl;

            // Save timing info in the global variable for access
            lastTimingInfo = timingInfo;

            return res;
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception: " << e.what() << std::endl;
            return img;
        }
        catch (const std::exception& e) {
            std::cerr << "Standard Exception: " << e.what() << std::endl;
            return img;
        }
        catch (...) {
            std::cerr << "Unknown exception in dehazeParallel function" << std::endl;
            return img;
        }
    }

    /*extern bool isCudaAvailable() {
        #ifdef CUDA_ENABLED
        try {
            int deviceCount = 0;
            cudaError_t error = cudaGetDeviceCount(&deviceCount);

            if (error != cudaSuccess) {
                std::cerr << "CUDA device check failed: " << cudaGetErrorString(error) << std::endl;
                return false;
            }

            if (deviceCount == 0) {
                std::cerr << "No CUDA devices found on this system." << std::endl;
                return false;
            }

            // Additional check: verify device capabilities
            cudaDeviceProp deviceProp;
            error = cudaGetDeviceProperties(&deviceProp, 0);

            if (error != cudaSuccess) {
                std::cerr << "Failed to get CUDA device properties: " << cudaGetErrorString(error) << std::endl;
                return false;
            }

            // Check for compute capability (minimum 3.0 required)
            if (deviceProp.major < 3) {
                std::cerr << "CUDA device has compute capability "
                    << deviceProp.major << "." << deviceProp.minor
                    << ", but 3.0 or higher is required." << std::endl;
                return false;
            }

            std::cout << "CUDA device available: " << deviceProp.name << " (Compute "
                << deviceProp.major << "." << deviceProp.minor << ")" << std::endl;

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception checking CUDA availability: " << e.what() << std::endl;
            return false;
        }
        #else
        std::cerr << "CUDA support not compiled into this build." << std::endl;
        return false;
        #endif
    }*/
}