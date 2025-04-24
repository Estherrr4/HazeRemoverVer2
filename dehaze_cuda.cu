#include "dehaze.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <stdio.h> // check CUDA

//Check CUDA Part
// Define a simple kernel
__global__ void testKernel() {
    printf("Inside CUDA kernel!\n");
}

// Define a function to launch the kernel
extern "C" void launchCudaPart() {
    printf("Launching CUDA kernel from launchCudaPart()...\n");
    testKernel << <1, 1 >> > ();
    cudaDeviceSynchronize(); // Ensure it finishes before returning
}
//Check CUDA Part End


// Define CUDA_ENABLED for isCudaAvailable() function
#define CUDA_ENABLED

#define CUDA_CALL(kernel, grid, block, ...) kernel<<<grid, block>>>(__VA_ARGS__)

// Error checking macro
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, const char* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " << file << ":" << line << " '" << func << "'\n";
        // Don't exit - just return to fallback on CPU
    }
}

// Kernel to compute dark channel
__global__ void darkChannelKernel(const unsigned char* imgData, float* darkChannel, int width, int height, int channels, int patch_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float minVal = 1.0f;
    int idx = y * width + x;

    // Search through patch around current pixel
    for (int dy = -patch_radius; dy <= patch_radius; dy++) {
        for (int dx = -patch_radius; dx <= patch_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = (ny * width + nx) * channels;

                // Find minimum across RGB channels
                float b = imgData[nidx] / 255.0f;
                float g = imgData[nidx + 1] / 255.0f;
                float r = imgData[nidx + 2] / 255.0f;

                float pixelMin = fminf(r, fminf(g, b));
                minVal = fminf(minVal, pixelMin);
            }
        }
    }

    darkChannel[idx] = minVal;
}

// Kernel to compute normalized dark channel for transmission
__global__ void transmissionKernel(const unsigned char* imgData, const float* atmospheric, float* transmission, int width, int height, int channels, int patch_radius, float omega) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float minVal = 1.0f;
    int idx = y * width + x;

    // Search through patch around current pixel
    for (int dy = -patch_radius; dy <= patch_radius; dy++) {
        for (int dx = -patch_radius; dx <= patch_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = (ny * width + nx) * channels;

                // Normalize by atmospheric light and find minimum
                float b = imgData[nidx] / 255.0f / atmospheric[0];
                float g = imgData[nidx + 1] / 255.0f / atmospheric[1];
                float r = imgData[nidx + 2] / 255.0f / atmospheric[2];

                float pixelMin = fminf(r, fminf(g, b));
                minVal = fminf(minVal, pixelMin);
            }
        }
    }

    // Calculate transmission
    transmission[idx] = 1.0f - omega * minVal;
}

// Kernel for scene reconstruction
__global__ void sceneRecoveryKernel(const unsigned char* imgData, const float* transmission, const float* atmospheric, unsigned char* outputData, int width, int height, int channels, float t0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    int tidx = y * width + x;

    // Get transmission value with lower bound t0
    float t = fmaxf(transmission[tidx], t0);

    // Process each channel
    for (int c = 0; c < 3; c++) {
        float normalized = imgData[idx + c] / 255.0f;
        float recovered = (normalized - atmospheric[c]) / t + atmospheric[c];

        // Clamp to [0, 1] range
        recovered = fminf(fmaxf(recovered, 0.0f), 1.0f);

        // Convert back to 8-bit
        outputData[idx + c] = static_cast<unsigned char>(recovered * 255.0f);
    }
}

namespace DarkChannel {
    extern TimingInfo lastTimingInfo;

    cv::Mat dehaze_cuda(const cv::Mat& img) {
        try {
            // Reset timing info
            TimingInfo timingInfo;

            // Start total time measurement
            auto totalStartTime = std::chrono::high_resolution_clock::now();

            // Check if CUDA device is available
            int deviceCount = 0;
            cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

            if (cudaStatus != cudaSuccess || deviceCount == 0) {
                std::cerr << "No CUDA-capable device found. Falling back to CPU implementation." << std::endl;
                return dehaze(img);
            }

            // Get device properties
            cudaDeviceProp deviceProp;
            checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
            std::cout << "Using CUDA device: " << deviceProp.name << std::endl;

            // Check if image is valid
            if (img.empty() || img.type() != CV_8UC3) {
                std::cerr << "Error: Input image is empty or not 3-channel 8-bit" << std::endl;
                return img;
            }

            // Image dimensions
            int width = img.cols;
            int height = img.rows;
            int channels = img.channels();
            int imgSize = width * height;

            // Allocate host and device memory
            unsigned char* d_img = nullptr;
            float* d_darkChannel = nullptr;
            float* d_transmission = nullptr;
            unsigned char* d_result = nullptr;

            // Start dark channel calculation timing
            auto darkChannelStartTime = std::chrono::high_resolution_clock::now();

            // Allocate device memory
            checkCudaErrors(cudaMalloc(&d_img, imgSize * channels));
            checkCudaErrors(cudaMalloc(&d_darkChannel, imgSize * sizeof(float)));

            // Copy input image to device
            checkCudaErrors(cudaMemcpy(d_img, img.data, imgSize * channels, cudaMemcpyHostToDevice));

            // Define block and grid dimensions
            dim3 blockSize(16, 16);
            dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

            // Dark channel calculation
            int patchRadius = 7;
            CUDA_CALL(darkChannelKernel, gridSize, blockSize, d_img, d_darkChannel, width, height, channels, patchRadius);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // Copy dark channel back to host for atmospheric light estimation
            float* h_darkChannel = new float[imgSize];
            checkCudaErrors(cudaMemcpy(h_darkChannel, d_darkChannel, imgSize * sizeof(float), cudaMemcpyDeviceToHost));

            // End dark channel timing
            auto darkChannelEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.darkChannelTime = std::chrono::duration<double, std::milli>(darkChannelEndTime - darkChannelStartTime).count();

            // Start atmospheric light estimation timing
            auto atmosphericStartTime = std::chrono::high_resolution_clock::now();

            // Find atmospheric light (top 0.1% brightest pixels in dark channel)
            std::vector<std::pair<float, int>> brightnessIndices;
            brightnessIndices.reserve(imgSize);

            for (int i = 0; i < imgSize; i++) {
                brightnessIndices.push_back(std::make_pair(h_darkChannel[i], i));
            }

            std::sort(brightnessIndices.begin(), brightnessIndices.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first > b.first;
                });

            // Take top 0.1% of pixels
            int numBrightestPixels = std::max(1, imgSize / 1000);
            float atmospheric[3] = { 0, 0, 0 };

            for (int i = 0; i < numBrightestPixels; i++) {
                int idx = brightnessIndices[i].second;
                int y = idx / width;
                int x = idx % width;

                cv::Vec3b pixelValue = img.at<cv::Vec3b>(y, x);
                atmospheric[0] += pixelValue[0] / 255.0f;  // B
                atmospheric[1] += pixelValue[1] / 255.0f;  // G
                atmospheric[2] += pixelValue[2] / 255.0f;  // R
            }

            atmospheric[0] /= numBrightestPixels;
            atmospheric[1] /= numBrightestPixels;
            atmospheric[2] /= numBrightestPixels;

            // Avoid division by zero
            for (int i = 0; i < 3; i++) {
                if (atmospheric[i] < 0.00001f) {
                    atmospheric[i] = 0.00001f;
                }
            }

            // End atmospheric light timing
            auto atmosphericEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.atmosphericLightTime = std::chrono::duration<double, std::milli>(atmosphericEndTime - atmosphericStartTime).count();

            // Start transmission estimation timing
            auto transmissionStartTime = std::chrono::high_resolution_clock::now();

            // Allocate memory for transmission map
            checkCudaErrors(cudaMalloc(&d_transmission, imgSize * sizeof(float)));

            // Copy atmospheric light to device
            float* d_atmospheric;
            checkCudaErrors(cudaMalloc(&d_atmospheric, 3 * sizeof(float)));
            checkCudaErrors(cudaMemcpy(d_atmospheric, atmospheric, 3 * sizeof(float), cudaMemcpyHostToDevice));

            // Estimate transmission using dark channel prior
            float omega = 0.95f;  // Preserves some haze for depth perception
            CUDA_CALL(transmissionKernel, gridSize, blockSize, d_img, d_atmospheric, d_transmission, width, height, channels, patchRadius, omega);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // End transmission timing
            auto transmissionEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.transmissionTime = std::chrono::duration<double, std::milli>(transmissionEndTime - transmissionStartTime).count();

            // Start refinement timing
            auto refinementStartTime = std::chrono::high_resolution_clock::now();

            // For simplicity, we'll skip guided filter refinement in the CUDA version
            // In a real implementation, you would implement the guided filter as CUDA kernels

            // End refinement timing
            auto refinementEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.refinementTime = std::chrono::duration<double, std::milli>(refinementEndTime - refinementStartTime).count();

            // Start scene reconstruction timing
            auto reconstructionStartTime = std::chrono::high_resolution_clock::now();

            // Allocate memory for result
            checkCudaErrors(cudaMalloc(&d_result, imgSize * channels));

            // Recover scene
            float t0 = 0.1f;  // Minimum transmission value
            CUDA_CALL(sceneRecoveryKernel, gridSize, blockSize, d_img, d_transmission, d_atmospheric, d_result, width, height, channels, t0);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // Copy result back to host
            cv::Mat result(height, width, CV_8UC3);
            checkCudaErrors(cudaMemcpy(result.data, d_result, imgSize * channels, cudaMemcpyDeviceToHost));

            // End scene reconstruction timing
            auto reconstructionEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.reconstructionTime = std::chrono::duration<double, std::milli>(reconstructionEndTime - reconstructionStartTime).count();

            // End total time measurement
            auto totalEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.totalTime = std::chrono::duration<double, std::milli>(totalEndTime - totalStartTime).count();

            // Output timing information
            std::cout << "\n===== CUDA Performance Timing (milliseconds) =====" << std::endl;
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

            // Cleanup
            cudaFree(d_img);
            cudaFree(d_darkChannel);
            cudaFree(d_transmission);
            cudaFree(d_atmospheric);
            cudaFree(d_result);
            delete[] h_darkChannel;

            return result;
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
            std::cerr << "Unknown exception in dehaze_cuda function" << std::endl;
            return img;
        }
    }
}