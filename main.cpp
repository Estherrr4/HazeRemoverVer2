#include "dehaze.h"
#include "dehaze_parallel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <omp.h>

#ifdef _WIN32
#include <Windows.h>
#include <commdlg.h>
#include <conio.h> // For _getch() on Windows
#endif

inline int get_max(int a, int b) {
    return (a > b) ? a : b;
}

// Simple function to check if a file exists
bool fileExists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

// Function to wait for a key press without blocking the return to menu
void waitForKeyPress() {
    std::cout << "\nPress any key to return to menu..." << std::endl;

#ifdef _WIN32
    _getch(); // Use _getch() on Windows which doesn't require Enter key
#else
    // For non-Windows platforms
    getchar();
#endif
}

// Function to show file open dialog (ANSI version)
std::string getImageFile() {
#ifdef _WIN32
    OPENFILENAMEA ofn;
    char fileName[MAX_PATH] = "";
    ZeroMemory(&ofn, sizeof(ofn));

    ofn.lStructSize = sizeof(OPENFILENAMEA);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Image Files (*.jpg;*.jpeg;*.png;*.bmp)\0*.jpg;*.jpeg;*.png;*.bmp\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
    ofn.lpstrDefExt = "jpg";

    if (GetOpenFileNameA(&ofn)) {
        return std::string(fileName);
    }
    return "";
#else
    // For non-Windows platforms, use simple input
    std::string filePath;
    std::cout << "Enter path to image file: ";
    std::getline(std::cin, filePath);
    return filePath;
#endif
}

// Function to resize large images while maintaining aspect ratio
cv::Mat resizeImageIfTooLarge(const cv::Mat& img, int maxDimension = 1200) {
    // Check if the image is too large
    if (img.cols > maxDimension || img.rows > maxDimension) {
        cv::Mat resized;
        double scale = (double)maxDimension / (img.cols > img.rows ? img.cols : img.rows);
        int newWidth = static_cast<int>(img.cols * scale);
        int newHeight = static_cast<int>(img.rows * scale);

        cv::resize(img, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
        std::cout << "Image resized from " << img.cols << "x" << img.rows
            << " to " << resized.cols << "x" << resized.rows << std::endl;
        return resized;
    }
    return img;
}

// Function to process a single image
bool processImage(const std::string& img_path, const std::string& project_path) {
    try {
        // Use forward slashes consistently
        std::string normalized_path = img_path;
        std::replace(normalized_path.begin(), normalized_path.end(), '\\', '/');

        std::cout << "Processing image: " << normalized_path << std::endl;

        // Load image
        cv::Mat original_img = cv::imread(normalized_path);
        if (original_img.empty()) {
            std::cerr << "Error: Failed to load image: " << normalized_path << std::endl;
            waitForKeyPress();
            return false;
        }

        // Extract just the filename for the output file
        size_t pos = normalized_path.find_last_of('/');
        std::string img_name = (pos != std::string::npos) ? normalized_path.substr(pos + 1) : normalized_path;

        std::cout << "Image loaded successfully. Original size: " << original_img.cols << "x" << original_img.rows << std::endl;

        // Resize large images for faster processing
        cv::Mat img = resizeImageIfTooLarge(original_img, 1200);

        // Save original and resized version of input image for reference
        std::string original_path = project_path + "original_" + img_name;
        cv::imwrite(original_path, original_img);

        if (img.data != original_img.data) {
            std::string resized_input_path = project_path + "resized_input_" + img_name;
            cv::imwrite(resized_input_path, img);
            std::cout << "Original image saved as: " << original_path << std::endl;
            std::cout << "Resized input saved as: " << resized_input_path << std::endl;
        }

        // Process with serial version
        std::cout << "\nRunning serial version..." << std::endl;
        cv::Mat serial_result = DarkChannel::dehaze(img);
        DarkChannel::TimingInfo serial_timing = DarkChannel::getLastTimingInfo();

        // Process with OpenMP version
        std::cout << "\nRunning OpenMP version..." << std::endl;
        int num_threads = get_max(1, omp_get_max_threads());
        std::cout << "Using " << num_threads << " OpenMP threads" << std::endl;
        cv::Mat openmp_result = DarkChannel::dehazeParallel(img, num_threads);
        DarkChannel::TimingInfo openmp_timing = DarkChannel::getLastTimingInfo();

        // Check CUDA availability
        bool cuda_available = DarkChannel::isCudaAvailable();
        cv::Mat cuda_result;
        DarkChannel::TimingInfo cuda_timing;

        if (cuda_available) {
            // Run CUDA implementation if available
            std::cout << "\nRunning CUDA version..." << std::endl;
            try {
                cuda_result = DarkChannel::dehaze_cuda(img);
                cuda_timing = DarkChannel::getLastTimingInfo();

                // Verify the CUDA result is valid
                if (cuda_result.empty()) {
                    std::cerr << "Warning: CUDA result is empty" << std::endl;
                    cuda_available = false;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA processing failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation." << std::endl;
                cuda_available = false;
            }
        }
        else {
            std::cout << "\nCUDA implementation not available." << std::endl;
        }

        // Create performance comparison table
        std::cout << "\n===== DEHAZING PERFORMANCE COMPARISON (milliseconds) =====" << std::endl;
        std::cout << "Stage                       Serial          OpenMP";
        if (cuda_available) std::cout << "          CUDA";
        std::cout << "           OMP Speedup";
        if (cuda_available) std::cout << "     CUDA Speedup";
        std::cout << std::endl;

        std::cout << "------------------------------------------------------------------------------------------------" << std::endl;

        // Compare dark channel calculation
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Dark Channel Calculation    " << std::setw(15) << serial_timing.darkChannelTime;
        std::cout << std::setw(15) << openmp_timing.darkChannelTime;

        if (cuda_available) {
            std::cout << std::setw(15) << cuda_timing.darkChannelTime;
        }

        double omp_speedup = (serial_timing.darkChannelTime > 0) ?
            (serial_timing.darkChannelTime / openmp_timing.darkChannelTime) : 0.0;
        std::cout << std::setw(15) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.darkChannelTime > 0) ?
                (serial_timing.darkChannelTime / cuda_timing.darkChannelTime) : 0.0;
            std::cout << std::setw(15) << cuda_speedup;
        }
        std::cout << std::endl;

        // Compare atmospheric light estimation
        std::cout << "Atmospheric Light          " << std::setw(15) << serial_timing.atmosphericLightTime;
        std::cout << std::setw(15) << openmp_timing.atmosphericLightTime;

        if (cuda_available) {
            std::cout << std::setw(15) << cuda_timing.atmosphericLightTime;
        }

        omp_speedup = (serial_timing.atmosphericLightTime > 0) ?
            (serial_timing.atmosphericLightTime / openmp_timing.atmosphericLightTime) : 0.0;
        std::cout << std::setw(15) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.atmosphericLightTime > 0) ?
                (serial_timing.atmosphericLightTime / cuda_timing.atmosphericLightTime) : 0.0;
            std::cout << std::setw(15) << cuda_speedup;
        }
        std::cout << std::endl;

        // Compare transmission estimation
        std::cout << "Transmission Estimation    " << std::setw(15) << serial_timing.transmissionTime;
        std::cout << std::setw(15) << openmp_timing.transmissionTime;

        if (cuda_available) {
            std::cout << std::setw(15) << cuda_timing.transmissionTime;
        }

        omp_speedup = (serial_timing.transmissionTime > 0) ?
            (serial_timing.transmissionTime / openmp_timing.transmissionTime) : 0.0;
        std::cout << std::setw(15) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.transmissionTime > 0) ?
                (serial_timing.transmissionTime / cuda_timing.transmissionTime) : 0.0;
            std::cout << std::setw(15) << cuda_speedup;
        }
        std::cout << std::endl;

        // Compare transmission refinement
        std::cout << "Transmission Refinement    " << std::setw(15) << serial_timing.refinementTime;
        std::cout << std::setw(15) << openmp_timing.refinementTime;

        if (cuda_available) {
            std::cout << std::setw(15) << cuda_timing.refinementTime;
        }

        omp_speedup = (serial_timing.refinementTime > 0) ?
            (serial_timing.refinementTime / openmp_timing.refinementTime) : 0.0;
        std::cout << std::setw(15) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.refinementTime > 0) ?
                (serial_timing.refinementTime / cuda_timing.refinementTime) : 0.0;
            std::cout << std::setw(15) << cuda_speedup;
        }
        std::cout << std::endl;

        // Compare scene reconstruction
        std::cout << "Scene Reconstruction       " << std::setw(15) << serial_timing.reconstructionTime;
        std::cout << std::setw(15) << openmp_timing.reconstructionTime;

        if (cuda_available) {
            std::cout << std::setw(15) << cuda_timing.reconstructionTime;
        }

        omp_speedup = (serial_timing.reconstructionTime > 0) ?
            (serial_timing.reconstructionTime / openmp_timing.reconstructionTime) : 0.0;
        std::cout << std::setw(15) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.reconstructionTime > 0) ?
                (serial_timing.reconstructionTime / cuda_timing.reconstructionTime) : 0.0;
            std::cout << std::setw(15) << cuda_speedup;
        }
        std::cout << std::endl;

        // Compare total execution time
        std::cout << "Total Execution Time       " << std::setw(15) << serial_timing.totalTime;
        std::cout << std::setw(15) << openmp_timing.totalTime;

        if (cuda_available) {
            std::cout << std::setw(15) << cuda_timing.totalTime;
        }

        omp_speedup = (serial_timing.totalTime > 0) ?
            (serial_timing.totalTime / openmp_timing.totalTime) : 0.0;
        std::cout << std::setw(15) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.totalTime > 0) ?
                (serial_timing.totalTime / cuda_timing.totalTime) : 0.0;
            std::cout << std::setw(15) << cuda_speedup;
        }
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;

        // Save results
        std::string serial_output_path = project_path + "serial_dehazed_" + img_name;
        std::string openmp_output_path = project_path + "openmp_dehazed_" + img_name;
        std::string cuda_output_path;

        cv::imwrite(serial_output_path, serial_result);
        cv::imwrite(openmp_output_path, openmp_result);

        if (cuda_available) {
            cuda_output_path = project_path + "cuda_dehazed_" + img_name;
            cv::imwrite(cuda_output_path, cuda_result);
        }

        std::cout << "Results saved as: " << std::endl;
        std::cout << "  Serial: " << serial_output_path << std::endl;
        std::cout << "  OpenMP: " << openmp_output_path << std::endl;
        if (cuda_available) {
            std::cout << "  CUDA: " << cuda_output_path << std::endl;
        }

        // Create a smaller version for display if the image is very large
        cv::Mat display_img = img;
        cv::Mat display_serial = serial_result;
        cv::Mat display_openmp = openmp_result;
        cv::Mat display_cuda;

        // Resize large images for display
        int maxDisplayDimension = 800;
        if (img.cols > maxDisplayDimension || img.rows > maxDisplayDimension) {
            double scale = (double)maxDisplayDimension / (img.cols > img.rows ? img.cols : img.rows);
            int newWidth = static_cast<int>(img.cols * scale);
            int newHeight = static_cast<int>(img.rows * scale);

            cv::resize(img, display_img, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
            cv::resize(serial_result, display_serial, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
            cv::resize(openmp_result, display_openmp, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);

            if (cuda_available) {
                cv::resize(cuda_result, display_cuda, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
            }

            std::cout << "Images resized for display to " << newWidth << "x" << newHeight << std::endl;
        }
        else if (cuda_available) {
            display_cuda = cuda_result;
        }

        // Create named windows with the WINDOW_NORMAL flag to allow resizing
        cv::namedWindow("Original", cv::WINDOW_NORMAL);
        cv::namedWindow("Serial Result", cv::WINDOW_NORMAL);
        cv::namedWindow("OpenMP Result", cv::WINDOW_NORMAL);

        // Resize windows for better viewing - based on the display image size
        int windowWidth = display_img.cols;
        int windowHeight = display_img.rows;

        cv::resizeWindow("Original", windowWidth, windowHeight);
        cv::resizeWindow("Serial Result", windowWidth, windowHeight);
        cv::resizeWindow("OpenMP Result", windowWidth, windowHeight);

        if (cuda_available) {
            cv::namedWindow("CUDA Result", cv::WINDOW_NORMAL);
            cv::resizeWindow("CUDA Result", windowWidth, windowHeight);
        }

        // Calculate window positions to avoid overlap and ensure visibility
        int screenWidth = 1920;  // Assuming a standard screen width
        int windowSpacing = 20;
        int windowsPerRow = 2;
        int windowX, windowY;

        // Position windows in a grid pattern
        windowX = 50;
        windowY = 50;
        cv::moveWindow("Original", windowX, windowY);

        windowX = 50 + windowWidth + windowSpacing;
        cv::moveWindow("Serial Result", windowX, windowY);

        windowX = 50;
        windowY = 50 + windowHeight + windowSpacing;
        cv::moveWindow("OpenMP Result", windowX, windowY);

        if (cuda_available) {
            windowX = 50 + windowWidth + windowSpacing;
            windowY = 50 + windowHeight + windowSpacing;
            cv::moveWindow("CUDA Result", windowX, windowY);
        }

        // Display the results
        cv::imshow("Original", display_img);
        cv::imshow("Serial Result", display_serial);
        cv::imshow("OpenMP Result", display_openmp);

        if (cuda_available) {
            std::cout << "Displaying CUDA result (size: " << display_cuda.cols << "x" << display_cuda.rows << ")" << std::endl;
            cv::imshow("CUDA Result", display_cuda);
            cv::waitKey(1); // Force update
        }

        // First wait for key in OpenCV window (this won't block the return to menu)
        std::cout << "\nViewing results. Press any key in the image window to continue..." << std::endl;
        cv::waitKey(0); // wait for a key press in the OpenCV window

        // Close the windows to prevent stacking up multiple windows
        cv::destroyWindow("Original");
        cv::destroyWindow("Serial Result");
        cv::destroyWindow("OpenMP Result");
        if (cuda_available) {
            cv::destroyWindow("CUDA Result");
        }
        cv::destroyAllWindows(); // Make sure all windows are closed

        // Then wait for key in console to ensure we return to menu
        waitForKeyPress();

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in processImage: " << e.what() << std::endl;
        waitForKeyPress();
        return false;
    }
}

// Declare the CUDA test function
extern "C" void launchCudaPart();

int main(int argc, char** argv) {
    try {
        // Test CUDA functionality at startup
        std::cout << "Testing CUDA functionality..." << std::endl;
        try {
            launchCudaPart();
            std::cout << "CUDA test completed." << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "CUDA test failed: " << e.what() << std::endl;
            std::cerr << "Some features may be unavailable." << std::endl;
        }

        std::string project_path = "./";

        // Check for command-line arguments
        bool interactive_mode = true;
        std::string img_path;

        if (argc > 1) {
            // Process the specified image from command line
            interactive_mode = false;
            std::string arg_path = argv[1];

            // Check if the argument is a relative path
            if (arg_path.find(':') == std::string::npos) {
                // It's a relative path, combine with project path
                img_path = project_path + arg_path;
            }
            else {
                // It's an absolute path, use it directly
                img_path = arg_path;
            }

            processImage(img_path, project_path);

            // After processing command line image, enter interactive mode
            interactive_mode = true;
        }

        // Interactive mode loop
        if (interactive_mode) {
            int choice = -1;
            while (choice != 0) {
                std::cout << "\n=== Dehaze Interactive Mode ===" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  1: Enter image path manually" << std::endl;
                std::cout << "  2: Use file dialog to select image" << std::endl;
                std::cout << "  3: Process multiple images" << std::endl;
                std::cout << "  4: Check CUDA availability" << std::endl;
                std::cout << "  0: Exit" << std::endl;

                std::cout << "\nEnter your choice (0-4): ";
                std::cin >> choice;
                std::cin.ignore(); // Clear the newline

                switch (choice) {
                case 0:
                    std::cout << "Exiting program." << std::endl;
                    break;

                case 1: {
                    // Manual path entry
                    std::cout << "Enter image path (or 'exit' to return to menu): ";
                    std::getline(std::cin, img_path);
                    if (img_path == "exit" || img_path == "quit") {
                        break;
                    }
                    processImage(img_path, project_path);
                    break;
                }

                case 2: {
                    // File dialog
                    img_path = getImageFile();
                    if (!img_path.empty()) {
                        processImage(img_path, project_path);
                    }
                    else {
                        std::cout << "No file selected." << std::endl;
                        waitForKeyPress();
                    }
                    break;
                }

                case 3: {
                    // Process multiple images
                    std::cout << "Enter 'exit' at any time to return to menu." << std::endl;
                    while (true) {
                        std::cout << "\nSelect option for next image:" << std::endl;
                        std::cout << "  1: Enter path manually" << std::endl;
                        std::cout << "  2: Use file dialog" << std::endl;
                        std::cout << "  0: Return to main menu" << std::endl;

                        int subChoice;
                        std::cout << "Choice: ";
                        std::cin >> subChoice;
                        std::cin.ignore(); // Clear the newline

                        if (subChoice == 0) {
                            break;
                        }
                        else if (subChoice == 1) {
                            std::cout << "Enter image path: ";
                            std::getline(std::cin, img_path);
                            if (img_path == "exit" || img_path == "quit") {
                                break;
                            }
                            processImage(img_path, project_path);
                        }
                        else if (subChoice == 2) {
                            img_path = getImageFile();
                            if (img_path.empty()) {
                                std::cout << "No file selected." << std::endl;
                                waitForKeyPress();
                                continue;
                            }
                            processImage(img_path, project_path);
                        }
                        else {
                            std::cout << "Invalid choice." << std::endl;
                            waitForKeyPress();
                            continue;
                        }
                    }
                    break;
                }

                case 4: {
                    // Check CUDA availability details
                    std::cout << "Checking CUDA availability..." << std::endl;
                    bool cuda_available = DarkChannel::isCudaAvailable();

                    if (cuda_available) {
                        std::cout << "CUDA is available and working correctly." << std::endl;

                        // Get more detailed CUDA info
                        int deviceCount = 0;
                        cudaGetDeviceCount(&deviceCount);

                        std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

                        for (int i = 0; i < deviceCount; i++) {
                            cudaDeviceProp deviceProp;
                            cudaGetDeviceProperties(&deviceProp, i);

                            std::cout << "\nDevice " << i << ": " << deviceProp.name << std::endl;
                            std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
                            std::cout << "  Total global memory: " << (deviceProp.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
                            std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
                            std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
                            std::cout << "  Max threads dimensions: ("
                                << deviceProp.maxThreadsDim[0] << ", "
                                << deviceProp.maxThreadsDim[1] << ", "
                                << deviceProp.maxThreadsDim[2] << ")" << std::endl;
                            std::cout << "  Max grid dimensions: ("
                                << deviceProp.maxGridSize[0] << ", "
                                << deviceProp.maxGridSize[1] << ", "
                                << deviceProp.maxGridSize[2] << ")" << std::endl;
                        }
                    }
                    else {
                        std::cout << "CUDA is not available or not functioning correctly." << std::endl;
                    }

                    waitForKeyPress();
                    break;
                }

                default:
                    std::cout << "Invalid choice. Please try again." << std::endl;
                    waitForKeyPress();
                }
            }
        }

        cv::destroyAllWindows();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in main: " << e.what() << std::endl;
        waitForKeyPress();
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown exception in main" << std::endl;
        waitForKeyPress();
        return 1;
    }
}