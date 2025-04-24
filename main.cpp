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

#ifdef _WIN32
#include <Windows.h>
#include <commdlg.h>
#include <conio.h> // For _getch() on Windows
#endif

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

// Function to process a single image
bool processImage(const std::string& img_path, const std::string& project_path) {
    // Use forward slashes consistently
    std::string normalized_path = img_path;
    std::replace(normalized_path.begin(), normalized_path.end(), '\\', '/');

    std::cout << "Processing image: " << normalized_path << std::endl;

    // Load image
    cv::Mat img = cv::imread(normalized_path);
    if (img.empty()) {
        std::cerr << "Error: Failed to load image: " << normalized_path << std::endl;
        waitForKeyPress();
        return false;
    }

    // Extract just the filename for the output file
    size_t pos = normalized_path.find_last_of('/');
    std::string img_name = (pos != std::string::npos) ? normalized_path.substr(pos + 1) : normalized_path;

    std::cout << "Image loaded successfully. Size: " << img.cols << "x" << img.rows << std::endl;

    // Process with serial version
    cv::Mat serial_result = DarkChannel::dehaze(img);

    // Process with non-threaded OpenMP version (vectorization only)
    cv::Mat openmp_result = DarkChannel::dehazeParallel(img, 8); // Using 8 as thread count, but not actually used for threading

    // Process with CUDA if available
    cv::Mat cuda_result;
    bool cuda_available = DarkChannel::isCudaAvailable();
    /*bool DarkChannel::isCudaAvailable() {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        return (error == cudaSuccess && device_count > 0);
    }*/

    if (cuda_available) {
        // If CUDA is available, use CUDA implementation
        cuda_result = DarkChannel::dehaze_cuda(img);
    }
    else {
        std::cout << "CUDA implementation not available in this build." << std::endl;
    }

    // Create performance comparison table
    std::cout << "\n===== DEHAZING PERFORMANCE COMPARISON (milliseconds) =====" << std::endl;
    std::cout << "Stage                       Serial          OpenMP          CUDA           OMP Speedup     CUDA Speedup" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------------" << std::endl;

    // Get timing info
    DarkChannel::TimingInfo serial_timing = DarkChannel::getLastTimingInfo();

    // Compare dark channel calculation
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Dark Channel Calculation    " << std::setw(15) << serial_timing.darkChannelTime;
    std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().darkChannelTime;
    if (cuda_available) {
        std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().darkChannelTime;
        std::cout << std::setw(15) << (serial_timing.darkChannelTime / DarkChannel::getLastTimingInfo().darkChannelTime);
        std::cout << std::setw(15) << (serial_timing.darkChannelTime > 0 ? "inf" : "0.00");
    }
    else {
        std::cout << std::setw(15) << "0.00";
        std::cout << std::setw(15) << (serial_timing.darkChannelTime / DarkChannel::getLastTimingInfo().darkChannelTime);
        std::cout << std::setw(15) << "inf";
    }
    std::cout << std::endl;

    // Compare atmospheric light estimation
    std::cout << "Atmospheric Light          " << std::setw(15) << serial_timing.atmosphericLightTime;
    std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().atmosphericLightTime;
    if (cuda_available) {
        std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().atmosphericLightTime;
        std::cout << std::setw(15) << (serial_timing.atmosphericLightTime / DarkChannel::getLastTimingInfo().atmosphericLightTime);
        std::cout << std::setw(15) << (serial_timing.atmosphericLightTime > 0 ? "inf" : "0.00");
    }
    else {
        std::cout << std::setw(15) << "0.00";
        std::cout << std::setw(15) << (serial_timing.atmosphericLightTime / DarkChannel::getLastTimingInfo().atmosphericLightTime);
        std::cout << std::setw(15) << "inf";
    }
    std::cout << std::endl;

    // Compare transmission estimation
    std::cout << "Transmission Estimation    " << std::setw(15) << serial_timing.transmissionTime;
    std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().transmissionTime;
    if (cuda_available) {
        std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().transmissionTime;
        std::cout << std::setw(15) << (serial_timing.transmissionTime / DarkChannel::getLastTimingInfo().transmissionTime);
        std::cout << std::setw(15) << (serial_timing.transmissionTime > 0 ? "inf" : "0.00");
    }
    else {
        std::cout << std::setw(15) << "0.00";
        std::cout << std::setw(15) << (serial_timing.transmissionTime / DarkChannel::getLastTimingInfo().transmissionTime);
        std::cout << std::setw(15) << "inf";
    }
    std::cout << std::endl;

    // Compare transmission refinement
    std::cout << "Transmission Refinement    " << std::setw(15) << serial_timing.refinementTime;
    std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().refinementTime;
    if (cuda_available) {
        std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().refinementTime;
        std::cout << std::setw(15) << (serial_timing.refinementTime / DarkChannel::getLastTimingInfo().refinementTime);
        std::cout << std::setw(15) << (serial_timing.refinementTime > 0 ? "inf" : "0.00");
    }
    else {
        std::cout << std::setw(15) << "0.00";
        std::cout << std::setw(15) << (serial_timing.refinementTime / DarkChannel::getLastTimingInfo().refinementTime);
        std::cout << std::setw(15) << "inf";
    }
    std::cout << std::endl;

    // Compare scene reconstruction
    std::cout << "Scene Reconstruction       " << std::setw(15) << serial_timing.reconstructionTime;
    std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().reconstructionTime;
    if (cuda_available) {
        std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().reconstructionTime;
        std::cout << std::setw(15) << (serial_timing.reconstructionTime / DarkChannel::getLastTimingInfo().reconstructionTime);
        std::cout << std::setw(15) << (serial_timing.reconstructionTime > 0 ? "inf" : "0.00");
    }
    else {
        std::cout << std::setw(15) << "0.00";
        std::cout << std::setw(15) << (serial_timing.reconstructionTime / DarkChannel::getLastTimingInfo().reconstructionTime);
        std::cout << std::setw(15) << "inf";
    }
    std::cout << std::endl;

    // Compare total execution time
    std::cout << "Total Execution Time       " << std::setw(15) << serial_timing.totalTime;
    std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().totalTime;
    if (cuda_available) {
        std::cout << std::setw(15) << DarkChannel::getLastTimingInfo().totalTime;
        std::cout << std::setw(15) << (serial_timing.totalTime / DarkChannel::getLastTimingInfo().totalTime);
        std::cout << std::setw(15) << (serial_timing.totalTime > 0 ? "inf" : "0.00");
    }
    else {
        std::cout << std::setw(15) << "0.00";
        std::cout << std::setw(15) << (serial_timing.totalTime / DarkChannel::getLastTimingInfo().totalTime);
        std::cout << std::setw(15) << "inf";
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

    // Create named windows with the WINDOW_NORMAL flag to allow resizing
    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Serial Result", cv::WINDOW_NORMAL);
    cv::namedWindow("OpenMP Result", cv::WINDOW_NORMAL);
    if (cuda_available) {
        cv::namedWindow("CUDA Result", cv::WINDOW_NORMAL);
    }

    // Position the windows side by side
    cv::moveWindow("Original", 50, 50);
    cv::moveWindow("Serial Result", 50 + img.cols + 20, 50);
    cv::moveWindow("OpenMP Result", 50 + 2 * (img.cols + 20), 50);
    if (cuda_available) {
        cv::moveWindow("CUDA Result", 50 + 3 * (img.cols + 20), 50);
    }

    // Display the results
    cv::imshow("Original", img);
    cv::imshow("Serial Result", serial_result);
    cv::imshow("OpenMP Result", openmp_result);
    if (cuda_available) {
        cv::imshow("CUDA Result", cuda_result);
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

// Declare the CUDA function, check CUDA part
extern "C" void launchCudaPart();

int main(int argc, char** argv) {
    
    //Check cuda part
    std::cout << "Starting main program...\n";

    // Call the CUDA functionality
    launchCudaPart();

    std::cout << "Back in main program after CUDA.\n";
    // Check cuda part end
    //return 0;

    try {
        std::string project_path = "C:/Users/esther/source/repos/Dehaze/";

        // Check if we should use interactive mode
        bool interactive_mode = true;
        std::string img_path;

        if (argc > 1) {
            // If arguments provided, process the specified image
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
                std::cout << "  0: Exit" << std::endl;

                std::cout << "\nEnter your choice (0-3): ";
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
        std::cerr << "Exception: " << e.what() << std::endl;
        waitForKeyPress();
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        waitForKeyPress();
        return 1;
    }
}
