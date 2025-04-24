cmake_minimum_required(VERSION 3.18)
project(Dehaze LANGUAGES CXX CUDA)

# Set build type to Release for better performance
SET(CMAKE_BUILD_TYPE "Release")

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Example CUDA path (adjust to your installation)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
set(CUDA_BIN_PATH "${CUDA_TOOLKIT_ROOT_DIR}/bin")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIBRARY_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")

# Optionally force environment path
set(ENV{CUDA_PATH} ${CUDA_TOOLKIT_ROOT_DIR})



# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Gather all source files
file(GLOB Dehaze_src 
    "*.cpp" 
    "*.h" 
    "*.cu"
)

# Add executable with CUDA support
add_executable(Dehaze ${Dehaze_src})

# Set OpenCV directory
set(OpenCV_DIR "C:\\opencv\\build")
find_package(OpenCV REQUIRED core imgproc highgui)

# Link libraries
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(Dehaze ${OpenCV_LIBS} ${CUDA_LIBRARIES})
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Set CUDA compilation flags
set_target_properties(Dehaze PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Copy image files to build directory
file(GLOB IMG images/*)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/images)
foreach(img ${IMG})
    configure_file(${img} images/ COPYONLY)
endforeach()

# Add preprocessor definition to enable CUDA
# target_compile_definitions(Dehaze PRIVATE CUDA_ENABLED)
if(CUDA_FOUND)
    target_compile_definitions(Dehaze PRIVATE CUDA_ENABLED)
endif()


# Set architecture if needed
set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80;86")