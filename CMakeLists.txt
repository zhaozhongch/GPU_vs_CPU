cmake_minimum_required(VERSION 2.8)
project(OpencvTest)

# set(CMAKE_CXX_FLAGS
#    "-O3 -g -std=c++11 -march=native -pthread"
# )
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA QUIET REQUIRED)
#find_package(Threads REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

add_executable(find_gradient find_gradient.cpp)
target_link_libraries(find_gradient ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(find_gradient_multi_thread find_gradient_multi_thread.cpp)
target_link_libraries(find_gradient_multi_thread ${OpenCV_LIBS} Eigen3::Eigen -lpthread)

cuda_add_executable(find_gradient_cuda find_gradient_cuda.cu)
target_link_libraries(find_gradient_cuda ${OpenCV_LIBS} ${CUDA_LIBRARIES})
