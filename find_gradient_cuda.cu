#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void ExtractHighGradient(uchar* im, cv::KeyPoint* kp,int rows, int cols, int pattern_row, int pattern_col){
    //NOTE THIS.......if there are a lot of pixel that index_x is larger than threadIdx.x_max, then they won't be reached
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_idy = threadIdx.y + blockIdx.y * blockDim.y;
    //int pattern = 3;
    int p = 7;//2*3+1
    int index;
    float gradient_x, gradient_y, norm;
    float max_norm = 0.0;
    if(thread_idx>pattern_col || thread_idy>pattern_row)
        return;
    // if(thread_idy == 0)
    //     printf("id x %d and pattern col is %d \n", thread_idx, pattern_col);
    for(int i = 0; i < p; i++)
        for(int j = 0; j< p; j++){
            index = i + p * (thread_idx+1) + (p*(thread_idy+1) + j)*cols;
            gradient_x = std::abs(im[index + 1] - im[index - 1]);//right and left
            gradient_y = std::abs(im[index + cols] - im[index - cols]);//down and up
            norm = sqrt(gradient_x*gradient_x + gradient_y*gradient_y);
            //push_to_key points
            if(norm>max_norm){
                kp[thread_idx + thread_idy*pattern_col].pt.y = p*(thread_idy+1) + j;
                kp[thread_idx + thread_idy*pattern_col].pt.x = p*(thread_idx+1) + i;
                max_norm = norm;
            }
        }
};

int main(int argc, char* argv[]){
    cv::Mat test;
    std::string add("../demo.png");
    if(argc != 2)
        std::cout<<"use default address "<<add.c_str()<<std::endl;
    else{
        add.clear();
        add.append(argv[1]);
    }
    //See https://stackoverflow.com/questions/25346395/why-is-the-first-cudamalloc-the-only-bottleneck
    //First cuda_realted function is responsible for initailize the cuda. It may cost about 100ms. once the initialization is done, copy data or do some other things shoudl be fast, so we just call a cudaFree(0) to initialize the device
    cudaFree(0);
    test = cv::imread(add, 0);//0 for gray and 1 for rgb
    cv::Mat im_with_keypoints;

    int rows = test.rows;
    int cols = test.cols;
    int p = 7;
    int row_pattern_num = rows / p + 1;
    int col_pattern_num = cols / p + 1;
    int keypoint_num = row_pattern_num * col_pattern_num;
    int image_size = sizeof(uchar) * rows * cols;
    int keypoint_size = sizeof(cv::KeyPoint) * keypoint_num;

    //printf("size of keypoint %ld \n", sizeof(cv::KeyPoint));

    double extract_time = (double)cv::getTickCount();
    //1 allocate memory in the host(CPU)
    uchar* host_input = (uchar *)malloc(image_size);
    cv::KeyPoint* host_output = (cv::KeyPoint*)malloc(keypoint_size);

    //2 assign value to the data (In reality, you get your data from a thousand ways)
    host_input = test.data;

    //3 allocate memory in the device (GPU)
    uchar* device_input;
    cv::KeyPoint* device_output;
    
    //double alloc_time = (double)cv::getTickCount();
    cudaMalloc((void**)&device_output, keypoint_size);
    cudaMalloc((void**)&device_input, image_size);
    
    //4 copy the data from the host to the device
    cudaMemcpy(device_input, host_input, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_output, host_output, keypoint_size, cudaMemcpyHostToDevice);

    //5 assign block number and thread number in the block
    // On my lenovo computer
    // Maximum number of threads per multiprocessor:  2048
    // Maximum number of threads per block:           1024
    // 6 multi processor
    dim3 block_number = dim3(4, 4);
    dim3 thread_number = dim3(32, 32);

    //6 call function in the device. During this step, results should be ready.
    ExtractHighGradient<<<block_number, thread_number>>>(device_input, device_output, rows, cols, row_pattern_num, col_pattern_num);//in cpp file '<<<' doesn't make sense and will lead to error

    cudaDeviceSynchronize();

    //7 copy memory from device to host
    cudaMemcpy(host_output, device_output, keypoint_size, cudaMemcpyDeviceToHost);

    extract_time = ((double)cv::getTickCount() - extract_time)/cv::getTickFrequency();
    std::cout<<"extract time is "<<extract_time<<std::endl;

    std::vector<cv::KeyPoint> hg_points(keypoint_num);
    for(int i = 0; i <keypoint_num; i++){
        hg_points[i] = host_output[i];
    }

    std::cout<<"point size "<<hg_points.size()<<std::endl;

    cv::drawKeypoints(test, hg_points, im_with_keypoints);

    cv::namedWindow("show key points", cv::WINDOW_AUTOSIZE);
    cv::imshow("show key points", im_with_keypoints);

    cv::waitKey(0);
    //8 free the momory in device and the host
    //free(host_input);//free host_input will leads to double free or corruption, maybe its because free host_input will free image.data too, which will make the image means nothing. Just a guess. No need to free here in fact, when the program is done, everthing will be freed
    free(host_output);
    cudaFree(device_input);
    cudaFree(device_output);
    return 1; 
}
